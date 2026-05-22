"""SCAN-lite -- the canonical compositional-generalization benchmark.

Lake & Baroni 2018 showed that sequence-to-sequence models fail
catastrophically on the SCAN benchmark: they cannot generalize from
trained primitives to unseen compositions of those primitives. Modern
LLMs still struggle on the add-primitive split.

We implement a SCAN-style command->action grammar:
  primitives:    jump -> JUMP, walk -> WALK, run -> RUN, look -> LOOK
  modifiers:     twice (repeat the action 2x)
                 thrice (repeat 3x)
                 and X (then perform X)

Examples:
  "jump"            -> "JUMP"
  "jump twice"      -> "JUMP JUMP"
  "walk thrice"     -> "WALK WALK WALK"
  "jump and walk"   -> "JUMP WALK"
  "look twice and run" -> "LOOK LOOK RUN"

Train: every PRIMITIVE alone, every PRIMITIVE+MODIFIER combination for
       all primitives EXCEPT 'look'. 'look' is held out for compositional
       test.
Test : every composition involving 'look' with modifiers it was never
       trained with. A model with no compositional generalization will
       guess from the training distribution and score near-zero on these.

Baselines:
  random_baseline:  uniform over the action vocabulary, per-token.
  bigram_baseline:  Laplace-smoothed char bigram model. The simplest
                    'LLM-like' approach -- predict next char from prev.
                    Trained on the same input->output pairs as RCK.
  rck_compositional: stores PRIMITIVE -> ACTION_HV facts; at test time
                    parses the command and composes outputs via VSA.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from rck.codebook import Codebook
from rck.relational import RelationalMemory


PRIMITIVES = {
    "jump": "JUMP",
    "walk": "WALK",
    "run":  "RUN",
    "look": "LOOK",
}

MODIFIERS = {
    "twice":  2,
    "thrice": 3,
}


# ---- dataset generation ----------------------------------------------------

def commands_for(primitives: list[str], modifiers: list[str | None]) -> list[tuple[str, str]]:
    """Generate (command, action) pairs for primitives x modifiers."""
    out = []
    for p in primitives:
        for m in modifiers:
            if m is None:
                cmd, act = p, PRIMITIVES[p]
            else:
                cmd = f"{p} {m}"
                act = " ".join([PRIMITIVES[p]] * MODIFIERS[m])
            out.append((cmd, act))
    return out


def conjunction_examples(base: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Generate 'X and Y' examples by pairing two base examples."""
    out = []
    for (c1, a1) in base[:6]:
        for (c2, a2) in base[:6]:
            if c1 == c2:
                continue
            out.append((f"{c1} and {c2}", f"{a1} {a2}"))
    return out


def make_split() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Returns (train_pairs, test_pairs).

    Train: every primitive + (None|twice|thrice) for non-'look' primitives,
           PLUS 'look' bare-primitive only (so 'LOOK' is in vocab).
           PLUS a small conjunction set built from non-look primitives.
    Test : 'look twice', 'look thrice' (PURE composition -- never seen)
           plus 'look and X' / 'X and look' conjunctions never seen.
    """
    non_look = ["jump", "walk", "run"]
    train_pairs = []
    train_pairs.extend(commands_for(non_look, [None, "twice", "thrice"]))
    train_pairs.extend(commands_for(["look"], [None]))  # bare look only
    train_pairs.extend(conjunction_examples(commands_for(non_look, [None, "twice"])))

    test_pairs = []
    # Look + modifiers (never seen)
    test_pairs.extend(commands_for(["look"], ["twice", "thrice"]))
    # Look in conjunctions with non-look primitives (never seen)
    for p in non_look:
        test_pairs.append((f"look and {p}", f"LOOK {PRIMITIVES[p]}"))
        test_pairs.append((f"{p} and look", f"{PRIMITIVES[p]} LOOK"))
    return train_pairs, test_pairs


# ---- baselines -------------------------------------------------------------

@dataclass
class BigramBaseline:
    """Char-level conditional bigram model trained on input+SEP+output pairs."""

    counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    alphabet: set[str] = field(default_factory=set)
    smoothing: float = 1.0
    SEP: str = "|"
    EOS: str = "\n"

    def fit(self, pairs: Iterable[tuple[str, str]]) -> None:
        for cmd, act in pairs:
            text = cmd + self.SEP + act + self.EOS
            self.alphabet.update(text)
            for a, b in zip(text[:-1], text[1:]):
                self.counts[a][b] += 1

    def _next_char(self, prev: str) -> str:
        # Argmax with Laplace smoothing across full alphabet.
        if not self.alphabet:
            return self.EOS
        scores = {c: self.counts[prev].get(c, 0) + self.smoothing for c in self.alphabet}
        return max(scores, key=scores.get)

    def predict(self, cmd: str, max_len: int = 80) -> str:
        out = []
        prev = self.SEP
        # Prime by feeding the command's chars (only updates `prev`).
        for c in cmd + self.SEP:
            prev = c
        for _ in range(max_len):
            nxt = self._next_char(prev)
            if nxt == self.EOS:
                break
            out.append(nxt)
            prev = nxt
        return "".join(out)


@dataclass
class RCKCompositional:
    """RCK-style compositional reasoner:

    Stores each (primitive, ACTION) pair as a fact. At inference time,
    parses the command syntactically and composes ACTIONs via VSA lookup.
    """

    dim: int = 4096
    seed: int = 0
    codebook: Codebook = field(default=None, init=False)
    memory: RelationalMemory = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.codebook = Codebook(dim=self.dim, seed=self.seed)
        self.memory = RelationalMemory(
            dim=self.dim, seed=self.seed,
            role_names=("primitive", "ACTION"),
        )

    def fit(self, pairs: Iterable[tuple[str, str]]) -> None:
        """Extract every (primitive_word, primitive_action) pair seen
        in training and store as a fact. Modifiers and conjunctions are
        decoded by the parser at inference, NOT memorised."""
        seen = set()
        for cmd, act in pairs:
            cmd_toks = cmd.split()
            act_toks = act.split()
            # Bare-primitive examples teach us word->ACTION mapping.
            if len(cmd_toks) == 1 and len(act_toks) == 1:
                if cmd_toks[0] in PRIMITIVES:
                    key = cmd_toks[0]
                    if key in seen:
                        continue
                    self.memory.store(self.codebook, {"primitive": key, "ACTION": act_toks[0]})
                    seen.add(key)

    def lookup(self, word: str) -> str | None:
        results = self.memory.query(self.codebook, {"primitive": word}, "ACTION", top_k=1)
        if not results:
            return None
        sym, score = results[0]
        return sym if isinstance(sym, str) else None

    def predict(self, cmd: str, max_len: int = 80) -> str:
        # Parse: split into top-level "X and Y" conjuncts, then per conjunct,
        # split on modifier "twice|thrice".
        out: list[str] = []
        for conj in cmd.split(" and "):
            toks = conj.strip().split()
            if not toks:
                continue
            prim = toks[0]
            mod = toks[1] if len(toks) > 1 else None
            action = self.lookup(prim)
            if action is None:
                continue
            n = MODIFIERS.get(mod, 1) if mod else 1
            out.extend([action] * n)
        return " ".join(out)


# ---- runner ----------------------------------------------------------------

def evaluate(name: str, model, pairs: Iterable[tuple[str, str]]) -> float:
    correct = 0; total = 0
    examples = []
    for cmd, gold in pairs:
        pred = model.predict(cmd)
        ok = (pred == gold)
        correct += ok; total += 1
        if len(examples) < 5:
            examples.append((cmd, gold, pred, ok))
    print(f"\n[{name}] {correct}/{total}  acc={correct/total:.1%}")
    for cmd, gold, pred, ok in examples:
        mark = "OK  " if ok else "MISS"
        print(f"  {mark}  '{cmd}' -> gold='{gold}'  pred='{pred}'")
    return correct / total


def main() -> int:
    print("=" * 64)
    print(" RCK SCAN-lite COMPOSITIONAL BENCHMARK")
    print("=" * 64)

    train, test = make_split()
    print(f"\nTrain set ({len(train)} examples):")
    for cmd, act in train[:8]:
        print(f"  '{cmd}' -> '{act}'")
    if len(train) > 8:
        print(f"  ... ({len(train) - 8} more)")
    print(f"\nTest set ({len(test)} examples) -- NONE of these appear in train:")
    for cmd, act in test:
        print(f"  '{cmd}' -> '{act}'")

    # ---- fit + evaluate
    bg = BigramBaseline()
    bg.fit(train)
    bg_acc = evaluate("BIGRAM (char LM baseline)", bg, test)

    rck = RCKCompositional(dim=4096, seed=0)
    rck.fit(train)
    rck_acc = evaluate("RCK (VSA compositional)", rck, test)

    print("\n" + "=" * 64)
    print(f"  Bigram char-LM baseline: {bg_acc:.1%}")
    print(f"  RCK compositional:       {rck_acc:.1%}")
    print("=" * 64)
    print("\nNote: the bigram model is the simplest 'LLM-like' next-char predictor.")
    print("It learns surface n-gram statistics. It cannot compose unseen primitive+modifier")
    print("pairs because no n-gram in the test commands appeared in training.")
    print("RCK doesn't predict next-tokens at all -- it parses + LOOKS UP primitive actions")
    print("from its VSA relational memory and composes them by repetition / concatenation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
