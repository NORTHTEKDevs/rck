"""GenerativeRCK -- a real generative AI built from RCK primitives.

This is the top-level user-facing model that ties everything together:

  * **Word-level tokens** instead of chars (via `rck/tokenizer.py`).
  * A **statistical generation path** -- RCKAgent over word tokens for
    fluent continuation when the input is open-ended.
  * A **structured knowledge path** -- RelationalMemory storing
    (subject, relation, object) facts that can be retrieved exactly.
  * A **question parser** that detects "what / who / where / when / why /
    how / which" questions and routes them to the structured path.
  * A **dialogue trainer** that ingests Q/A pairs and learns
    question-shape -> answer-shape patterns.

The model exposes three primary verbs:

  ingest(text)         -- learn the text statistically.
  tell(s, r, o)        -- store an exact fact.
  ask(question)        -- answer a question (structured if possible,
                          generative fallback otherwise).
  generate(prompt, n)  -- free-form n-word continuation.

This is the missing piece between "RCK can compose unseen multi-slot
queries" and "RCK is a generative AI you can talk to."
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from rck.agent import RCKAgent
from rck.codebook import Codebook
from rck.relational import RelationalMemory
from rck.tokenizer import detokenize, sentences, tokenize


# ---------------------------------------------------------------------------
#  Question parsing
# ---------------------------------------------------------------------------

# Maps the WH-word -> the relation slot we expect a "be"-form sentence to fill.
WH_TO_RELATION = {
    "what": "is",
    "who":  "is",
    "where": "in",
    "when": "when",
    "why":  "because",
    "how":  "how",
    "which": "is",
}

# Stop words we strip when extracting the ENTITY (the subject we ask about).
# Note: attribute words like "color" must NOT be in this set, because they
# are legitimate attribute names elsewhere.
_ENTITY_STOP = {"a", "an", "the", "of", "do", "does", "did", "is", "are",
                "was", "were", "in", "on", "live", "lives", "located", "to"}

# A small set of verb tokens to strip from the entity slot of "where" queries.
_WHERE_VERBS = {"live", "lives", "lived", "is", "are", "was", "were",
                "located", "stays"}

_STOP = _ENTITY_STOP  # backwards-compatible alias for internal use


def parse_question(q: str) -> tuple[str, str, str | None] | None:
    """Best-effort extraction of (entity, relation, value-slot) from a
    natural-language question. Returns None if no clear pattern matches.

    Examples (token-lowered):
      "what color is the sky"          -> ("sky", "color", None)
      "what is the capital of france"  -> ("france", "capital", None)
      "who wrote hamlet"               -> ("hamlet", "author", None)
      "where does alice live"          -> ("alice", "lives_in", None)
    """
    toks = tokenize(q)
    if not toks:
        return None
    # Drop trailing punctuation.
    while toks and toks[-1] in {"?", "."}:
        toks.pop()
    if not toks:
        return None
    wh = toks[0]
    if wh not in WH_TO_RELATION:
        return None
    body = toks[1:]

    # Heuristic 1: "what <attr> is the <entity>"  (attr=color, kind, type, ...)
    if wh in ("what", "which") and len(body) >= 4 and body[1] == "is":
        attr = body[0]
        entity_toks = [t for t in body[2:] if t not in _STOP]
        if entity_toks and attr not in _STOP:
            return (entity_toks[-1], attr, None)

    # Heuristic 2: "what is the <attr> of <entity>"
    if wh == "what" and "of" in body and "is" in body:
        of_idx = body.index("of")
        is_idx = body.index("is")
        attr_toks = [t for t in body[is_idx + 1:of_idx] if t not in _STOP]
        entity_toks = [t for t in body[of_idx + 1:] if t not in _STOP]
        if attr_toks and entity_toks:
            return (entity_toks[-1], attr_toks[-1], None)

    # Heuristic 3: "who wrote <work>" / "who invented <thing>" etc.
    if wh == "who" and len(body) >= 2 and body[0] not in _STOP:
        verb = body[0]
        obj_toks = [t for t in body[1:] if t not in _STOP]
        if obj_toks:
            return (obj_toks[-1], verb, None)

    # Heuristic 3b: "what does <entity> have" -> (entity, "has", _)
    if wh == "what" and len(body) >= 3 and body[0] == "does" and body[-1] in {"have", "has"}:
        entity_toks = [t for t in body[1:-1] if t not in _ENTITY_STOP]
        if entity_toks:
            return (entity_toks[-1], "has", None)

    # Heuristic 3c: "what is <entity> made of" -> (entity, "madeof", _)
    if wh == "what" and len(body) >= 3 and "made" in body and "of" in body:
        # Strip leading "is the", "is", etc.
        body_clean = [t for t in body if t not in {"is", "the", "a", "an", "of", "made"}]
        if body_clean:
            return (body_clean[-1], "madeof", None)

    # Heuristic 3d: "what causes <X>" -> (X, causedby, _) [we have the inverse stored]
    if wh == "what" and len(body) >= 2 and body[0] == "causes":
        entity_toks = [t for t in body[1:] if t not in _ENTITY_STOP]
        if entity_toks:
            return (entity_toks[-1], "causedby", None)

    # Heuristic 4: "where does <entity> live" / "where is <entity>"
    if wh == "where" and len(body) >= 2:
        entity_toks = [t for t in body if t not in _ENTITY_STOP
                       and t not in _WHERE_VERBS]
        if entity_toks:
            return (entity_toks[-1], "lives_in" if any(v in body for v in _WHERE_VERBS) else "in", None)

    # Fallback: take the last content token as the entity, and WH-to-relation.
    content = [t for t in body if t not in _STOP]
    if content:
        return (content[-1], WH_TO_RELATION[wh], None)
    return None


# ---------------------------------------------------------------------------
#  Generative model
# ---------------------------------------------------------------------------

@dataclass
class GenerativeRCK:
    """The user-facing generative model.

    Args:
        dim:           hypervector dimensionality. 4096 is a good default;
                       8192 if you plan to store thousands of facts.
        seed:          random seed (also used for codebook + roles).
        vocab_size:    max input alphabet for the LM agent (auto-grows up
                       to this cap).
        reservoir_dim: LSM reservoir size.
        n_columns:     number of cortical columns for the LM.
        bigram_order:  n-gram order for the bigram associative memory.
    """

    dim: int = 4096
    seed: int = 0
    vocab_size: int = 8192
    reservoir_dim: int = 128
    n_columns: int = 2
    bigram_order: int = 2

    codebook: Codebook = field(default=None, init=False)
    memory: RelationalMemory = field(default=None, init=False)
    lm: RCKAgent = field(default=None, init=False)

    _fact_count: int = field(default=0, init=False)
    _ingest_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        # NOTE: the LM agent constructs its own codebook + relational memory
        # internally; we keep our own shared codebook + memory for facts.
        # We DO NOT share the LM's codebook with the relational memory --
        # facts can be retrieved without any LM token-state.
        self.codebook = Codebook(dim=self.dim, seed=self.seed)
        self.memory = RelationalMemory(
            dim=self.dim, seed=self.seed,
            role_names=("S", "R", "O"),
        )
        self.lm = RCKAgent(
            vocab_size=self.vocab_size, hv_dim=self.dim,
            n_columns=self.n_columns, reservoir_dim=self.reservoir_dim,
            n_clauses=16, fep_rank=64, bigram_order=self.bigram_order,
            seed=self.seed + 1,
        )

    # ---- knowledge ingestion ----------------------------------------------

    def tell(self, subject: str, relation: str, obj: str) -> None:
        """Store a structured (S, R, O) fact in relational memory."""
        s = subject.lower().strip()
        r = relation.lower().strip()
        o = obj.lower().strip()
        self.memory.store(self.codebook, {"S": s, "R": r, "O": o})
        self._fact_count += 1

    def tell_many(self, triples: Iterable[tuple[str, str, str]]) -> None:
        for s, r, o in triples:
            self.tell(s, r, o)

    def ingest(self, text: str) -> dict:
        """Statistically learn the text via the LM. Also opportunistically
        extracts simple "X is Y" / "X has Y" patterns as facts.
        """
        ingested_facts = 0
        for sent in sentences(text):
            facts = _extract_simple_facts(sent)
            for s, r, o in facts:
                self.tell(s, r, o)
                ingested_facts += 1
            tokens = tokenize(sent)
            if tokens:
                self.lm.observe(tokens, learn=True)
                self._ingest_count += len(tokens)
        return {"tokens_seen": self._ingest_count, "new_facts": ingested_facts}

    # ---- question answering ----------------------------------------------

    def ask(self, question: str, top_k: int = 3) -> dict:
        """Answer a natural-language question.

        Returns a dict with the answer, candidates, source path
        (structured / generated / unknown) and confidence.
        """
        parsed = parse_question(question)
        if parsed is None:
            return self._generative_fallback(question)
        entity, relation, _ = parsed

        # Multi-relation cascade: a "what color is X" question maps to a
        # `color` relation slot, but the sentence "The X is Y" stores it
        # under `is`. We try the specific relation first, then "is", then
        # the reverse subject lookup.
        relation_candidates = [relation]
        if relation != "is":
            relation_candidates.append("is")
        for rel in relation_candidates:
            results = self.memory.query(
                self.codebook,
                {"S": entity, "R": rel},
                "O",
                top_k=top_k,
            )
            if results and results[0][1] > 0.10:
                top = results[0]
                return {
                    "answer": str(top[0]),
                    "confidence": float(top[1]),
                    "source": "structured" if rel == relation else f"structured-via-{rel}",
                    "candidates": [(str(s), float(c)) for s, c in results],
                    "parsed": {"entity": entity, "relation": rel},
                }

        # Reverse-direction lookup as a last structured try.
        results = self.memory.query(
            self.codebook,
            {"R": relation, "O": entity},
            "S",
            top_k=top_k,
        )
        if results and results[0][1] > 0.10:
            top = results[0]
            return {
                "answer": str(top[0]),
                "confidence": float(top[1]),
                "source": "structured-reverse",
                "candidates": [(str(s), float(c)) for s, c in results],
                "parsed": {"entity": entity, "relation": relation},
            }
        # Fallback to LM generation.
        return self._generative_fallback(question)

    def _generative_fallback(self, prompt: str) -> dict:
        emitted = self.generate(prompt, max_words=16)
        return {
            "answer": emitted,
            "confidence": 0.0,
            "source": "generated",
            "candidates": [],
            "parsed": None,
        }

    # ---- free-form generation --------------------------------------------

    def generate(self, prompt: str, max_words: int = 24,
                 temperature: float = 0.3) -> str:
        """Word-level LM continuation."""
        tokens = tokenize(prompt)
        if not tokens:
            return ""
        self.lm.stochastic_decode = temperature > 1e-4
        self.lm.fep.temperature = max(temperature, 1e-3)
        emitted, _ = self.lm.generate(tokens, max_new=max_words)
        return detokenize([str(t) for t in emitted])

    # ---- introspection ---------------------------------------------------

    def state(self) -> dict:
        return {
            "version": "1.2.0",
            "fact_count": self._fact_count,
            "tokens_ingested": self._ingest_count,
            "codebook_size": self.codebook.size(),
            "lm_codebook_size": self.lm.codebook.size(),
            "memory_facts": self.memory.size(),
        }


# ---------------------------------------------------------------------------
#  Naive fact extraction
# ---------------------------------------------------------------------------

# Patterns we recognise on ingest. They MUST be conservative -- false
# positives are worse than missing facts because they corrupt the memory.
_BE_RE = re.compile(r"^the\s+(\w+)\s+is\s+(\w+)\s*\.?$", re.IGNORECASE)
_HAS_RE = re.compile(r"^the\s+(\w+)\s+has\s+(\w+(?:\s+\w+){0,3})\s*\.?$", re.IGNORECASE)
_OF_RE = re.compile(r"^the\s+(\w+)\s+of\s+(?:the\s+)?(\w+)\s+is\s+(\w+(?:\s+\w+){0,3})\s*\.?$", re.IGNORECASE)
_LIVES_RE = re.compile(r"^(\w+)\s+lives\s+in\s+(\w+(?:\s+\w+){0,2})\s*\.?$", re.IGNORECASE)
_WROTE_RE = re.compile(r"^(\w+)\s+wrote\s+(\w+(?:\s+\w+){0,3})\s*\.?$", re.IGNORECASE)


def _extract_simple_facts(sent: str) -> list[tuple[str, str, str]]:
    """Pull (S, R, O) triples from a sentence using lightweight regexes."""
    s = sent.strip()
    out: list[tuple[str, str, str]] = []
    m = _OF_RE.match(s)
    if m:
        attr, entity, value = m.group(1), m.group(2), m.group(3)
        out.append((entity.lower(), attr.lower(), _last_word(value)))
        return out
    m = _BE_RE.match(s)
    if m:
        entity, value = m.group(1), m.group(2)
        out.append((entity.lower(), "is", value.lower()))
        return out
    m = _HAS_RE.match(s)
    if m:
        entity, value = m.group(1), m.group(2)
        out.append((entity.lower(), "has", _last_word(value)))
        return out
    m = _LIVES_RE.match(s)
    if m:
        entity, value = m.group(1), m.group(2)
        out.append((entity.lower(), "lives_in", _last_word(value)))
        return out
    m = _WROTE_RE.match(s)
    if m:
        author, work = m.group(1), m.group(2)
        out.append((_last_word(work), "wrote", author.lower()))
        out.append((author.lower(), "wrote", _last_word(work)))
        return out
    return out


def _last_word(s: str) -> str:
    return s.strip().split()[-1].lower()
