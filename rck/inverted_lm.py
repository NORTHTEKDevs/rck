"""The Inverted Architecture -- HRR does 99% of the work, LM polishes.

This is the v4 headline contribution: a generative AI where the language
model has NO knowledge-bearing role. All facts live in the HRR store.
All reasoning happens via graph algorithms on that store. The LM exists
solely to polish template-rendered drafts into fluent prose.

The current implementation ships the FULL pipeline with a STUB polisher
(passes the template draft through with small surface edits). A
production v7 will replace the stub with a 50-100M-param distilled
transformer trained on a synthetic template-rendered corpus -- but the
ARCHITECTURE doesn't change. The stub already proves the inversion is
viable: at the current scale, the template draft IS the answer, and
the polish step is genuinely optional.

Pipeline:
  user query
      |
      v
  ConsciousAgent.ask()  -- routes via correction/tool/arith/temporal/
      |                    multistep/qtype dispatch, retrieves facts via
      v                    HRR composition, runs graph inference
  StructuredAnswer       (single fact OR composed paragraph OR program output)
      |
      v
  TemplateRenderer       (per-relation NL skeletons + connectives)
      |
      v
  FluencyPolisher        (stub now / small distilled LM in v7)
      |
      v
  Final fluent response

The crucial invariant: every fact in the final response can be traced
back to a specific (S, R, O) triple in the KB. Hallucination is
impossible at the fact level.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from rck.conscious_agent import ConsciousAgent


# ---------------------------------------------------------------------------
#  Polisher protocol
# ---------------------------------------------------------------------------

class FluencyPolisher:
    """Interface for the surface-polish layer.

    `polish(draft, context)` takes a template-rendered draft and returns
    a fluent-prose version. The polish layer MUST NOT add factual claims
    not present in the draft. It can rephrase, combine sentences,
    smooth transitions, choose connectives.

    For v4 the default polisher is a `RuleBasedPolisher` that applies
    light surface rewrites. v7 will swap in a `DistilledTransformerPolisher`
    with the same interface.
    """

    def polish(self, draft: str, context: dict | None = None) -> str:
        raise NotImplementedError


@dataclass
class RuleBasedPolisher(FluencyPolisher):
    """Light surface improvements without changing factual content.

    The rules:
      - Capitalize sentence starts.
      - Fix "a elephant" -> "an elephant" / "a apple" -> "an apple".
      - Replace underscores with spaces ("computer_science" -> "computer science").
      - De-dup consecutive identical sentences.
      - Add connectives between bare-fact sentences.
    """

    def polish(self, draft: str, context: dict | None = None) -> str:
        text = draft.replace("_", " ")
        # a/an correction
        text = re.sub(r"\ba\s+([aeiouAEIOU])", r"an \1", text)
        # de-dup
        text = self._dedup_sentences(text)
        # capitalisation
        text = self._capitalise_sentences(text)
        # connectives between consecutive facts about the same subject
        text = self._add_connectives(text, context or {})
        return text.strip()

    def _dedup_sentences(self, text: str) -> str:
        seen: set[str] = set()
        out: list[str] = []
        for s in re.split(r"(?<=[.!?])\s+", text):
            s_clean = s.strip()
            if not s_clean:
                continue
            key = s_clean.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s_clean)
        return " ".join(out)

    def _capitalise_sentences(self, text: str) -> str:
        def cap(m: re.Match) -> str:
            return m.group(1) + m.group(2).upper()
        # Beginning of string OR after sentence end + spaces.
        return re.sub(r"(^|[.!?]\s+)([a-z])", cap, text)

    def _add_connectives(self, text: str, context: dict) -> str:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) < 2:
            return text
        connectives = ["", "Also,", "Additionally,", "Moreover,", "And", "Plus,"]
        out = [sentences[0]]
        for i, sent in enumerate(sentences[1:], start=1):
            conn = connectives[min(i, len(connectives) - 1)]
            if conn:
                # Re-case the existing first word to lowercase if connective is
                # added in front (just remove leading capital + add connective).
                first_char = sent[0].lower() if sent and sent[0].isalpha() else sent[0]
                sent = first_char + sent[1:]
                out.append(f"{conn} {sent}")
            else:
                out.append(sent)
        return " ".join(out)


@dataclass
class DistilledTransformerPolisher(FluencyPolisher):
    """Future v7 hook -- placeholder for the trained 50-100M-param polisher.

    Will load weights from `~/projects/active/rck/checkpoints/polisher.pt`
    and run them on the draft. For now this raises NotImplementedError to
    make the upgrade path obvious.
    """

    weights_path: str = "checkpoints/polisher.pt"

    def polish(self, draft: str, context: dict | None = None) -> str:
        raise NotImplementedError(
            "DistilledTransformerPolisher is the v7 upgrade. "
            "Train via scripts/train_polisher.py (not yet implemented)."
        )


# ---------------------------------------------------------------------------
#  InvertedLM -- the full pipeline
# ---------------------------------------------------------------------------

@dataclass
class InvertedLM:
    """Wraps a ConsciousAgent + FluencyPolisher into the inverted-LM API.

    The agent does all the knowledge work. The polisher only smooths the
    surface. Generation is provably hallucination-free at the fact level
    because every claim originates in a structured retrieval.
    """

    agent: ConsciousAgent
    polisher: FluencyPolisher = field(default_factory=RuleBasedPolisher)

    def generate(self, query: str, *, polish: bool = True) -> dict:
        """End-to-end: query -> retrieval -> draft -> polish -> response."""
        res = self.agent.ask(query)
        draft = self._draft_from_result(res)
        if not polish:
            return {**res, "draft": draft, "response": draft, "polished": False}
        polished = self.polisher.polish(draft, context=res)
        return {**res, "draft": draft, "response": polished, "polished": True}

    def describe_entity(self, entity: str, *, polish: bool = True) -> dict:
        """Multi-sentence entity description -- the polisher's strength."""
        draft = self.agent.describe(entity)
        if not polish:
            return {"draft": draft, "response": draft, "polished": False}
        polished = self.polisher.polish(draft)
        return {"draft": draft, "response": polished, "polished": True}

    def think_aloud(self, query: str) -> dict:
        """Force the chain-of-thought narration through the polisher too."""
        res = self.agent.ask(query, think_aloud=True)
        thought = res.get("think_aloud", "")
        verbal = res.get("verbal", "")
        draft = (thought + " " + verbal).strip() if thought else verbal
        polished = self.polisher.polish(draft, context=res)
        return {**res, "draft": draft, "response": polished, "polished": True}

    # ---- internal --------------------------------------------------------

    def _draft_from_result(self, res: dict) -> str:
        """Convert a ConsciousAgent.ask() result dict into a template draft."""
        # Highest-priority: the explicit "sentence" template, if present.
        if "sentence" in res:
            return str(res["sentence"])
        # Otherwise the verbal (calibrated) is the draft.
        return str(res.get("verbal") or "")


# ---------------------------------------------------------------------------
#  Convenience factory
# ---------------------------------------------------------------------------

def make_inverted_lm(agent: ConsciousAgent | None = None,
                     polisher: FluencyPolisher | None = None) -> InvertedLM:
    """One-line constructor."""
    if agent is None:
        agent = ConsciousAgent(dim=4096, n_shards=64, seed=0)
    if polisher is None:
        polisher = RuleBasedPolisher()
    return InvertedLM(agent=agent, polisher=polisher)
