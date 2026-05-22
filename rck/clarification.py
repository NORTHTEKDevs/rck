"""Clarification + counter-question logic.

When a query is ambiguous, RCK should ASK rather than guess. The
ambiguity signals we detect:

  1. Multiple top-K candidates with similar cosine -- "did you mean X or Y?"
  2. Multiple possible parses of the question -- "are you asking about X or Y?"
  3. Pronoun reference with multiple recent entities -- "by 'it' do you mean A, B, or C?"
  4. Multi-meaning entity -- "paris" the city vs person -- "which paris?"

LLMs almost always commit to one interpretation silently. RCK can
defer and ask, giving the user control over what's actually answered.
"""
from __future__ import annotations

from dataclasses import dataclass

from rck.knowledge_base import ShardedKnowledgeBase


@dataclass
class ClarificationRequest:
    reason: str
    candidates: list[str]
    question: str

    def is_needed(self) -> bool:
        return len(self.candidates) >= 2


def detect_ambiguous_top_k(
    results: list[tuple[str, float]], *,
    ratio_threshold: float = 0.85,
) -> ClarificationRequest | None:
    """If top-K candidates have very similar cosines, ask which one."""
    if len(results) < 2:
        return None
    top = results[0]
    runners_up = [(s, c) for s, c in results[1:5]
                  if c >= top[1] * ratio_threshold]
    if not runners_up:
        return None
    candidates = [str(top[0])] + [str(s) for s, _ in runners_up]
    return ClarificationRequest(
        reason="similar_confidence",
        candidates=candidates,
        question=("I see multiple plausible answers: "
                  + ", ".join(candidates[:4])
                  + ". Which one did you mean?"),
    )


def detect_ambiguous_entity(
    kb: ShardedKnowledgeBase, name: str,
    *, top_k: int = 4,
) -> ClarificationRequest | None:
    """If a name has multiple isa parents (paris=city + paris=person),
    ask which sense."""
    parents = kb.query({"S": name.lower(), "R": "isa"}, "O", top_k=top_k)
    parents = [(str(s), float(c)) for s, c in parents if c >= 0.10]
    if len(parents) < 2:
        return None
    candidates = [f"{name} the {p}" for p, _ in parents]
    return ClarificationRequest(
        reason="entity_polysemy",
        candidates=candidates,
        question=(f"'{name}' could mean: "
                  + ", ".join(candidates)
                  + ". Which sense?"),
    )


def detect_pronoun_ambiguity(
    pronoun: str, recent_entities: list[str],
) -> ClarificationRequest | None:
    """If 'it' could plausibly refer to several recent entities, ask."""
    if len(recent_entities) < 2:
        return None
    return ClarificationRequest(
        reason="pronoun_ambiguity",
        candidates=recent_entities[:4],
        question=(f"By '{pronoun}' do you mean "
                  + ", ".join(recent_entities[:3])
                  + "?"),
    )
