"""Adversarial test generator.

Builds queries designed to stress every part of the system:

  1. Negation: "What is NOT a fruit?"
  2. Multi-hop with red herrings: "What is the capital of the country
     where the person who wrote Hamlet lived?"
  3. Confidence stress: questions with known low-confidence facts
  4. Polysemy: queries on entities that have multiple meanings
  5. Composition: queries combining seen primitives in unseen ways
  6. Confusion: queries with similar but distinct entities
  7. Long-form: questions requiring multi-paragraph answers
  8. Contradiction: queries where two stored facts conflict

Use for: self-evaluation, regression testing, training data.
"""
from __future__ import annotations

from dataclasses import dataclass

from rck.knowledge_base import ShardedKnowledgeBase


@dataclass
class AdversarialCase:
    query: str
    category: str
    expected_behaviour: str    # "answer" / "hedge" / "refuse" / "ask"
    expected_answer: str | None = None


def gen_negation_cases(kb: ShardedKnowledgeBase, n: int = 5) -> list[AdversarialCase]:
    """Negation queries: pick an entity not in a category."""
    out: list[AdversarialCase] = []
    # Find entities that are NOT mammals.
    for cat in ("mammal", "bird", "fish", "reptile", "fruit", "vegetable"):
        out.append(AdversarialCase(
            query=f"What is something that is NOT a {cat}?",
            category="negation",
            expected_behaviour="answer",
        ))
        if len(out) >= n:
            break
    return out


def gen_compound_cases(kb: ShardedKnowledgeBase, n: int = 5) -> list[AdversarialCase]:
    """Compound multi-hop with red herrings."""
    return [
        AdversarialCase(
            query="What is the continent of the country whose capital is paris?",
            category="compound",
            expected_behaviour="answer",
            expected_answer="europe",
        ),
        AdversarialCase(
            query="What is the field of the person who wrote hamlet?",
            category="compound",
            expected_behaviour="answer",
        ),
        AdversarialCase(
            query="What is the capital of the country where einstein was born?",
            category="compound",
            expected_behaviour="answer",
        ),
        AdversarialCase(
            query="What is the language of the country whose capital is berlin?",
            category="compound",
            expected_behaviour="answer",
            expected_answer="german",
        ),
        AdversarialCase(
            query="What is the founder of the company that makes ipads?",
            category="compound",
            expected_behaviour="answer",
        ),
    ][:n]


def gen_polysemy_cases() -> list[AdversarialCase]:
    """Same-name entities."""
    return [
        AdversarialCase(
            query="Tell me about paris.",
            category="polysemy",
            expected_behaviour="ask",
        ),
        AdversarialCase(
            query="What is mercury?",
            category="polysemy",  # planet vs element
            expected_behaviour="ask",
        ),
        AdversarialCase(
            query="What is jupiter?",
            category="polysemy",  # planet vs god
            expected_behaviour="ask",
        ),
    ]


def gen_confusion_cases() -> list[AdversarialCase]:
    """Similar but distinct entities."""
    return [
        AdversarialCase(
            query="What is the diet of the leopard?",
            category="confusion",
            expected_behaviour="answer",
        ),
        AdversarialCase(
            query="What is the diet of the leopard frog?",
            category="confusion",
            expected_behaviour="answer",
        ),
    ]


def gen_contradiction_cases() -> list[AdversarialCase]:
    """Queries on contradictory facts."""
    return [
        AdversarialCase(
            query="Is the sky red?",
            category="contradiction",
            expected_behaviour="refuse",
        ),
        AdversarialCase(
            query="Is the dog a fish?",
            category="contradiction",
            expected_behaviour="refuse",
        ),
    ]


def generate_test_set(kb: ShardedKnowledgeBase, *,
                      per_category: int = 5) -> list[AdversarialCase]:
    """Bundle them all into a representative adversarial test set."""
    out: list[AdversarialCase] = []
    out.extend(gen_negation_cases(kb, n=per_category))
    out.extend(gen_compound_cases(kb, n=per_category))
    out.extend(gen_polysemy_cases())
    out.extend(gen_confusion_cases())
    out.extend(gen_contradiction_cases())
    return out
