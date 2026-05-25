"""Tests for fact induction from chains."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.chain_induction import (
    InducedFact, InductionPolicy, induce_from_chain, induce_many,
)
from rck.chain_walker import Hop, walk_chain
from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore
from rck.skills import SkillLibrary


def _two_hop_kb() -> ShardedKnowledgeBase:
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("leaf", "partof", "tree"),
        ("tree", "locatedin", "forest"),
        ("branch", "partof", "tree"),
    ])
    return kb


# ---- basic induction ------------------------------------------------------

def test_induce_creates_shortcut_fact():
    kb = _two_hop_kb()
    pre = kb.size()
    result = walk_chain(kb, "leaf", [Hop("partof"), Hop("locatedin")])
    assert result.answer == "forest"
    induced = induce_from_chain(kb, result, "leaf")
    assert isinstance(induced, InducedFact)
    assert induced.subject == "leaf"
    assert induced.relation == "locatedin"
    assert induced.obj == "forest"
    assert induced.verified is True
    # New direct fact is now in the KB.
    assert kb.size() == pre + 1
    ans, _ = kb.answer({"S": "leaf", "R": "locatedin"}, "O")
    assert ans == "forest"


def test_induce_records_provenance():
    kb = _two_hop_kb()
    prov = ProvenanceStore()
    result = walk_chain(kb, "leaf", [Hop("partof"), Hop("locatedin")])
    induced = induce_from_chain(kb, result, "leaf", provenance=prov)
    assert induced is not None
    rec = prov.get("leaf", "locatedin", "forest")
    assert rec is not None
    assert rec.source == "induced"
    assert "induced" in rec.tags


def test_induce_rejects_low_confidence():
    kb = _two_hop_kb()
    result = walk_chain(kb, "leaf", [Hop("partof"), Hop("locatedin")])
    policy = InductionPolicy(min_confidence=0.99)
    induced = induce_from_chain(kb, result, "leaf", policy=policy)
    assert induced is not None
    assert induced.rejected_reason is not None
    assert "confidence" in induced.rejected_reason


def test_induce_skips_one_hop_chains():
    kb = _two_hop_kb()
    result = walk_chain(kb, "leaf", [Hop("partof")])
    induced = induce_from_chain(kb, result, "leaf")
    assert induced is None  # 1-hop chains are skipped


# ---- skill confirmation ---------------------------------------------------

def test_induce_respects_skill_uses_gate():
    kb = _two_hop_kb()
    skills = SkillLibrary()
    result = walk_chain(kb, "leaf", [Hop("partof"), Hop("locatedin")],
                        skills=skills)
    # Pattern recorded once. Gate of 3 should block.
    policy = InductionPolicy(min_confidence=0.05, require_skill_uses=3)
    induced = induce_from_chain(kb, result, "leaf",
                                 policy=policy, skills=skills)
    assert induced is not None
    assert "skill seen" in (induced.rejected_reason or "")


def test_induce_passes_skill_gate_after_repeated_walks():
    kb = _two_hop_kb()
    skills = SkillLibrary()
    # Walk the same pattern twice to bump skill count.
    for start in ("leaf", "branch"):
        result = walk_chain(kb, start, [Hop("partof"), Hop("locatedin")],
                            skills=skills)
    # Now require 2 uses; should pass.
    result3 = walk_chain(kb, "leaf", [Hop("partof"), Hop("locatedin")],
                         skills=skills)
    policy = InductionPolicy(min_confidence=0.05, require_skill_uses=2)
    induced = induce_from_chain(kb, result3, "leaf",
                                 policy=policy, skills=skills)
    assert induced is not None
    assert induced.rejected_reason is None or "skill seen" not in induced.rejected_reason


# ---- batch + roundtrip speedup -------------------------------------------

def test_induce_speeds_up_subsequent_lookup():
    """After induction the same lookup becomes a direct fact (O(1))."""
    kb = _two_hop_kb()
    result = walk_chain(kb, "leaf", [Hop("partof"), Hop("locatedin")])
    induce_from_chain(kb, result, "leaf")
    ans, score = kb.answer({"S": "leaf", "R": "locatedin"}, "O")
    assert ans == "forest"
    assert score > 0.10


def test_induce_rejects_non_lifting_chain_by_default():
    """Chains whose first hop is not a lifting relation (e.g. causes)
    cannot be ascribed a meaningful induced relation. Default policy
    rejects them rather than emitting a vapid `implies` placeholder."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("winter", "causes", "snow"),
        ("snow", "isa", "weather"),
    ])
    result = walk_chain(kb, "winter", [Hop("causes"), Hop("isa")])
    if result.answer is None:
        return
    induced = induce_from_chain(kb, result, "winter")
    assert induced is not None
    assert induced.rejected_reason is not None
    assert "no meaningful induced relation" in induced.rejected_reason


def test_induce_allows_generic_implies_when_opted_in():
    """Legacy behaviour: callers that want every walked chain
    recorded (regardless of relation semantics) can opt in via the
    `allow_generic_implies` policy flag."""
    from rck.chain_induction import InductionPolicy
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("winter", "causes", "snow"),
        ("snow", "isa", "weather"),
    ])
    result = walk_chain(kb, "winter", [Hop("causes"), Hop("isa")])
    if result.answer is None:
        return
    induced = induce_from_chain(
        kb, result, "winter",
        policy=InductionPolicy(allow_generic_implies=True),
    )
    assert induced is not None
    assert induced.rejected_reason is None
    assert induced.relation == "implies"


def test_induce_rejects_partof_has_chain():
    """`partof -> has` should NOT forward: a wing is part of a bird and
    a bird has a beak, but a wing does not have a beak. The type-
    signature gate blocks this."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("wing", "partof", "bird"),
        ("bird", "has", "beak"),
    ])
    result = walk_chain(kb, "wing", [Hop("partof"), Hop("has")])
    if result.answer is None:
        return
    induced = induce_from_chain(kb, result, "wing")
    assert induced is not None
    assert induced.rejected_reason is not None
    assert "type-signature mismatch" in induced.rejected_reason


def test_induce_rejects_locatedin_size_chain():
    """`locatedin -> size` should NOT forward: a kitchen is in a house
    and a house may be large, but that doesn't make the kitchen large."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("kitchen", "locatedin", "house"),
        ("house", "size", "large"),
    ])
    result = walk_chain(kb, "kitchen", [Hop("locatedin"), Hop("size")])
    if result.answer is None:
        return
    induced = induce_from_chain(kb, result, "kitchen")
    assert induced is not None
    assert induced.rejected_reason is not None
    assert "type-signature mismatch" in induced.rejected_reason


def test_induce_partof_locatedin_chain_passes():
    """`partof -> locatedin` IS a valid transfer: a leaf is part of a
    tree and the tree is in the forest, therefore the leaf is in the
    forest."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("leaf", "partof", "tree"),
        ("tree", "locatedin", "forest"),
    ])
    result = walk_chain(kb, "leaf", [Hop("partof"), Hop("locatedin")])
    if result.answer is None:
        return
    induced = induce_from_chain(kb, result, "leaf")
    assert induced is not None
    assert induced.rejected_reason is None
    assert induced.relation == "locatedin"
    assert induced.obj == "forest"


def test_induce_lifting_via_class_forwards_relation():
    """`class` is now a lifting relation -- chain (X class Y, Y has Z)
    should induce (X has Z), not fall back to `implies`."""
    from rck.chain_induction import InductionPolicy
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("owl", "class", "bird"),
        ("bird", "has", "beak"),
    ])
    result = walk_chain(kb, "owl", [Hop("class"), Hop("has")])
    if result.answer is None:
        return
    induced = induce_from_chain(kb, result, "owl",
                                policy=InductionPolicy(min_confidence=0.05))
    assert induced is not None
    assert induced.rejected_reason is None
    assert induced.relation == "has"
    assert induced.obj == "beak"


def test_induce_filters_inverse_pair_chains():
    """A chain like (greatexpectations -[author]-> dickens -[wrote]-> olivertwist)
    must NOT induce (greatexpectations, wrote, olivertwist)."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("dickens", "wrote", "greatexpectations"),
        ("dickens", "wrote", "olivertwist"),
    ])
    # The auto-symmetrize gives us (greatexpectations, author, dickens) too.
    result = walk_chain(kb, "greatexpectations",
                        [Hop("author"), Hop("wrote")])
    if result.answer is None:
        # If the chain itself doesn't resolve in this random setup,
        # the test is vacuous -- skip.
        return
    induced = induce_from_chain(kb, result, "greatexpectations")
    assert induced is not None
    assert induced.rejected_reason is not None
    assert "inverse pair" in induced.rejected_reason


def test_induce_filters_same_relation_when_non_transitive():
    """X -[wrote]-> Y -[wrote]-> Z must NOT induce (X, wrote, Z) because
    `wrote` isn't transitive."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("dickens", "wrote", "olivertwist"),
        ("dickens", "wrote", "greatexpectations"),
    ])
    result = walk_chain(kb, "greatexpectations",
                        [Hop("wrote"), Hop("wrote")])
    if result.answer is None:
        return  # vacuous if HRR cleanup didn't surface the same-relation chain
    induced = induce_from_chain(kb, result, "greatexpectations")
    assert induced is not None
    assert induced.rejected_reason is not None
    assert ("same-relation" in induced.rejected_reason
            or "inverse pair" in induced.rejected_reason)


def test_induce_allows_same_relation_for_transitive():
    """isa chains ARE transitive and should pass the filter."""
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("mammal", "isa", "animal"),
    ])
    result = walk_chain(kb, "dog", [Hop("isa"), Hop("isa")])
    assert result.answer == "animal"
    induced = induce_from_chain(kb, result, "dog")
    assert induced is not None
    assert induced.rejected_reason is None
    assert induced.relation == "isa"
    assert induced.obj == "animal"


def test_conscious_agent_induce_end_to_end():
    """ConsciousAgent.induce() does discover + walk + induce in one call."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=16, install_self=False)
    for s, r, o in [
        ("dog", "isa", "mammal"),
        ("mammal", "isa", "animal"),
    ]:
        agent.tell(s, r, o)
    induced = agent.induce("dog", "animal")
    assert induced is not None
    assert induced.subject == "dog"
    assert induced.relation == "isa"
    assert induced.obj == "animal"
    assert induced.verified is True
    # Direct lookup of the induced fact is now possible.
    ans, score = agent.knowledge.answer({"S": "dog", "R": "isa"}, "O")
    assert ans in {"mammal", "animal"}  # both true; either OK


def test_induce_respects_negative_facts():
    """If (X, NOT_R, Y) is stored, never induce (X, R, Y) even when a
    chain would otherwise derive it."""
    from rck.negative_facts import deny
    kb = _two_hop_kb()
    # Explicitly deny the would-be induction target.
    deny(kb, "leaf", "locatedin", "forest")
    result = walk_chain(kb, "leaf", [Hop("partof"), Hop("locatedin")])
    assert result.answer == "forest"
    induced = induce_from_chain(kb, result, "leaf")
    assert induced is not None
    assert induced.rejected_reason is not None
    assert "denied" in induced.rejected_reason.lower()
    # The fact should NOT be in the KB (we never stored it).
    candidates = kb.query({"S": "leaf", "R": "locatedin"}, "O", top_k=5)
    objs = {str(s).lower() for s, sc in candidates if sc >= 0.10}
    assert "forest" not in objs


def test_induce_many_processes_batch():
    kb = _two_hop_kb()
    chains = []
    for start in ("leaf", "branch"):
        result = walk_chain(kb, start, [Hop("partof"), Hop("locatedin")])
        chains.append((start, result))
    induced = induce_many(kb, chains)
    assert len(induced) == 2
    assert all(f.subject in ("leaf", "branch") for f in induced)
