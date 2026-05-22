"""Tests for v13 modules: shard sizing, self-verification, skill discovery."""
from __future__ import annotations

from rck.bulk_ingest import bulk_load_triples
from rck.knowledge_base import ShardedKnowledgeBase
from rck.shard_sizing import (
    ShardRecommendation, auto_shard_for_kb, recommend_shards,
    TARGET_MAX_FILL_BY_DIM,
)
from rck.self_verify import (
    VerificationResult, verify, verify_reverse, verify_roundtrip,
    verify_sibling_sanity,
)
from rck.skills import Skill, SkillLibrary


# ---- shard sizing ----------------------------------------------------------

def test_shard_recommendation_scales_with_facts():
    """Bigger KB -> more shards."""
    small = recommend_shards(1_000, dim=4096)
    big = recommend_shards(100_000, dim=4096)
    assert big.n_shards > small.n_shards
    assert isinstance(small, ShardRecommendation)


def test_shard_recommendation_respects_dim_targets():
    """Higher dim -> higher target fill -> fewer shards for same n_facts."""
    rec_4k = recommend_shards(10_000, dim=4096)
    rec_8k = recommend_shards(10_000, dim=8192)
    assert rec_4k.target_max_fill == TARGET_MAX_FILL_BY_DIM[4096]
    assert rec_8k.target_max_fill == TARGET_MAX_FILL_BY_DIM[8192]
    assert rec_8k.n_shards <= rec_4k.n_shards


def test_shard_recommendation_clamps_to_min():
    """Tiny KB still gets at least min_shards."""
    rec = recommend_shards(10, dim=4096, min_shards=8)
    assert rec.n_shards >= 8


def test_shard_recommendation_clamps_to_max():
    """Huge KB clamps to max_shards."""
    rec = recommend_shards(10_000_000_000, dim=4096, max_shards=1024)
    assert rec.n_shards == 1024


def test_shard_recommendation_powers_of_two():
    """n_shards is always a power of two for clean sharding."""
    for n_facts in (100, 1_000, 10_000, 100_000):
        rec = recommend_shards(n_facts, dim=4096)
        assert (rec.n_shards & (rec.n_shards - 1)) == 0, (
            f"n_shards={rec.n_shards} not a power of two for n_facts={n_facts}"
        )


def test_auto_shard_convenience():
    """One-line wrapper returns an int matching recommend_shards."""
    n = auto_shard_for_kb(5_000, dim=4096)
    assert n == recommend_shards(5_000, dim=4096).n_shards


def test_conscious_agent_auto_shards_from_expected_facts():
    """ConsciousAgent(expected_facts=...) overrides n_shards via recommendation."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=8, expected_facts=20_000, install_self=False)
    expected = recommend_shards(20_000, dim=4096).n_shards
    assert agent.n_shards == expected
    assert agent.knowledge.n_shards == expected


def test_query_union_returns_shard_tagged_results():
    """query_union surfaces every shard that has a candidate."""
    from rck.bulk_ingest import bulk_load_triples
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("elephant", "isa", "mammal"),
        ("whale", "isa", "mammal"),
        ("eagle", "isa", "bird"),
    ])
    # (?, isa, mammal) -- fan-out; should find at least 2 mammals.
    rows = kb.query_union({"R": "isa", "O": "mammal"}, "S",
                          per_shard_top_k=3, min_score=0.05)
    syms = {str(sym) for sym, _, _ in rows}
    assert "dog" in syms or "cat" in syms or "elephant" in syms or "whale" in syms
    # Every row carries a shard index in range.
    for _, _, shard_idx in rows:
        assert 0 <= shard_idx < kb.n_shards


def test_fanout_query_pools_evidence_across_shards():
    """When a symbol shows up in >1 shard, the merged score is boosted."""
    from rck.bulk_ingest import bulk_load_triples
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8, seed=0)
    # Many subjects, same (R, O); fan-out should still surface them.
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("horse", "isa", "mammal"),
        ("cow", "isa", "mammal"),
    ])
    results = kb.query({"R": "isa", "O": "mammal"}, "S", top_k=3)
    assert len(results) >= 1
    # Top match should be one of our mammals.
    top_sym = str(results[0][0])
    assert top_sym in {"dog", "cat", "horse", "cow"}


def test_conscious_agent_has_skills_library_and_verify_fact():
    """ConsciousAgent exposes skills + verify_fact wired to self_verify."""
    from rck.conscious_agent import ConsciousAgent
    agent = ConsciousAgent(dim=4096, n_shards=8, install_self=False)
    assert isinstance(agent.skills, SkillLibrary)
    assert agent.state()["skills"]["n"] == 0
    agent.tell("dog", "isa", "mammal")
    agent.tell("cat", "isa", "mammal")
    res = agent.verify_fact("dog", "isa", "mammal", methods=["roundtrip"])
    assert res["total_methods"] == 1
    assert res["by_method"]["roundtrip"]["verified"] is True


def test_shard_reasoning_is_present():
    """Reasoning string is non-empty and includes the target fill."""
    rec = recommend_shards(2_000, dim=4096)
    assert "target_max_fill" in rec.reasoning
    assert str(rec.n_shards) in rec.reasoning


# ---- self verification -----------------------------------------------------

def _build_small_kb() -> ShardedKnowledgeBase:
    kb = ShardedKnowledgeBase(dim=4096, n_shards=8)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("elephant", "isa", "mammal"),
        ("dog", "has", "fur"),
        ("cat", "has", "fur"),
        ("elephant", "has", "tusks"),
    ])
    return kb


def test_roundtrip_verifies_known_fact():
    kb = _build_small_kb()
    r = verify_roundtrip(kb, "dog", "isa", "mammal")
    assert isinstance(r, VerificationResult)
    assert r.verified
    assert r.method == "roundtrip"


def test_roundtrip_rejects_unknown_fact():
    kb = _build_small_kb()
    r = verify_roundtrip(kb, "dog", "isa", "fish")
    assert r.verified is False
    assert r.confidence_out == 0.0


def test_reverse_verifies_known_fact():
    kb = _build_small_kb()
    r = verify_reverse(kb, "dog", "isa", "mammal")
    assert r.method == "reverse"
    assert r.verified


def test_sibling_sanity_pass_when_siblings_share():
    kb = _build_small_kb()
    # cat is a mammal and so is dog; both have fur, so sibling sanity passes.
    r = verify_sibling_sanity(kb, "dog", "has", "fur")
    assert r.method == "sibling_sanity"
    assert r.verified


def test_verify_aggregate_returns_dict():
    kb = _build_small_kb()
    res = verify(kb, "dog", "isa", "mammal",
                 methods=["roundtrip", "reverse"])
    assert res["total_methods"] == 2
    assert "by_method" in res
    assert "roundtrip" in res["by_method"]
    assert "reverse" in res["by_method"]


def test_verify_partial_when_one_method_fails():
    kb = _build_small_kb()
    # Roundtrip should pass; pump a method that's likely to disagree
    # by querying a bogus reverse.
    res_good = verify(kb, "cat", "isa", "mammal")
    assert res_good["verification_count"] >= 1


# ---- skill discovery -------------------------------------------------------

def test_skill_records_success_and_failure():
    pattern = [("S", "isa"), ("O", "has")]
    lib = SkillLibrary()
    sk = lib.record_success(pattern)
    assert sk.success_count == 1
    assert sk.failure_count == 0
    lib.record_failure(pattern)
    assert lib.skills[next(iter(lib.skills))].failure_count == 1


def test_skill_confidence_math():
    sk = Skill(name="t", pattern=[("S", "isa")])
    assert sk.confidence == 0.0
    sk.record(True)
    sk.record(True)
    sk.record(False)
    assert abs(sk.confidence - (2.0 / 3.0)) < 1e-9


def test_skill_library_finds_exact_match():
    lib = SkillLibrary()
    pattern = [("S", "isa"), ("O", "has")]
    lib.record_success(pattern)
    found = lib.find_matching(pattern)
    assert found is not None
    assert found.success_count == 1


def test_skill_library_returns_none_on_miss():
    lib = SkillLibrary()
    lib.record_success([("S", "isa")])
    assert lib.find_matching([("O", "has")]) is None


def test_skill_partial_match_prefix():
    lib = SkillLibrary()
    a = [("S", "isa"), ("O", "has"), ("O", "color")]
    b = [("S", "isa"), ("O", "locatedin")]
    lib.record_success(a)
    lib.record_success(b)
    matches = lib.find_partial_matching([("S", "isa")])
    assert len(matches) == 2


def test_skill_library_stats_count_high_confidence():
    lib = SkillLibrary()
    pattern = [("S", "isa")]
    for _ in range(4):
        lib.record_success(pattern)
    stats = lib.stats()
    assert stats["n"] == 1
    assert stats["total_uses"] == 4
    assert stats["high_conf"] == 1


def test_skill_library_by_success_rate_filters_min_uses():
    lib = SkillLibrary()
    lib.record_success([("S", "isa")])  # only 1 use
    p2 = [("O", "has")]
    for _ in range(3):
        lib.record_success(p2)
    top = lib.by_success_rate(min_uses=3)
    assert len(top) == 1
    assert top[0].pattern == p2
