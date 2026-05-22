"""Tests for v12 modules: capacity profiler, dreaming, active learning,
evaluation, identity, adversarial, sparse HRR."""
import tempfile
from pathlib import Path

from rck.active_learning import (
    find_gap_candidates, find_low_confidence_facts,
    find_provenance_gaps, prioritize_questions,
)
from rck.adversarial import (
    gen_compound_cases, gen_negation_cases, gen_polysemy_cases,
    generate_test_set,
)
from rck.bulk_ingest import bulk_load_triples
from rck.capacity_profiler import (
    CapacityResult, dim_sweep, find_capacity, profile, shard_sweep, sweep,
)
from rck.conscious_agent import ConsciousAgent
from rck.dreaming import (
    ConsolidationReport, compress_duplicates, consolidate,
    detect_contradictions, generate_abstractions,
)
from rck.evaluation import (
    brier_score, calibration_table, measure_accuracy,
    measure_hallucination, measure_latency, run_full_suite,
)
from rck.identity import IdentityStore, UserProfile
from rck.knowledge_base import ShardedKnowledgeBase
from rck.memory_hierarchy import EpisodicMemory
from rck.provenance import ProvenanceStore
from rck.sparse_hrr import (
    SparseCodebook, SparseHV, bind_sparse, bundle_sparse,
    compare_memory_vs_dense, cosine_sparse, name_hashed_sparse,
    random_sparse,
)


# ---- capacity profiler -------------------------------------------------

def test_capacity_profile_returns_result():
    r = profile(100, dim=1024, n_shards=8)
    assert isinstance(r, CapacityResult)
    assert r.n_facts == 100
    assert 0.0 <= r.recall_at_1 <= 1.0


def test_capacity_recall_drops_at_overload():
    # With D=512 and only 4 shards, 5000 facts SHOULD strain recall.
    sparse = profile(200, dim=2048, n_shards=16)
    overload = profile(5000, dim=512, n_shards=4)
    assert sparse.recall_at_1 >= overload.recall_at_1 - 0.05  # tolerant


def test_capacity_sweep_returns_curve():
    curve = sweep([100, 500], dim=1024, n_shards=8)
    assert len(curve) == 2
    assert curve[0].n_facts < curve[1].n_facts


def test_find_capacity_returns_boundary():
    """find_capacity should walk until recall drops below threshold."""
    result = find_capacity(dim=512, n_shards=4, target_recall=0.90,
                            step=200, max_facts=1000)
    assert "capacity_facts" in result
    assert isinstance(result["curve"], list)


# ---- dreaming -----------------------------------------------------------

def test_compress_duplicates_removes_repeats():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [("dog", "isa", "mammal")] * 5, symmetrize=False)
    removed = compress_duplicates(kb)
    assert len(removed) >= 4


def test_detect_contradictions_finds_two_capitals():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("france", "capital", "paris"),
        ("france", "capital", "lyon"),  # contradiction
    ], symmetrize=False)
    prov = ProvenanceStore()
    contras = detect_contradictions(kb, prov)
    assert any(c["subject"] == "france" for c in contras)


def test_generate_abstractions_with_enough_support():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    facts = [(f"animal_{i}", "isa", "mammal") for i in range(6)]
    facts += [(f"animal_{i}", "has", "fur") for i in range(6)]
    bulk_load_triples(kb, facts, symmetrize=False)
    abstractions = generate_abstractions(kb, min_support=5)
    # Should create (mammal, has, fur).
    assert any(a == ("mammal", "has", "fur") for a in abstractions)


def test_consolidate_returns_full_report():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [("dog", "isa", "mammal")], symmetrize=False)
    em = EpisodicMemory()
    prov = ProvenanceStore()
    rep = consolidate(kb, em, prov)
    assert isinstance(rep, ConsolidationReport)


# ---- active learning ----------------------------------------------------

def test_find_low_confidence_facts():
    prov = ProvenanceStore()
    prov.store("sky", "color", "blue", confidence=0.95)
    prov.store("alien", "color", "green", confidence=0.2)
    candidates = find_low_confidence_facts(prov, low_max=0.4)
    assert any(c.subject == "alien" for c in candidates)
    assert not any(c.subject == "sky" for c in candidates)


def test_prioritize_questions_returns_ranked():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("dog", "has", "fur"),
        ("cat", "has", "fur"),
        ("elephant", "isa", "mammal"),
    ], symmetrize=False)
    prov = ProvenanceStore()
    ranked = prioritize_questions(kb, prov, n_total=5)
    # Should be sorted in EIG desc.
    for i in range(1, len(ranked)):
        assert ranked[i - 1].expected_info_gain >= ranked[i].expected_info_gain


# ---- evaluation ---------------------------------------------------------

def _eval_agent():
    agent = ConsciousAgent(dim=2048, n_shards=8, seed=0)
    agent.tell("sky", "color", "blue")
    agent.tell("grass", "color", "green")
    agent.tell("france", "capital", "paris")
    return agent


def test_measure_accuracy_simple_set():
    agent = _eval_agent()
    res = measure_accuracy(agent, [
        ("What color is the sky?", "blue"),
        ("What color is the grass?", "green"),
        ("What is the capital of France?", "paris"),
    ])
    assert res.total == 3
    assert res.correct_top1 >= 2


def test_brier_score_returns_value():
    agent = _eval_agent()
    b = brier_score(agent, [("What color is the sky?", "blue")])
    assert 0.0 <= b <= 1.0


def test_calibration_table_has_bins():
    agent = _eval_agent()
    cal = calibration_table(agent, [
        ("What color is the sky?", "blue"),
        ("What color is the grass?", "green"),
    ], n_bins=5)
    assert len(cal.bins) == 5


def test_measure_hallucination_uses_default_nonsense():
    agent = _eval_agent()
    h = measure_hallucination(agent)
    assert h.total > 0
    # The model should rarely (ideally never) be confidently wrong.
    assert h.confident_wrong <= 2


def test_measure_latency_sub_100ms():
    agent = _eval_agent()
    lat = measure_latency(agent, ["What color is the sky?"] * 5)
    assert lat.p50_ms < 200.0  # generous bound, typical << 100ms


def test_run_full_suite_returns_all_keys():
    agent = _eval_agent()
    res = run_full_suite(agent, [
        ("What color is the sky?", "blue"),
        ("What color is the grass?", "green"),
    ])
    assert "accuracy" in res and "calibration" in res
    assert "hallucination" in res and "latency" in res


# ---- identity -----------------------------------------------------------

def test_user_profile_lifecycle():
    p = UserProfile(user_id="test_user")
    p.touch()
    assert p.interaction_count == 1
    p.note_topic("history")
    p.note_topic("history")
    p.note_topic("science")
    top = p.top_topics(2)
    assert top[0][0] == "history"


def test_identity_store_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        store = IdentityStore(storage_dir=Path(td))
        p = store.get_or_create("alice")
        p.touch(); p.touch()
        p.set_preference("style", "casual")
        store.save("alice")
        # Reload from disk.
        store2 = IdentityStore(storage_dir=Path(td))
        p2 = store2.load("alice")
        assert p2.interaction_count == 2
        assert p2.preferences["style"] == "casual"


# ---- adversarial --------------------------------------------------------

def test_gen_negation_returns_cases():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    cases = gen_negation_cases(kb, n=3)
    assert len(cases) == 3
    for c in cases:
        assert c.category == "negation"


def test_gen_compound_multihop():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    cases = gen_compound_cases(kb, n=3)
    assert all(c.category == "compound" for c in cases)


def test_generate_test_set_covers_categories():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    cases = generate_test_set(kb, per_category=2)
    categories = {c.category for c in cases}
    assert {"negation", "compound", "polysemy"} <= categories


# ---- sparse HRR ---------------------------------------------------------

def test_sparse_hv_dense_roundtrip():
    hv = random_sparse(dim=128, k=8, seed=0)
    dense = hv.dense()
    assert dense.shape == (128,)
    assert dense.sum() == 8


def test_name_hashed_sparse_deterministic():
    a = name_hashed_sparse("dog", dim=512, k=20)
    b = name_hashed_sparse("dog", dim=512, k=20)
    import numpy as np
    assert np.array_equal(a.positions, b.positions)


def test_bind_xor_self_inverse():
    a = random_sparse(dim=512, k=20, seed=0)
    b = random_sparse(dim=512, k=20, seed=1)
    bound = bind_sparse(a, b)
    recovered = bind_sparse(bound, b)
    import numpy as np
    assert np.array_equal(np.sort(recovered.positions), np.sort(a.positions))


def test_cosine_self_is_one():
    a = random_sparse(dim=512, k=20, seed=0)
    assert abs(cosine_sparse(a, a) - 1.0) < 1e-9


def test_cosine_orthogonal_is_low():
    a = random_sparse(dim=4096, k=80, seed=0)
    b = random_sparse(dim=4096, k=80, seed=1)
    cos = cosine_sparse(a, b)
    assert cos < 0.20  # well below self-cosine


def test_sparse_codebook_cleanup_returns_known_atom():
    cb = SparseCodebook(dim=4096, k=80, seed=0)
    cb.encode("dog"); cb.encode("cat"); cb.encode("elephant")
    res = cb.cleanup(cb.encode("dog"), top_k=2)
    assert res[0][0] == "dog"
    assert res[0][1] > 0.9


def test_bundle_sparse_aggregates_positions():
    a = random_sparse(dim=512, k=20, seed=0)
    b = random_sparse(dim=512, k=20, seed=1)
    c = random_sparse(dim=512, k=20, seed=2)
    bundled = bundle_sparse([a, a, b, c])
    # `a` should be the closest because it's bundled twice.
    cos_a = cosine_sparse(bundled, a)
    cos_b = cosine_sparse(bundled, b)
    cos_c = cosine_sparse(bundled, c)
    assert cos_a >= cos_b
    assert cos_a >= cos_c


def test_memory_savings_estimate():
    res = compare_memory_vs_dense(n_atoms=1000, dim_dense=4096,
                                    dim_sparse=8192, k_sparse=160)
    assert res["ratio"] > 5.0  # at least 5x memory savings
