"""Tests for v11 modules: multi-task corpus, curriculum, confidence
propagation, clarification, query cache."""
import time

from rck.bulk_ingest import bulk_load_triples
from rck.clarification import (
    detect_ambiguous_entity, detect_ambiguous_top_k,
    detect_pronoun_ambiguity,
)
from rck.confidence_propagation import (
    PropagationConfig, propagate, verbalize_chain_confidence,
)
from rck.curriculum import (
    CurriculumScorer, report_difficulty_distribution,
    sort_examples_by_difficulty,
)
from rck.knowledge_base import ShardedKnowledgeBase
from rck.multi_task_corpus import (
    gen_boolean_yes, gen_contrast, gen_declarative_qa,
    gen_fill_blank, gen_paraphrase, gen_summarize,
    generate_multi_task_examples, stream_multi_task_corpus,
)
from rck.query_cache import QueryCache


# ---- multi-task corpus ---------------------------------------------------

def test_paraphrase_returns_pairs():
    ex = gen_paraphrase(("dog", "isa", "mammal"), max_pairs=3)
    assert len(ex) >= 2
    for e in ex:
        assert e.task == "paraphrase"
        assert e.draft != e.target


def test_qa_task_generates_question_answer_form():
    ex = gen_declarative_qa(("france", "capital", "paris"))
    assert any("what is the capital of france" in e.draft.lower() for e in ex)
    assert any("paris" in e.target.lower() for e in ex)


def test_fill_blank_target_is_just_the_value():
    ex = gen_fill_blank(("france", "capital", "paris"))
    assert len(ex) == 1
    assert "___" in ex[0].draft
    assert ex[0].target == "paris"


def test_boolean_yes_for_isa():
    ex = gen_boolean_yes(("dog", "isa", "mammal"))
    assert len(ex) == 1
    assert ex[0].draft.lower().startswith("is")
    assert "yes" in ex[0].target.lower()


def test_summarize_combines_facts():
    facts = [
        ("elephant", "isa", "mammal"),
        ("elephant", "has", "tusks"),
        ("elephant", "size", "huge"),
    ]
    ex = gen_summarize(facts)
    assert len(ex) >= 1
    e = ex[0]
    assert e.task == "summarize"
    # Should mention multiple attributes.
    out_lower = e.target.lower()
    hits = sum(w in out_lower for w in ("mammal", "tusks", "huge"))
    assert hits >= 2


def test_contrast_produces_unlike_X():
    facts = [
        ("sky", "color", "blue"),
        ("grass", "color", "green"),
    ]
    ex = gen_contrast(facts)
    assert len(ex) >= 1
    assert "unlike" in ex[0].target.lower()


def test_multi_task_per_triple_density():
    """ONE triple should yield more than the v7 paraphrase count alone."""
    ex = generate_multi_task_examples(("dog", "isa", "mammal"))
    tasks = {e.task for e in ex}
    # Should cover paraphrase + qa + fill_blank + boolean at minimum.
    assert len(tasks) >= 3


def test_stream_corpus_from_kb():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("france", "capital", "paris"),
    ], symmetrize=False)
    examples = list(stream_multi_task_corpus(kb))
    assert len(examples) >= 8  # several per fact


# ---- curriculum ----------------------------------------------------------

def test_curriculum_scorer_easier_lower_score():
    scorer = CurriculumScorer()
    easy = scorer.score(draft_len=3, target_len=2, task="boolean")
    hard = scorer.score(draft_len=20, target_len=20, task="contrast")
    assert easy < hard


def test_curriculum_sort_puts_easy_first():
    examples = [
        {"draft": "a long contrast example with many words " * 4,
         "target": "another long contrast example with many words " * 4,
         "task": "contrast"},
        {"draft": "is x y?", "target": "yes", "task": "boolean"},
        {"draft": "what is x?", "target": "x is y.", "task": "qa"},
    ]
    out = sort_examples_by_difficulty(examples, n_tiers=4)
    # Easy boolean should come BEFORE the long contrast.
    boolean_idx = next(i for i, e in enumerate(out) if e.get("task") == "boolean")
    contrast_idx = next(i for i, e in enumerate(out) if e.get("task") == "contrast")
    assert boolean_idx < contrast_idx


def test_difficulty_distribution_reports_tiers():
    examples = [{"draft": "a", "target": "b", "task": "boolean"}] * 5
    report = report_difficulty_distribution(examples)
    assert report["total"] == 5
    assert sum(report["per_tier"]) == 5


# ---- confidence propagation ---------------------------------------------

def test_product_rule_combines_links():
    res = propagate([0.5, 0.5], config=PropagationConfig(rule="product",
                                                          chain_decay=1.0))
    # 0.5 * 0.5 = 0.25
    assert abs(res["final_confidence"] - 0.25) < 1e-6


def test_min_rule_uses_weakest():
    res = propagate([0.9, 0.3, 0.7], config=PropagationConfig(rule="min",
                                                                chain_decay=1.0))
    assert abs(res["final_confidence"] - 0.3) < 1e-6
    assert res["weakest_link_index"] == 1


def test_chain_decay_lowers_long_chains():
    short = propagate([0.5, 0.5], config=PropagationConfig(chain_decay=0.9))
    long = propagate([0.5, 0.5, 0.5, 0.5],
                     config=PropagationConfig(chain_decay=0.9))
    assert long["final_confidence"] < short["final_confidence"]


def test_hedge_categories():
    assert propagate([0.5, 0.8])["hedge"] in ("strong", "moderate")
    assert propagate([0.05, 0.05])["hedge"] in ("weak", "uncertain")


def test_verbalize_hedge_changes_with_confidence():
    high = verbalize_chain_confidence(
        {"hedge": "strong"}, "the sky is blue")
    low = verbalize_chain_confidence(
        {"hedge": "uncertain"}, "the sky is blue")
    assert high != low
    assert "confident" in high.lower()


# ---- clarification -------------------------------------------------------

def test_ambiguous_top_k_when_close_scores():
    results = [("a", 0.50), ("b", 0.48), ("c", 0.20)]
    req = detect_ambiguous_top_k(results, ratio_threshold=0.9)
    assert req is not None
    assert "a" in req.candidates and "b" in req.candidates


def test_no_ambiguity_when_one_dominant():
    results = [("a", 0.90), ("b", 0.10)]
    req = detect_ambiguous_top_k(results, ratio_threshold=0.9)
    assert req is None


def test_ambiguous_entity_when_multiple_isa():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("paris", "isa", "city"),
        ("paris", "isa", "person"),
    ], symmetrize=False)
    req = detect_ambiguous_entity(kb, "paris")
    assert req is not None
    assert any("city" in c for c in req.candidates)
    assert any("person" in c for c in req.candidates)


def test_pronoun_ambiguity_when_multiple_recent():
    req = detect_pronoun_ambiguity("it", ["sky", "grass", "rose"])
    assert req is not None
    assert "sky" in req.question.lower()


# ---- query cache ---------------------------------------------------------

def test_cache_miss_then_hit():
    cache = QueryCache(max_size=4, ttl_seconds=60)
    assert cache.get("q1") is None
    cache.put("q1", {"answer": "x"})
    assert cache.get("q1") == {"answer": "x"}


def test_cache_normalisation_matches_variants():
    cache = QueryCache()
    cache.put("What is the capital of France?", {"answer": "paris"})
    # Different casing + spacing should hit the same entry.
    assert cache.get("what is the capital of france?") == {"answer": "paris"}
    assert cache.get("WHAT  IS  THE  CAPITAL  OF  FRANCE?") == {"answer": "paris"}


def test_cache_ttl_evicts():
    cache = QueryCache(max_size=4, ttl_seconds=0.05)
    cache.put("q1", {"answer": "x"})
    time.sleep(0.1)
    assert cache.get("q1") is None


def test_cache_lru_evicts_oldest():
    cache = QueryCache(max_size=2)
    cache.put("q1", {"answer": "1"})
    cache.put("q2", {"answer": "2"})
    cache.put("q3", {"answer": "3"})
    # q1 should be evicted.
    assert cache.get("q1") is None
    assert cache.get("q3") == {"answer": "3"}


def test_cache_invalidate_clears_all():
    cache = QueryCache()
    cache.put("q1", {"answer": "x"})
    n = cache.invalidate()
    assert n == 1
    assert cache.get("q1") is None


def test_cache_stats_track_hit_rate():
    cache = QueryCache()
    cache.put("q1", {"answer": "x"})
    cache.get("q1")
    cache.get("q1")
    cache.get("q2")  # miss
    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert 0.6 < stats["hit_rate"] < 0.7  # 2/3
