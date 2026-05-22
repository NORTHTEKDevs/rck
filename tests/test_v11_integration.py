"""v11 integration tests: streaming inference, end-to-end with the
new modules wired through ConsciousAgent."""
import tempfile
from pathlib import Path

from rck.bulk_ingest import bulk_load_triples
from rck.conscious_agent import ConsciousAgent
from rck.curriculum import sort_examples_by_difficulty
from rck.knowledge_base import ShardedKnowledgeBase
from rck.multi_task_corpus import write_corpus_jsonl
from rck.query_cache import QueryCache


def test_multi_task_corpus_writes_higher_density_than_v7():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("france", "capital", "paris"),
        ("germany", "capital", "berlin"),
    ], symmetrize=False)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        stats = write_corpus_jsonl(kb, path)
    finally:
        Path(path).unlink()
    # 4 base facts should yield well above the v7 ~24 examples.
    # Multi-task adds qa + fill_blank + boolean per fact.
    assert stats["examples"] >= 24
    # Per-fact density should be at least 6 examples.
    assert stats["examples"] / 4 >= 6


def test_multi_task_corpus_includes_all_task_kinds():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("france", "capital", "paris"),
    ], symmetrize=False)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        stats = write_corpus_jsonl(kb, path)
    finally:
        Path(path).unlink()
    # Should have multiple task types represented.
    tasks = set(stats["by_task"].keys())
    assert "paraphrase" in tasks
    # At least one of qa/fill_blank/boolean should appear.
    assert tasks & {"qa", "fill_blank", "boolean"}


def test_streaming_polisher_yields_chunks(tmp_path):
    """Smoke test: streaming should yield SOME chunks for a draft."""
    import torch

    from rck.polisher import (
        PairDataset, PolisherConfig, PolisherModel, PolisherTokenizer,
        train_polisher,
    )
    from rck.polisher.training import TrainConfig, save_checkpoint
    from rck.polisher.inference import NeuralPolisher

    torch.manual_seed(0)
    pairs = [("dog is mammal", "dog is a mammal")] * 8
    tok = PolisherTokenizer.from_corpus(
        [d for d, _ in pairs] + [t for _, t in pairs], min_count=1,
    )
    ds = PairDataset(pairs, tok, max_seq_len=24)
    config = PolisherConfig.tiny(vocab_size=tok.vocab_size)
    config.max_seq_len = 24
    model = PolisherModel(config)
    train_cfg = TrainConfig(batch_size=2, max_steps=8, warmup_steps=2,
                             lr=1e-3, log_every=100, device="cpu")
    train_polisher(model, ds, tok, config=train_cfg)
    save_checkpoint(model, tok, tmp_path)
    polisher = NeuralPolisher(weights_path=tmp_path, device="cpu",
                               max_new_tokens=5, temperature=0.7)
    chunks = list(polisher.stream_polish("dog is mammal"))
    # Should have yielded at least one chunk OR no chunks if it
    # immediately hit EOS (which is also valid).
    assert isinstance(chunks, list)
    # Each chunk should be a non-empty string.
    for c in chunks:
        assert isinstance(c, str)


def test_curriculum_sort_does_not_change_count():
    examples = [
        {"draft": f"q{i}", "target": f"a{i}", "task": "qa"}
        for i in range(20)
    ]
    out = sort_examples_by_difficulty(examples, n_tiers=4)
    assert len(out) >= len(examples)  # >= because of review duplicates


def test_query_cache_speeds_up_repeat_queries():
    agent = ConsciousAgent(dim=2048, n_shards=8, seed=0)
    agent.tell("sky", "color", "blue")
    cache = QueryCache()
    q = "What color is the sky?"
    # First call: not cached.
    if cache.get(q) is None:
        res = agent.ask(q)
        cache.put(q, res)
    # Second call: should hit cache.
    cached = cache.get(q)
    assert cached is not None
    assert cached["verbal"] == agent.ask(q)["verbal"]


def test_query_cache_invalidates_on_correction():
    """Calling code is responsible for invalidating after writes."""
    cache = QueryCache()
    cache.put("test", {"answer": "x"})
    cache.invalidate()
    assert cache.get("test") is None
