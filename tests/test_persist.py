import tempfile
from pathlib import Path

from rck.agent import RCKAgent
from rck.persist import save, load


def test_save_load_roundtrip_preserves_emission():
    agent = RCKAgent(hv_dim=256, n_columns=2, reservoir_dim=32, n_clauses=8,
                     vocab_size=32, fep_rank=16, bigram_order=2, seed=0)
    agent.observe("hello world", learn=True)
    out_before, _ = agent.generate("h", max_new=6)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "rck"
        save(agent, path)
        assert path.with_suffix(".npz").exists()
        assert path.with_suffix(".json").exists()
        agent2 = load(path)

    # Determinism: after reset_temporal, identical generation.
    agent.reset_temporal()
    agent2.reset_temporal()
    out_a, _ = agent.generate("h", max_new=6)
    out_b, _ = agent2.generate("h", max_new=6)
    assert out_a == out_b


def test_save_load_preserves_codebook():
    agent = RCKAgent(hv_dim=128, n_columns=2, reservoir_dim=16, n_clauses=4,
                     vocab_size=16, fep_rank=8, bigram_order=1, seed=0)
    agent.observe("abc", learn=True)
    syms_before = sorted(agent.codebook.symbols())

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "rck"
        save(agent, path)
        agent2 = load(path)

    syms_after = sorted(agent2.codebook.symbols())
    assert syms_before == syms_after
