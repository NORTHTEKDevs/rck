from rck.agent import RCKAgent, StepTrace


def test_agent_step_produces_trace():
    agent = RCKAgent(hv_dim=512, n_columns=2, reservoir_dim=64, n_clauses=8, vocab_size=64, seed=0)
    tr = agent.step("a", learn=True, teacher_next="b")
    assert isinstance(tr, StepTrace)
    assert tr.input_symbol == "a"
    assert isinstance(tr.column_uncertainty, float)


def test_agent_observe_sequence():
    agent = RCKAgent(hv_dim=512, n_columns=2, reservoir_dim=64, n_clauses=8, vocab_size=64, seed=0)
    traces = agent.observe("hello", learn=True)
    assert len(traces) == 5
    for t in traces:
        assert t.emitted_symbol is not None


def test_agent_generate_after_observe():
    agent = RCKAgent(hv_dim=512, n_columns=2, reservoir_dim=64, n_clauses=8, vocab_size=64, seed=0)
    agent.observe("abcabcabcabc", learn=True)
    out, _ = agent.generate("a", max_new=5)
    assert len(out) == 5


def test_agent_reset_temporal_does_not_lose_codebook():
    agent = RCKAgent(hv_dim=256, n_columns=2, reservoir_dim=32, n_clauses=8, vocab_size=32, seed=0)
    agent.observe("xyz", learn=True)
    n_before = agent.codebook.size()
    agent.reset_temporal()
    assert agent.codebook.size() == n_before
