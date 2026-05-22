"""Falsifiable claim: RCK does not catastrophically forget on small workloads.

We train on corpus A, then corpus B, then check that emission distribution
on A-only symbols still includes the A symbols (not all routed to B).
This is a *weak* test -- the strong one lives in examples/continual_learning.py
-- but it gates the pipeline.
"""
from rck.agent import RCKAgent


def test_no_total_catastrophic_forgetting():
    agent = RCKAgent(hv_dim=512, n_columns=2, reservoir_dim=64, n_clauses=8, vocab_size=64, seed=0)
    text_a = "alpha beta gamma " * 12
    text_b = "delta epsilon zeta " * 12
    agent.observe(text_a, learn=True)
    n_a_symbols = agent.codebook.size()
    agent.observe(text_b, learn=True)
    # Codebook should grow, not shrink.
    assert agent.codebook.size() > n_a_symbols
    # After learning B, an "a" input should not always emit a B-only symbol.
    agent.reset_temporal()
    emissions = []
    for _ in range(20):
        tr = agent.step("a", learn=False)
        emissions.append(tr.emitted_symbol)
    # The model still has access to A-era symbols in its codebook.
    a_chars = set(text_a)
    b_only = set(text_b) - a_chars
    # At least one emission should be from outside b_only -- proving A wasn't
    # wholly overwritten. (Stronger numeric tests in examples/.)
    assert any(e not in b_only for e in emissions)
