import numpy as np

from rck.tsetlin import TsetlinLayer


def test_evaluate_returns_score_and_outputs():
    t = TsetlinLayer(n_features=16, n_clauses=8, seed=0)
    hv = np.ones(16, dtype=np.int8)
    score, outs = t.evaluate(hv)
    assert outs.shape == (8,)
    assert isinstance(score, float)


def test_feedback_runs_without_error():
    t = TsetlinLayer(n_features=32, n_clauses=8, seed=0)
    rng = np.random.default_rng(0)
    hv = rng.choice(np.array([-1, 1], dtype=np.int8), size=32)
    for _ in range(30):
        t.feedback(hv, target=1)
    # Should not crash and ta values stay in [0, n_states-1].
    assert (t._ta >= 0).all()
    assert (t._ta < t.n_states).all()


def test_explain_returns_strings():
    t = TsetlinLayer(n_features=8, n_clauses=4, seed=0)
    hv = np.array([1, 1, -1, 1, -1, 1, 1, -1], dtype=np.int8)
    # Force-include one literal so explain has something to show.
    t._ta[0, 0] = t.n_states - 1  # include feature 0
    exp = t.explain(hv)
    assert isinstance(exp, list)
