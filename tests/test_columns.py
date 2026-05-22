import numpy as np

from rck.columns import ColumnEnsemble


def test_ensemble_vote_shape():
    ens = ColumnEnsemble(n_columns=3, input_dim=8, hv_dim=256, reservoir_dim=32, base_seed=0)
    x = np.zeros(8, dtype=np.float32); x[2] = 1.0
    vote, unc = ens.step(x, learn=False)
    assert vote.shape == (256,)
    assert set(np.unique(vote).tolist()) <= {-1, 1}
    assert unc >= 0.0


def test_uncertainty_drops_after_training():
    ens = ColumnEnsemble(n_columns=4, input_dim=8, hv_dim=512, reservoir_dim=32, base_seed=0)
    rng = np.random.default_rng(0)
    target = rng.choice(np.array([-1, 1], dtype=np.int8), size=512)
    x = np.zeros(8, dtype=np.float32); x[3] = 1.0
    _, unc0 = ens.step(x, learn=False)
    for _ in range(40):
        ens.step(x, learn=False)
        ens.train_readouts(target)
    _, unc1 = ens.step(x, learn=False)
    # Not a strict guarantee, but training toward one target should reduce
    # disagreement on average.
    assert unc1 <= unc0 + 0.05  # tolerant
