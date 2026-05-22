import numpy as np

from rck.lsm import LiquidStateMachine


def test_lsm_state_changes_over_time():
    lsm = LiquidStateMachine(input_dim=16, reservoir_dim=64, hv_dim=128, seed=0)
    u = np.ones(16, dtype=np.float32)
    s0 = lsm.step_state(u).copy()
    s1 = lsm.step_state(u).copy()
    # Even with constant input, dynamics evolve.
    assert not np.array_equal(s0, s1)


def test_lsm_echo_state_property():
    """Spectral radius < 1 -> state stays bounded under bounded input."""
    lsm = LiquidStateMachine(input_dim=16, reservoir_dim=64, hv_dim=128,
                             spectral_radius=0.9, seed=1)
    u = np.ones(16, dtype=np.float32)
    for _ in range(100):
        lsm.step_state(u)
    assert np.max(np.abs(lsm._state)) < 5.0


def test_lsm_readout_learns_target():
    lsm = LiquidStateMachine(input_dim=8, reservoir_dim=64, hv_dim=64,
                             ridge=1e-2, seed=2)
    rng = np.random.default_rng(0)
    target = rng.choice(np.array([-1, 1], dtype=np.int8), size=64)
    u = np.ones(8, dtype=np.float32)
    for _ in range(80):
        lsm.step_state(u)
        lsm.train_readout(target)
    out = lsm.readout()
    # After many updates the readout should be majority-correct.
    match = (out == target).sum()
    assert match > 64 * 0.7
