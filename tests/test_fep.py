import numpy as np

from rck.codebook import Codebook
from rck.fep import ActiveInference
from rck.vsa import random_hv


def test_perceive_reduces_error_over_time():
    fep = ActiveInference(dim=256, rank=16, lr=0.05)
    rng = np.random.default_rng(0)
    s = random_hv(256, rng)
    s_next = random_hv(256, rng)
    errs = []
    for _ in range(80):
        errs.append(fep.perceive(s, s_next))
    assert errs[-1] < errs[0]


def test_act_returns_candidate():
    cb = Codebook(dim=256, seed=0)
    for ch in "abcde":
        cb.encode(ch)
    fep = ActiveInference(dim=256, rank=16, lr=0.01)
    rng = np.random.default_rng(0)
    s = random_hv(256, rng)
    sym, efe = fep.act(s, cb, top_k=3, stochastic=False)
    assert sym in set("abcde")
    assert sym in efe


def test_repetition_penalty_avoids_immediate_repeat():
    """If we just emitted 'a' and 'a' has G near a tie, the penalty pushes
    decoding toward another candidate."""
    cb = Codebook(dim=512, seed=0)
    for ch in "ab":
        cb.encode(ch)
    fep = ActiveInference(dim=512, rank=8, repetition_penalty=10.0)
    rng = np.random.default_rng(0)
    s = random_hv(512, rng)
    fep._recent.append("a")  # pretend we just emitted 'a'
    sym, _ = fep.act(s, cb, top_k=2, stochastic=False)
    assert sym == "b"


def test_low_rank_perceive_is_cheap():
    """Just verifies the path runs and weights stay bounded."""
    fep = ActiveInference(dim=1024, rank=32, lr=1e-2)
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = random_hv(1024, rng)
        sn = random_hv(1024, rng)
        fep.perceive(s, sn)
    assert np.isfinite(fep._U).all()
    assert np.isfinite(fep._V).all()
