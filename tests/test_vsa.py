import numpy as np
import pytest

from rck.vsa import bind, unbind, bundle, permute, cosine, binarize, random_hv


def test_random_hv_is_bipolar():
    rng = np.random.default_rng(0)
    h = random_hv(1000, rng)
    assert set(np.unique(h).tolist()) <= {-1, 1}
    assert h.shape == (1000,)


def test_bind_is_self_inverse_for_bipolar():
    rng = np.random.default_rng(1)
    a = random_hv(2000, rng)
    b = random_hv(2000, rng)
    bound = bind(a, b)
    recovered = unbind(bound, b)
    assert np.array_equal(recovered, a)


def test_bundle_preserves_dominant_component():
    rng = np.random.default_rng(2)
    a = random_hv(2000, rng)
    b = random_hv(2000, rng)
    c = random_hv(2000, rng)
    # Bundle two copies of `a` with one of `b`, one of `c`.
    s = bundle(a, a, a, b, c)
    assert cosine(s, a) > cosine(s, b)
    assert cosine(s, a) > cosine(s, c)


def test_permute_changes_vector():
    rng = np.random.default_rng(3)
    a = random_hv(1000, rng)
    p1 = permute(a, 1)
    assert not np.array_equal(p1, a)
    # Roll by full length is identity.
    p_full = permute(a, 1000)
    assert np.array_equal(p_full, a)


def test_cosine_self_is_one():
    rng = np.random.default_rng(4)
    a = random_hv(1000, rng)
    assert pytest.approx(cosine(a, a), abs=1e-6) == 1.0


def test_cosine_random_pairs_near_zero():
    rng = np.random.default_rng(5)
    sims = []
    for _ in range(50):
        a = random_hv(10_000, rng)
        b = random_hv(10_000, rng)
        sims.append(cosine(a, b))
    assert abs(np.mean(sims)) < 0.05


def test_binarize_pushes_zero_to_plus_one():
    x = np.array([0.0, -0.1, 0.1])
    b = binarize(x)
    assert b.tolist() == [1, -1, 1]
