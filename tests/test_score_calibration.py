"""Tests for empirical score calibration (cosine -> P(correct)) and the
calibrated_product propagation rule.

The point: raw HRR cosine is a similarity FEATURE, not a probability.
ScoreCalibrator learns the monotone mapping cosine -> empirical accuracy
via isotonic regression (pool-adjacent-violators), and calibrated_product
restores real multiplicative semantics to chain confidence: the chain
score approximates P(every hop correct), so it is length-sensitive
(unlike geometric mean) without collapsing on clean hops (unlike the raw
product of cosines).
"""
import numpy as np
import pytest

from rck.confidence_propagation import PropagationConfig, propagate
from rck.score_calibration import ScoreCalibrator


def test_pav_separable_case():
    scores = [0.1, 0.2, 0.3, 0.4]
    correct = [False, False, True, True]
    cal = ScoreCalibrator.fit(scores, correct, smoothing=0.0)
    assert cal.prob(0.05) == pytest.approx(0.0)
    assert cal.prob(0.5) == pytest.approx(1.0)


def test_pav_pools_violators():
    # Alternating labels force pooling; pooled block accuracy = 0.5.
    scores = [0.1, 0.2, 0.3, 0.4]
    correct = [False, True, False, True]
    cal = ScoreCalibrator.fit(scores, correct, smoothing=0.0)
    assert cal.prob(0.15) == pytest.approx(0.5)
    assert cal.prob(0.35) <= 1.0


def test_smoothing_prevents_certainty():
    # 50 identical all-correct samples: one tie-grouped block, Beta(1,1)
    # posterior mean 51/52 -- never exactly 1.0.
    cal = ScoreCalibrator.fit([0.7] * 50, [True] * 50)
    assert cal.prob(0.7) == pytest.approx(51 / 52)
    cfg = PropagationConfig(rule="calibrated_product", calibrator=cal)
    assert cfg.combine([0.7] * 50, 50) < cfg.combine([0.7] * 2, 2)


def test_smoothing_pools_equal_mean_runs():
    # A strictly-increasing all-correct run must smooth as ONE block of
    # weight 100 (-> ~0.990), not 100 singletons (-> 2/3 each).
    scores = [0.5 + i * 0.004 for i in range(100)]
    cal = ScoreCalibrator.fit(scores, [True] * 100)
    assert cal.prob(0.7) == pytest.approx(101 / 102)


def test_predictions_monotone_in_score():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 400)
    correct = rng.uniform(0, 1, 400) < scores  # noisy but increasing
    cal = ScoreCalibrator.fit(scores.tolist(), correct.tolist())
    grid = np.linspace(0, 1, 101)
    preds = [cal.prob(float(s)) for s in grid]
    assert all(b >= a - 1e-12 for a, b in zip(preds, preds[1:]))


def test_prob_clamps_outside_fitted_range():
    cal = ScoreCalibrator.fit([0.3, 0.4, 0.5], [False, True, True])
    assert 0.0 <= cal.prob(-1.0) <= cal.prob(2.0) <= 1.0
    assert cal.prob(2.0) == pytest.approx(cal.prob(0.5))


def test_prob_nan_maps_to_lowest_block_not_highest():
    # numpy sorts NaN above everything; unguarded searchsorted would
    # hand a NaN score the HIGHEST-probability block.
    cal = ScoreCalibrator.fit([0.1, 0.2, 0.7, 0.8],
                              [False, False, True, True])
    assert cal.prob(float("nan")) == pytest.approx(cal.prob(0.0))
    assert cal.prob(float("nan")) < cal.prob(0.8)


def test_fit_rejects_empty_or_mismatched():
    with pytest.raises(ValueError):
        ScoreCalibrator.fit([], [])
    with pytest.raises(ValueError):
        ScoreCalibrator.fit([0.1, 0.2], [True])


def test_roundtrip_dict():
    cal = ScoreCalibrator.fit([0.1, 0.4, 0.7, 0.9],
                              [False, False, True, True])
    clone = ScoreCalibrator.from_dict(cal.to_dict())
    for s in (0.0, 0.2, 0.5, 0.8, 1.0):
        assert clone.prob(s) == pytest.approx(cal.prob(s))


class _IdentityCalibrator:
    """Stub: treats the score itself as the probability."""

    def prob(self, score: float) -> float:
        return max(0.0, min(1.0, score))


def test_calibrated_product_is_true_product_without_decay():
    cfg = PropagationConfig(rule="calibrated_product",
                            calibrator=_IdentityCalibrator())
    conf = cfg.combine([0.9, 0.9, 0.9], 3)
    assert conf == pytest.approx(0.9 ** 3)


def test_calibrated_product_is_length_sensitive_unlike_geometric_mean():
    cal = _IdentityCalibrator()
    cfg = PropagationConfig(rule="calibrated_product", calibrator=cal)
    gm = PropagationConfig(rule="geometric_mean", chain_decay=1.0)
    short = cfg.combine([0.95] * 2, 2)
    long = cfg.combine([0.95] * 40, 40)
    assert long < short  # product decays with length
    assert gm.combine([0.95] * 40, 40) == pytest.approx(
        gm.combine([0.95] * 2, 2))  # geometric mean does not


def test_calibrated_product_survives_long_clean_chains():
    # Clean hops calibrate to ~0.99+; a 50-hop chain must stay usable,
    # which the raw-cosine product (0.7^50 ~ 1e-8) never could.
    cal = ScoreCalibrator.fit([0.65, 0.68, 0.70, 0.72, 0.75, 0.2, 0.15],
                              [True, True, True, True, True, False, False],
                              smoothing=0.0)
    cfg = PropagationConfig(rule="calibrated_product", calibrator=cal)
    conf = cfg.combine([0.70] * 50, 50)
    assert conf > 0.30  # per-hop P=1.0 on this toy data -> stays strong


def test_calibrated_product_requires_calibrator():
    cfg = PropagationConfig(rule="calibrated_product")
    with pytest.raises(ValueError, match="calibrator"):
        cfg.combine([0.9], 1)


def test_propagate_accepts_calibrated_rule():
    cfg = PropagationConfig(rule="calibrated_product",
                            calibrator=_IdentityCalibrator())
    out = propagate([0.9, 0.8], config=cfg)
    assert out["final_confidence"] == pytest.approx(0.72)
    assert out["hedge"] == "strong"
