"""Empirical score calibration: cosine similarity -> P(correct).

A raw HRR cleanup cosine is a similarity FEATURE, not a probability. A
clean single-fact retrieval at D=4096 lands around 0.7 cosine and is
almost always correct; treating 0.7 as "70% probable" is what made the
legacy product rule collapse over long chains (0.7^10 ~ 0.03) and what
made the geometric-mean workaround necessary.

This module learns the actual monotone mapping from cosine to empirical
accuracy with isotonic regression (pool-adjacent-violators), fitted on
labeled retrievals sampled across bundle loads (see
`scripts/confidence_calibration_study.py`). With calibrated per-hop
probabilities, plain multiplication becomes the RIGHT rule again:

    chain_confidence = prod_i  P_hat(correct | cosine_i)
                     ~ P(every hop correct)   (independence assumption)

which is length-sensitive (a 50-hop chain IS less certain than a 2-hop
chain) without collapsing on clean hops (P_hat ~ 0.99+, not 0.7).
Plug it in via `PropagationConfig(rule="calibrated_product",
calibrator=...)`.

Complements `rck.confidence_calibration` (provenance-based discounts):
that module adjusts for WHERE a fact came from; this one adjusts for
what a similarity score is worth as evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoreCalibrator:
    """Monotone step-function mapping score -> empirical P(correct).

    `thresholds[i]` is the highest score in pooled block i (ascending);
    `probs[i]` is that block's empirical accuracy. Prediction takes the
    block whose score range contains the query, clamping outside the
    fitted range.
    """

    thresholds: np.ndarray
    probs: np.ndarray
    n_samples: int = 0

    @classmethod
    def fit(cls, scores: list[float], correct: list[bool],
            smoothing: float = 1.0) -> "ScoreCalibrator":
        """Isotonic regression via pool-adjacent-violators, with
        Beta-smoothed block probabilities.

        Args:
            scores: retrieval scores (any monotone evidence feature).
            correct: whether each retrieval's top-1 answer was right.
            smoothing: Laplace/Beta prior strength `a`. Each pooled
                block reports (successes + a) / (weight + 2a) -- the
                posterior mean under Beta(a, a). This keeps the
                calibrator from ever claiming certainty: a block of 300
                all-correct samples maps to ~0.9967, not 1.0, so
                calibrated chain products stay length-sensitive. Pass
                0.0 for the raw empirical fit.
        """
        if len(scores) == 0:
            raise ValueError("cannot fit calibrator on zero samples")
        if len(scores) != len(correct):
            raise ValueError(
                f"scores ({len(scores)}) and correct ({len(correct)}) "
                "must have equal length")
        order = np.argsort(np.asarray(scores, dtype=np.float64),
                           kind="stable")
        s = np.asarray(scores, dtype=np.float64)[order]
        y = np.asarray(correct, dtype=np.float64)[order]

        # 1. Group exact-score ties into single blocks.
        means: list[float] = []
        weights: list[float] = []
        tops: list[float] = []
        i = 0
        while i < len(s):
            j = i
            while j < len(s) and s[j] == s[i]:
                j += 1
            means.append(float(y[i:j].mean()))
            weights.append(float(j - i))
            tops.append(float(s[i]))
            i = j

        # 2. PAV: merge while a block's mean exceeds its successor's.
        def _pav(ms: list[float], ws: list[float], ts: list[float]
                 ) -> tuple[list[float], list[float], list[float]]:
            om: list[float] = []
            ow: list[float] = []
            ot: list[float] = []
            for m, w, t in zip(ms, ws, ts):
                om.append(m)
                ow.append(w)
                ot.append(t)
                while len(om) > 1 and om[-2] > om[-1]:
                    m2, w2, t2 = om.pop(), ow.pop(), ot.pop()
                    w1 = ow[-1]
                    om[-1] = (om[-1] * w1 + m2 * w2) / (w1 + w2)
                    ow[-1] = w1 + w2
                    ot[-1] = t2
            return om, ow, ot

        means, weights, tops = _pav(means, weights, tops)

        # 3. Pool adjacent EQUAL-mean blocks so smoothing sees the full
        #    evidence mass behind each probability level (a run of
        #    all-correct singletons is one big block, not many tiny ones).
        pm: list[float] = []
        pw: list[float] = []
        pt: list[float] = []
        for m, w, t in zip(means, weights, tops):
            if pm and pm[-1] == m:
                pw[-1] += w
                pt[-1] = t
            else:
                pm.append(m)
                pw.append(w)
                pt.append(t)

        # 4. Beta-smooth each block, then 5. re-PAV: smoothing can break
        #    monotonicity where a small block follows a heavy one.
        if smoothing > 0.0:
            sm = [(m * w + smoothing) / (w + 2.0 * smoothing)
                  for m, w in zip(pm, pw)]
            pm, pw, pt = _pav(sm, pw, pt)

        return cls(thresholds=np.asarray(pt),
                   probs=np.asarray(pm),
                   n_samples=len(s))

    def prob(self, score: float) -> float:
        """P(correct) for a retrieval with this score."""
        idx = int(np.searchsorted(self.thresholds, float(score),
                                  side="left"))
        idx = min(idx, len(self.probs) - 1)
        return float(self.probs[idx])

    def to_dict(self) -> dict:
        return {
            "thresholds": self.thresholds.tolist(),
            "probs": self.probs.tolist(),
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScoreCalibrator":
        return cls(thresholds=np.asarray(d["thresholds"], dtype=np.float64),
                   probs=np.asarray(d["probs"], dtype=np.float64),
                   n_samples=int(d.get("n_samples", 0)))

    def reliability_table(self, scores: list[float], correct: list[bool],
                          n_bins: int = 10) -> list[dict]:
        """Equal-count reliability bins for reporting: per bin, the mean
        score, the calibrated probability, and the empirical accuracy."""
        s = np.asarray(scores, dtype=np.float64)
        y = np.asarray(correct, dtype=np.float64)
        order = np.argsort(s, kind="stable")
        s, y = s[order], y[order]
        bins = np.array_split(np.arange(len(s)), n_bins)
        out = []
        for b in bins:
            if len(b) == 0:
                continue
            out.append({
                "mean_score": float(s[b].mean()),
                "empirical_accuracy": float(y[b].mean()),
                "calibrated_prob": self.prob(float(s[b].mean())),
                "count": int(len(b)),
            })
        return out
