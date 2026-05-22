"""Thousand-Brains reference-frame columns.

N independent column instances each maintain their own PCN + LSM with
different random initialisations. They each produce a candidate HV. We
bundle their votes; the variance of pairwise cosine similarities is the
uncertainty signal.

High variance = columns disagree = the model is uncertain. The agent can
use this to trigger exploratory behaviour (raise temperature, ask for
clarification, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rck.lsm import LiquidStateMachine
from rck.pcn import PCNEncoder
from rck.vsa import bundle, cosine


@dataclass
class Column:
    pcn: PCNEncoder
    lsm: LiquidStateMachine

    def step(self, x_input: np.ndarray, learn: bool = True) -> np.ndarray:
        hv_pcn = self.pcn.encode(x_input, learn=learn)
        return self.lsm.step(hv_pcn)


@dataclass
class ColumnEnsemble:
    """Stack of N independent columns voting via bundling."""

    n_columns: int
    input_dim: int
    hv_dim: int = 10_000
    reservoir_dim: int = 256
    base_seed: int = 0
    _columns: list[Column] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        for k in range(self.n_columns):
            seed = self.base_seed * 1000 + k
            pcn = PCNEncoder(
                input_dim=self.input_dim,
                hidden_dims=(128, 128),
                hv_dim=self.hv_dim,
                seed=seed,
            )
            lsm = LiquidStateMachine(
                input_dim=self.hv_dim,
                reservoir_dim=self.reservoir_dim,
                hv_dim=self.hv_dim,
                seed=seed + 1,
            )
            self._columns.append(Column(pcn=pcn, lsm=lsm))

    def step(self, x_input: np.ndarray, learn: bool = True) -> tuple[np.ndarray, float]:
        """Run all columns, bundle their votes, return (vote_hv, uncertainty)."""
        hvs = [c.step(x_input, learn=learn) for c in self._columns]
        vote = bundle(*hvs)
        # Pairwise cosine similarities; high variance = disagreement.
        sims = []
        for i in range(len(hvs)):
            for j in range(i + 1, len(hvs)):
                sims.append(cosine(hvs[i], hvs[j]))
        uncertainty = float(np.var(sims)) if sims else 0.0
        return vote, uncertainty

    def train_readouts(self, target_hv: np.ndarray) -> None:
        """Train each column's LSM readout toward the same target."""
        for c in self._columns:
            c.lsm.train_readout(target_hv)

    def reset(self) -> None:
        for c in self._columns:
            c.lsm.reset()
