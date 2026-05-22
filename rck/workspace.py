"""Global Workspace: cosine-WTA competition + bundle-broadcast.

Each cycle, every module submits a candidate hypervector with a salience
score. The winner (highest salience) is bundled into the running workspace
HV and broadcast back to all modules as shared context.

This replaces transformer attention with a parameter-free competition.
Salience can be supplied externally (e.g. PCN prediction error magnitude,
column-vote variance, novelty signal).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rck.vsa import bundle, cosine


@dataclass
class GlobalWorkspace:
    """Running workspace HV with WTA broadcast.

    The workspace is an exponential moving bundle of past winners, decayed
    each step to keep recent context dominant.
    """

    dim: int = 10_000
    decay: float = 0.7  # Workspace memory: 0 = no memory, 1 = never forget.
    _ws: np.ndarray = field(default=None, init=False)
    _last_winner: str | None = field(default=None, init=False)
    _last_score: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._ws = np.zeros(self.dim, dtype=np.float32)

    def state(self) -> np.ndarray:
        """Return the current bipolar workspace HV."""
        out = np.sign(self._ws)
        out[out == 0] = 1
        return out.astype(np.int8)

    def last_winner(self) -> tuple[str | None, float]:
        return self._last_winner, self._last_score

    def step(self, candidates: dict[str, tuple[np.ndarray, float]]) -> np.ndarray:
        """Pick winning HV by salience and update workspace.

        Args:
            candidates: name -> (hv, salience). Higher salience wins.
        Returns: the new bipolar workspace HV.
        """
        if not candidates:
            return self.state()
        name, (hv, score) = max(candidates.items(), key=lambda kv: kv[1][1])
        self._last_winner, self._last_score = name, float(score)
        # Decay then add winner (real-valued bundling).
        self._ws = self.decay * self._ws + hv.astype(np.float32)
        return self.state()

    def cosine_step(
        self,
        candidates: dict[str, np.ndarray],
        query: np.ndarray | None = None,
    ) -> np.ndarray:
        """Salience defaults to cosine similarity with query (or current WS)."""
        q = self.state() if query is None else query
        scored = {name: (hv, cosine(hv, q)) for name, hv in candidates.items()}
        return self.step(scored)

    def reset(self) -> None:
        self._ws = np.zeros(self.dim, dtype=np.float32)
        self._last_winner = None
        self._last_score = 0.0
