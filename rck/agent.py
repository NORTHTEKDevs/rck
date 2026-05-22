"""Top-level RCK agent: ties all modules into one closed cognitive loop.

v0.2 changes vs v0.1:
  - Added VSA n-gram memory (bigram.py) as a primary candidate source.
  - FEP is now low-rank (O(D*r) per step instead of O(D^2)).
  - Deterministic decoding by default; sampling is opt-in via temperature.
  - Anti-repetition + frequency prior live inside FEP.act.

Per token, the agent runs this pipeline:

    raw_token
        |
        v
    one-hot --[ PCN encoder ]--> hv_perc
                                   |
                                   v
                              [ LSM reservoir ]--> hv_temp
                                                     |
                                                     v
                  [ Codebook bind (perm-encoded position) ]--> hv_bound
                                                                 |
                                                                 v
                        [ Column ensemble vote ]--> hv_vote (+ uncertainty)
                                                                 |
                                                                 v
                                              [ Global Workspace WTA ]
                                                                 |
                                                                 v
                                                  workspace_state (hv_ws)
                                                                 |
                              [ Tsetlin layer ] --> reasoning_trace
                                                                 |
                                                                 v
              [ VSA n-gram memory query: top-k from context ]
                                                                 |
                                                                 v
        [ Active Inference: argmin EFE over n-gram + LSM + FEP candidates ]
                                                                 |
                                                                 v
                                                         emitted_token
                                                                 |
                                                                 v
                          [ Local updates: PCN, LSM, FEP, Bigram ]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable

import numpy as np

from rck.bigram import BigramMemory
from rck.codebook import Codebook
from rck.columns import ColumnEnsemble
from rck.fep import ActiveInference
from rck.lsm import LiquidStateMachine
from rck.pcn import PCNEncoder
from rck.tsetlin import TsetlinLayer
from rck.vsa import bind, cosine, permute
from rck.workspace import GlobalWorkspace


@dataclass
class StepTrace:
    input_symbol: Hashable
    emitted_symbol: Hashable
    workspace_winner: str | None
    workspace_score: float
    column_uncertainty: float
    tsetlin_score: float
    tsetlin_clauses: list[str]
    efe: dict[Hashable, float]
    bigram_top: list[tuple[Hashable, float]]
    pred_err: float


@dataclass
class RCKAgent:
    vocab_size: int = 128
    hv_dim: int = 4096
    n_columns: int = 4
    reservoir_dim: int = 256
    n_clauses: int = 32
    fep_rank: int = 64
    bigram_order: int = 2
    seed: int = 0
    sampling_temperature: float = 0.5
    stochastic_decode: bool = False

    codebook: Codebook = field(default=None, init=False)
    pcn: PCNEncoder = field(default=None, init=False)
    lsm: LiquidStateMachine = field(default=None, init=False)
    columns: ColumnEnsemble = field(default=None, init=False)
    tsetlin: TsetlinLayer = field(default=None, init=False)
    workspace: GlobalWorkspace = field(default=None, init=False)
    fep: ActiveInference = field(default=None, init=False)
    bigram: BigramMemory = field(default=None, init=False)

    _symbol_to_id: dict[Hashable, int] = field(default_factory=dict, init=False)
    _id_to_symbol: dict[int, Hashable] = field(default_factory=dict, init=False)
    _position: int = field(default=0, init=False)
    _last_ws: np.ndarray = field(default=None, init=False)
    _last_input: Hashable | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.codebook = Codebook(dim=self.hv_dim, seed=self.seed)
        self.pcn = PCNEncoder(
            input_dim=self.vocab_size, hidden_dims=(128, 128),
            hv_dim=self.hv_dim, seed=self.seed + 1,
        )
        self.lsm = LiquidStateMachine(
            input_dim=self.hv_dim, reservoir_dim=self.reservoir_dim,
            hv_dim=self.hv_dim, seed=self.seed + 2,
        )
        self.columns = ColumnEnsemble(
            n_columns=self.n_columns, input_dim=self.vocab_size,
            hv_dim=self.hv_dim, reservoir_dim=max(64, self.reservoir_dim // 2),
            base_seed=self.seed + 3,
        )
        self.tsetlin = TsetlinLayer(
            n_features=self.hv_dim, n_clauses=self.n_clauses, seed=self.seed + 4,
        )
        self.workspace = GlobalWorkspace(dim=self.hv_dim)
        self.fep = ActiveInference(
            dim=self.hv_dim, rank=self.fep_rank,
            temperature=self.sampling_temperature,
        )
        self.bigram = BigramMemory(dim=self.hv_dim, order=self.bigram_order)
        self._last_ws = np.zeros(self.hv_dim, dtype=np.int8)

    # ---- vocab -------------------------------------------------------------

    def _id_of(self, symbol: Hashable) -> int:
        if symbol not in self._symbol_to_id:
            new_id = len(self._symbol_to_id)
            if new_id >= self.vocab_size:
                new_id = new_id % self.vocab_size
            self._symbol_to_id[symbol] = new_id
            self._id_to_symbol[new_id] = symbol
        return self._symbol_to_id[symbol]

    def _one_hot(self, symbol: Hashable) -> np.ndarray:
        v = np.zeros(self.vocab_size, dtype=np.float32)
        v[self._id_of(symbol)] = 1.0
        return v

    # ---- candidate set -----------------------------------------------------

    def _candidate_symbols(self, hv_ws: np.ndarray) -> list[Hashable]:
        """Union of candidate sources -- LSM cleanup + n-gram query + FEP."""
        candidates: set[Hashable] = set()
        # LSM readout is trained to predict the next-char HV, so its current
        # readout cleanup is a primary candidate.
        lsm_pred = self.lsm.readout()
        for sym, _ in self.codebook.fast_cleanup(lsm_pred, top_k=6):
            candidates.add(sym)
        # n-gram query.
        for sym, _ in self.bigram.query(self.codebook, top_k=6):
            candidates.add(sym)
        # FEP transition cleanup.
        fep_pred = self.fep.predict(hv_ws)
        for sym, _ in self.codebook.fast_cleanup(fep_pred, top_k=6):
            candidates.add(sym)
        # Always include the most-common unigrams as a fallback.
        for sym in self.bigram.unigram_top(k=4):
            candidates.add(sym)
        return list(candidates)

    # ---- core step ---------------------------------------------------------

    def step(
        self,
        symbol: Hashable,
        learn: bool = True,
        teacher_next: Hashable | None = None,
    ) -> StepTrace:
        x = self._one_hot(symbol)
        sym_hv = self.codebook.encode(symbol)

        # 1. PCN encode.
        hv_perc = self.pcn.encode(x, learn=learn)

        # 2. LSM temporal integration.
        hv_temp = self.lsm.step(hv_perc)

        # 3. VSA bind with position permutation.
        hv_bound = bind(hv_temp, permute(sym_hv, self._position % self.hv_dim))

        # 4. Column ensemble vote.
        hv_vote, uncertainty = self.columns.step(x, learn=learn)

        # 5. Workspace WTA.
        cands = {
            "perception": (hv_perc, abs(cosine(hv_perc, self._last_ws))),
            "temporal":   (hv_temp, abs(cosine(hv_temp, self._last_ws)) + 0.05),
            "binding":    (hv_bound, abs(cosine(hv_bound, self._last_ws)) + 0.10),
            "columns":    (hv_vote, abs(cosine(hv_vote, self._last_ws)) + 0.05 - uncertainty),
        }
        hv_ws = self.workspace.step(cands)
        winner, win_score = self.workspace.last_winner()

        # 6. Tsetlin reasoning trace.
        ts_score, _ = self.tsetlin.evaluate(hv_ws)
        clauses = self.tsetlin.explain(hv_ws)

        # 7. Bigram query (purely for trace -- candidate set step is below).
        bigram_top = self.bigram.query(self.codebook, top_k=5)

        # 8. Active inference -- choose next symbol from weighted ensemble.
        # The diagnostic told us LSM (0.43 top-5 hit) > FEP (0.36) > bigram (0.12).
        # Weight signals proportionally.
        cand_syms = self._candidate_symbols(hv_ws)
        lsm_pred = self.lsm.readout().astype(np.float32)
        bigram_key = self.bigram._key(self.codebook, self.bigram._ctx)
        bigram_pred = bigram_key.astype(np.float32) * self.bigram._mem
        fep_pred = self.fep.predict(hv_ws)
        signals = {"lsm": lsm_pred, "bigram": bigram_pred, "fep": fep_pred}
        signal_weights = {"lsm": 4.0, "bigram": 0.5, "fep": 0.2}
        emitted, efe = self.fep.act(
            hv_ws, self.codebook,
            top_k=12,
            stochastic=self.stochastic_decode,
            bias_candidates=cand_syms,
            teacher_freq_update=teacher_next,
            signals=signals,
            signal_weights=signal_weights,
        )

        # 9. Local FEP transition update.
        pred_err = self.fep.perceive(self._last_ws, hv_ws) if learn else 0.0

        # 10. Teacher feedback.
        if learn and teacher_next is not None:
            target_hv = self.codebook.encode(teacher_next)
            tgt = 1 if cosine(target_hv, hv_ws) > 0 else -1
            self.tsetlin.feedback(hv_ws, tgt)
            self.lsm.train_readout(target_hv)
            self.columns.train_readouts(target_hv)

        # 11. Bigram memory update.
        if learn:
            self.bigram.observe(self.codebook, symbol, teacher_next)

        # 12. Bookkeeping.
        self._last_ws = hv_ws
        self._last_input = symbol
        self._position += 1

        return StepTrace(
            input_symbol=symbol,
            emitted_symbol=emitted,
            workspace_winner=winner,
            workspace_score=win_score,
            column_uncertainty=uncertainty,
            tsetlin_score=ts_score,
            tsetlin_clauses=clauses[:5],
            efe=efe,
            bigram_top=bigram_top,
            pred_err=pred_err,
        )

    # ---- high-level convenience --------------------------------------------

    def observe(self, symbols: Iterable[Hashable], learn: bool = True) -> list[StepTrace]:
        traces = []
        symbols = list(symbols)
        for i, s in enumerate(symbols):
            teacher = symbols[i + 1] if i + 1 < len(symbols) else None
            traces.append(self.step(s, learn=learn, teacher_next=teacher))
        return traces

    def generate(
        self,
        prompt: Iterable[Hashable],
        max_new: int = 64,
        stop_on: Hashable | None = None,
    ) -> tuple[list[Hashable], list[StepTrace]]:
        prompt = list(prompt)
        traces = self.observe(prompt, learn=False)
        out: list[Hashable] = []
        last_sym = traces[-1].emitted_symbol if traces else prompt[-1]
        for _ in range(max_new):
            tr = self.step(last_sym, learn=False, teacher_next=None)
            out.append(tr.emitted_symbol)
            traces.append(tr)
            last_sym = tr.emitted_symbol
            if stop_on is not None and last_sym == stop_on:
                break
        return out, traces

    def reset_temporal(self) -> None:
        self.lsm.reset()
        self.columns.reset()
        self.workspace.reset()
        self.bigram.reset_context()
        self.fep.reset_recent()
        self._last_ws = np.zeros(self.hv_dim, dtype=np.int8)
        self._position = 0
