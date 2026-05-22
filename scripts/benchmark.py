"""Per-module benchmark: how long does each subsystem actually take?"""
from __future__ import annotations

import time

import numpy as np

from rck.agent import RCKAgent
from rck.bigram import BigramMemory
from rck.codebook import Codebook
from rck.columns import ColumnEnsemble
from rck.fep import ActiveInference
from rck.lsm import LiquidStateMachine
from rck.pcn import PCNEncoder
from rck.tsetlin import TsetlinLayer
from rck.vsa import bind, bundle, random_hv
from rck.workspace import GlobalWorkspace


def bench(name: str, fn, n: int = 500) -> float:
    # warmup
    for _ in range(5):
        fn()
    t0 = time.time()
    for _ in range(n):
        fn()
    dt = (time.time() - t0) / n * 1000
    print(f"  {name:24} {dt:7.3f} ms/call")
    return dt


def main() -> None:
    D = 2048
    print(f"\n=== Per-module benchmark (D={D}, n=500 calls) ===\n")

    rng = np.random.default_rng(0)
    hv = random_hv(D, rng)
    hv2 = random_hv(D, rng)

    print("VSA primitives:")
    bench("bind",          lambda: bind(hv, hv2))
    bench("bundle 4",      lambda: bundle(hv, hv2, hv, hv2))
    bench("random_hv",     lambda: random_hv(D, rng))

    print("\nCodebook (vocab=80):")
    cb = Codebook(dim=D, seed=0)
    for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789.,?!":
        cb.encode(ch)
    bench("encode (cached)", lambda: cb.encode("a"))
    bench("fast_cleanup k=5", lambda: cb.fast_cleanup(hv, top_k=5))

    print("\nPCN encoder:")
    pcn = PCNEncoder(input_dim=80, hidden_dims=(128, 128), hv_dim=D, seed=0)
    x = np.zeros(80, dtype=np.float32); x[5] = 1.0
    bench("encode (no learn)", lambda: pcn.encode(x, learn=False))
    bench("encode (learn)",    lambda: pcn.encode(x, learn=True))

    print("\nLSM reservoir:")
    lsm = LiquidStateMachine(input_dim=D, reservoir_dim=128, hv_dim=D, seed=0)
    bench("step",          lambda: lsm.step(hv))
    bench("train_readout", lambda: lsm.train_readout(hv2))

    print("\nTsetlin (clauses=16, n_features=D):")
    ts = TsetlinLayer(n_features=D, n_clauses=16, seed=0)
    bench("evaluate",  lambda: ts.evaluate(hv))
    bench("feedback",  lambda: ts.feedback(hv, target=1))
    bench("explain",   lambda: ts.explain(hv))

    print("\nGlobal Workspace:")
    ws = GlobalWorkspace(dim=D)
    cands = {"a": (hv, 0.5), "b": (hv2, 0.3)}
    bench("step",      lambda: ws.step(cands))

    print("\nFEP (rank=96):")
    fep = ActiveInference(dim=D, rank=96)
    bench("perceive",  lambda: fep.perceive(hv, hv2))
    bench("predict",   lambda: fep.predict(hv))
    bench("act",       lambda: fep.act(hv, cb, top_k=12, stochastic=False))

    print("\nColumns (n=3):")
    cols = ColumnEnsemble(n_columns=3, input_dim=80, hv_dim=D, reservoir_dim=64, base_seed=0)
    bench("step",      lambda: cols.step(x, learn=False))

    print("\nBigram (order=3):")
    bm = BigramMemory(dim=D, order=3)
    bm.observe(cb, "a", "b"); bm.observe(cb, "b", "c"); bm.observe(cb, "c", "d")
    bench("observe",   lambda: bm.observe(cb, "a", "b"))
    bench("query k=5", lambda: bm.query(cb, top_k=5))

    print("\nEnd-to-end agent.step:")
    agent = RCKAgent(
        hv_dim=D, n_columns=3, reservoir_dim=128, n_clauses=16,
        vocab_size=80, fep_rank=96, bigram_order=3, seed=0,
    )
    agent.observe("hello world", learn=True)
    bench("step (learn)",    lambda: agent.step("h", learn=True, teacher_next="e"), n=200)
    bench("step (no-learn)", lambda: agent.step("h", learn=False), n=200)


if __name__ == "__main__":
    main()
