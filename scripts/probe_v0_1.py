"""Probe v0.1 to surface real failure modes before building v0.2.

Tests:
  (1) Next-char accuracy on held-out Shakespeare vs uniform baseline.
  (2) Bigram learning: after training, does the agent predict 'h' after 't'?
      Does it predict 'u' after 'q'?
  (3) PCN sanity: are different inputs producing well-separated HVs after
      training, or has the encoder collapsed?
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np

from rck.agent import RCKAgent
from rck.vsa import cosine


def load_text(path: str, n: int) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")[:n]


def train(agent: RCKAgent, text: str) -> float:
    t0 = time.time()
    agent.observe(text, learn=True)
    return time.time() - t0


def next_char_topk_accuracy(agent: RCKAgent, text: str, k_list=(1, 3, 5)) -> dict[int, float]:
    """Pure top-k next-char accuracy under stochastic=False decoding."""
    agent.reset_temporal()
    hits = {k: 0 for k in k_list}
    total = 0
    # Pre-feed first few chars to prime the state without scoring them.
    prime = text[:8]
    for c in prime:
        agent.step(c, learn=False)
    for i in range(8, len(text) - 1):
        c = text[i]
        target = text[i + 1]
        tr = agent.step(c, learn=False)
        # Pull top-k from the efe dict (lowest G = best). Map back to symbols.
        ranked = sorted(tr.efe.items(), key=lambda kv: kv[1])
        ranked_syms = [s for s, _ in ranked]
        for k in k_list:
            if target in ranked_syms[:k]:
                hits[k] += 1
        total += 1
    return {k: hits[k] / max(total, 1) for k in k_list}


def bigram_probe(agent: RCKAgent, probes: list[tuple[str, str]]) -> list[tuple[str, str, str, list[str]]]:
    """For each (context, expected_next), feed context fresh and show top-5."""
    out = []
    for ctx, expected in probes:
        agent.reset_temporal()
        tr = None
        for c in ctx:
            tr = agent.step(c, learn=False)
        if tr is None:
            continue
        ranked = sorted(tr.efe.items(), key=lambda kv: kv[1])
        top5 = [s for s, _ in ranked[:5]]
        out.append((ctx, expected, tr.emitted_symbol, top5))
    return out


def pcn_collapse_check(agent: RCKAgent) -> tuple[float, float]:
    """Encode N distinct char inputs through PCN; measure pairwise cosine.

    If PCN has collapsed, all outputs are near-identical (cos ~ 1).
    Healthy: random-text-like cos ~ 0.
    """
    chars = list(set(agent.codebook.symbols()))[:20]
    hvs = []
    for c in chars:
        x = agent._one_hot(c)
        hv = agent.pcn.encode(x, learn=False)
        hvs.append(hv)
    sims = []
    for i in range(len(hvs)):
        for j in range(i + 1, len(hvs)):
            sims.append(cosine(hvs[i], hvs[j]))
    return float(np.mean(sims)), float(np.std(sims))


def main() -> None:
    print("=" * 64)
    print(" RCK v0.1 PROBE")
    print("=" * 64)

    text_full = load_text("data/tiny_shakespeare.txt", 6_000)
    train_text = text_full[:5_000]
    eval_text = text_full[5_000:6_000]

    agent = RCKAgent(
        hv_dim=1024, n_columns=2, reservoir_dim=96, n_clauses=8,
        vocab_size=80, seed=0,
    )
    print(f"\nTraining on {len(train_text):,} chars ...")
    elapsed = train(agent, train_text)
    print(f"  done in {elapsed:.1f}s")
    print(f"  codebook size: {agent.codebook.size()}")

    print("\n[1] Next-char top-k accuracy on held-out 5k chars:")
    acc = next_char_topk_accuracy(agent, eval_text)
    vocab = agent.codebook.size()
    uniform = 1.0 / max(vocab, 1)
    print(f"  uniform baseline (vocab={vocab}): {uniform:.3f}")
    for k, v in acc.items():
        print(f"  top-{k}: {v:.3f}")

    print("\n[2] Bigram probe (does it learn t->h, q->u, etc?):")
    probes = [
        ("t",  "h"),
        ("q",  "u"),
        ("th", "e"),
        ("the", " "),
        (" t",  "h"),
        ("\nT",  "h"),
        ("ng",  " "),
    ]
    rows = bigram_probe(agent, probes)
    for ctx, expected, emitted, top5 in rows:
        ok = "OK" if expected in top5 else "MISS"
        print(f"  {ok}: '{ctx}' -> expected={expected!r} emitted={emitted!r} top5={top5}")

    print("\n[3] PCN collapse check (mean +/- std cosine across 20 chars):")
    mean, std = pcn_collapse_check(agent)
    health = "HEALTHY" if abs(mean) < 0.3 else "COLLAPSED" if mean > 0.8 else "DRIFTING"
    print(f"  mean cos = {mean:.3f}, std = {std:.3f}  [{health}]")

    print("\n[4] Sample generation from 'ROMEO:':")
    out, _ = agent.generate("ROMEO:", max_new=80)
    print("  ROMEO:" + "".join(str(c) for c in out))


if __name__ == "__main__":
    main()
