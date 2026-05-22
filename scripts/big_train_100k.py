"""v1.0 calibration train: 100k chars Tiny Shakespeare.

Trains and saves a checkpoint at ~/projects/active/rck/checkpoints/rck_100k.
Reports next-char accuracy + 4 sample generations both deterministic and
T=0.4 sampled.
"""
from __future__ import annotations

import time
from pathlib import Path

from rck.agent import RCKAgent
from rck.persist import save
from rck.train import recall_score


CHECKPOINT = Path("checkpoints/rck_100k")


def main() -> None:
    text = Path("data/tiny_shakespeare.txt").read_text(encoding="utf-8", errors="ignore")
    train_text = text[:100_000]
    eval_text = text[100_000:102_000]

    print("=" * 64)
    print(" RCK v1.0 CALIBRATION TRAIN -- 100k chars Tiny Shakespeare")
    print("=" * 64)

    agent = RCKAgent(
        hv_dim=2048, n_columns=2, reservoir_dim=160, n_clauses=24,
        vocab_size=80, fep_rank=128, bigram_order=3, seed=0,
    )
    print(f"hv_dim={agent.hv_dim} cols={agent.n_columns} reservoir={agent.reservoir_dim} "
          f"clauses={agent.n_clauses} fep_rank={agent.fep_rank}")

    t0 = time.time()
    last_report = t0
    chunk = 10_000
    for start in range(0, len(train_text), chunk):
        agent.observe(train_text[start:start + chunk], learn=True)
        now = time.time()
        rate = chunk / (now - last_report)
        print(f"  trained {start + chunk:>6,} chars  "
              f"elapsed={now - t0:6.1f}s  rate={rate:5.0f} chars/s  "
              f"codebook={agent.codebook.size()}")
        last_report = now

    save(agent, CHECKPOINT)
    print(f"\nsaved -> {CHECKPOINT}.npz / .json")

    print("\nheld-out next-char accuracy:")
    acc = recall_score(agent, eval_text, n_eval=500)
    acc2 = recall_score(agent, eval_text[500:], n_eval=500)
    mean = (acc + acc2) / 2
    print(f"  block A: {acc:.3f}")
    print(f"  block B: {acc2:.3f}")
    print(f"  mean:    {mean:.3f}")
    print(f"  random:  {1.0 / agent.codebook.size():.3f}")
    print(f"  -> {mean * agent.codebook.size():.1f}x uniform baseline")

    print("\ndeterministic generations:")
    for prompt in ["ROMEO:", "JULIET:", "What is", "the king is "]:
        out, _ = agent.generate(prompt, max_new=80)
        print(f"  '{prompt}' -> '{''.join(str(c) for c in out)}'")

    agent.stochastic_decode = True
    agent.fep.temperature = 0.5
    print("\nT=0.5 sampled generations:")
    for prompt in ["ROMEO:", "JULIET:", "What is", "the king is "]:
        out, _ = agent.generate(prompt, max_new=80)
        print(f"  '{prompt}' -> '{''.join(str(c) for c in out)}'")


if __name__ == "__main__":
    main()
