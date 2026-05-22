"""Train at meaningful scale and report numbers + a sample generation.

20k chars Tiny Shakespeare, slightly bigger hyperparameters than the
default. Reports next-char accuracy on a fresh held-out 2k chars.
"""
from __future__ import annotations

import time
from pathlib import Path

from rck.agent import RCKAgent
from rck.train import recall_score


def main() -> None:
    text = Path("data/tiny_shakespeare.txt").read_text(encoding="utf-8", errors="ignore")
    train_text = text[:20_000]
    eval_text = text[20_000:22_000]

    print("=" * 64)
    print(" RCK v0.2 BIG TRAIN -- 20k chars Tiny Shakespeare")
    print("=" * 64)

    agent = RCKAgent(
        hv_dim=1536, n_columns=3, reservoir_dim=128, n_clauses=16,
        vocab_size=80, fep_rank=96, bigram_order=3, seed=0,
    )
    print(f"hv_dim={agent.hv_dim} columns={agent.n_columns} reservoir={agent.reservoir_dim} "
          f"clauses={agent.n_clauses} fep_rank={agent.fep_rank}")

    t0 = time.time()
    agent.observe(train_text, learn=True)
    print(f"\ntrained in {time.time() - t0:.1f}s, codebook={agent.codebook.size()}")

    acc1 = recall_score(agent, eval_text, n_eval=500)
    acc2 = recall_score(agent, eval_text[500:], n_eval=500)
    print(f"\nheld-out next-char top-1 accuracy:")
    print(f"  block A: {acc1:.3f}")
    print(f"  block B: {acc2:.3f}")
    print(f"  mean:    {(acc1 + acc2) / 2:.3f}")
    print(f"  random:  {1.0 / agent.codebook.size():.3f}")
    mult = (acc1 + acc2) / 2 / (1.0 / agent.codebook.size())
    print(f"  -> {mult:.1f}x uniform baseline")

    print("\nsample generations (deterministic):")
    for prompt in ["ROMEO:", "JULIET:", "What is", "the king"]:
        out, _ = agent.generate(prompt, max_new=60)
        text_out = "".join(str(c) for c in out)
        print(f"  '{prompt}' -> '{text_out}'")

    # Toggle sampling to break loops.
    agent.stochastic_decode = True
    agent.fep.temperature = 0.4
    print("\nsample generations (T=0.4 sampling):")
    for prompt in ["ROMEO:", "JULIET:", "What is", "the king"]:
        out, _ = agent.generate(prompt, max_new=60)
        text_out = "".join(str(c) for c in out)
        print(f"  '{prompt}' -> '{text_out}'")


if __name__ == "__main__":
    main()
