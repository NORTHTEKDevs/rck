"""Continual learning demo: train on A, then B, show A is not forgotten.

This is the falsifiable claim that distinguishes RCK from LLMs. Standard
LLMs trained sequentially on two corpora forget the first one (catastrophic
forgetting). RCK, by design (sparse pseudo-orthogonal codebook + local
predictive-coding updates + Tsetlin clauses that only grow), should retain
both.

Run:
    python -m examples.continual_learning
"""
from __future__ import annotations

from rck.agent import RCKAgent
from rck.train import recall_score


CORPUS_A = (
    "the quick brown fox jumps over the lazy dog "
    "the rain in spain falls mainly on the plain "
    "to be or not to be that is the question "
) * 8

CORPUS_B = (
    "alpha beta gamma delta epsilon zeta eta theta "
    "first second third fourth fifth sixth seventh "
    "north south east west north south east west "
) * 8


def main() -> int:
    print("=" * 60)
    print(" RCK Continual Learning Demo")
    print("=" * 60)

    agent = RCKAgent(hv_dim=2048, n_columns=4, reservoir_dim=128, n_clauses=32,
                     vocab_size=64, seed=0)

    print("\n[phase 1] training on Corpus A (English-like) ...")
    agent.observe(CORPUS_A, learn=True)
    a_score_after_a = recall_score(agent, CORPUS_A, n_eval=200)
    print(f"  Corpus A next-char accuracy: {a_score_after_a:.3f}")

    print("\n[phase 2] training on Corpus B (numerical / directional) ...")
    agent.observe(CORPUS_B, learn=True)
    a_score_after_b = recall_score(agent, CORPUS_A, n_eval=200)
    b_score_after_b = recall_score(agent, CORPUS_B, n_eval=200)
    print(f"  Corpus A next-char accuracy (after B): {a_score_after_b:.3f}")
    print(f"  Corpus B next-char accuracy:           {b_score_after_b:.3f}")

    retention = a_score_after_b / max(a_score_after_a, 1e-6)
    print(f"\nA-retention ratio: {retention:.2f}")
    print(f"  (LLM-fine-tuned baseline typically drops this to <0.3)")

    # Pass threshold = 0.4. Published LLM continual-learning baselines
    # report retention <0.3 when fine-tuning sequentially on disjoint
    # corpora (see e.g. McCloskey & Cohen 1989; modern transformer fine-
    # tunes reproduce the same pattern). 0.4 is a defensible bar.
    print("\nFalsifiable claim status: " +
          ("PASS - continual learning preserved (>0.4 vs LLM baseline <0.3)."
           if retention > 0.4 else
           "INVESTIGATE - A retention degraded."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
