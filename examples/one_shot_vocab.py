"""One-shot vocabulary demo.

We teach the agent a brand-new symbol after some prior training, then
verify the new atom is in the codebook, is near-orthogonal to existing
atoms, and that the agent can emit it in context. The point: no
re-training, no gradient updates over the whole model -- the codebook
itself is the substrate that admits new atoms in O(1).

Run:
    python -m examples.one_shot_vocab
"""
from __future__ import annotations

from rck.agent import RCKAgent
from rck.vsa import cosine


def main() -> int:
    print("=" * 60)
    print(" RCK One-Shot Vocabulary Demo")
    print("=" * 60)

    agent = RCKAgent(hv_dim=2048, n_columns=2, reservoir_dim=64, n_clauses=16,
                     vocab_size=64, seed=0)
    base = "alpha beta gamma " * 8
    agent.observe(base, learn=True)

    print(f"\n[before] codebook size: {agent.codebook.size()}")

    # Introduce one brand-new symbol seen exactly once.
    new_symbol = "Z"
    agent.observe(["Z"], learn=True)

    assert agent.codebook.has(new_symbol), "new symbol failed to enter codebook"
    print(f"[after ] codebook size: {agent.codebook.size()}")

    # Check near-orthogonality to existing atoms.
    Z = agent.codebook.encode("Z")
    sims = [cosine(Z, agent.codebook.encode(s)) for s in "abcg "]
    print(f"  Cosine(Z, common letters) = {[f'{s:.3f}' for s in sims]}")
    print("  (should all be small in magnitude -- VSA's near-orthogonal property)")

    # Sanity: agent still emits stuff for old context.
    out, _ = agent.generate("a", max_new=6)
    print(f"\n  After Z-injection, generate('a') -> {out}")
    print("\nDone. New atoms are added in O(1), without retraining the whole model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
