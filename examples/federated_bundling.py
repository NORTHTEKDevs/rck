"""Federated knowledge merge -- two agents combine via hypervector bundling.

Two CompositionalReasoners learn DISJOINT knowledge. We then merge them by
adding their memory tensors and unioning their codebooks. The merged agent
must answer correctly on BOTH knowledge sets.

This is structurally impossible with neural networks: averaging two models
trained on disjoint tasks gives a model that knows neither well. With VSA,
the bundle of two memories is an exact superposition.

Run:
    python -m examples.federated_bundling
"""
from __future__ import annotations

import copy

from rck.compose import CompositionalReasoner


def main() -> int:
    print("=" * 64)
    print(" RCK FEDERATED BUNDLING")
    print("=" * 64)

    AGENT_ALPHA_DATA = {
        ("color", "red"):       "R",
        ("color", "blue"):      "L",
        ("color", "green"):     "G",
        ("shape", "ball"):      "o",
        ("shape", "cube"):      "c",
    }
    AGENT_BETA_DATA = {
        ("color", "yellow"):    "Y",
        ("color", "purple"):    "P",
        ("shape", "ring"):      "i",
        ("shape", "pyramid"):   "p",
        ("quantity", "one"):    "1",
        ("quantity", "two"):    "2",
        ("quantity", "three"):  "3",
    }

    alpha = CompositionalReasoner(dim=4096, seed=0)
    for (slot, val), out in AGENT_ALPHA_DATA.items():
        alpha.teach_pair(slot, val, out)
    beta = CompositionalReasoner(dim=4096, seed=0)
    for (slot, val), out in AGENT_BETA_DATA.items():
        beta.teach_pair(slot, val, out)

    print(f"\n[alpha] learned: {list(AGENT_ALPHA_DATA.keys())}")
    print(f"        codebook={alpha.codebook.size()} facts={alpha.memory.size()}")
    print(f"[beta]  learned: {list(AGENT_BETA_DATA.keys())}")
    print(f"        codebook={beta.codebook.size()} facts={beta.memory.size()}")

    # Sanity: each agent knows only its own subset.
    print("\n[sanity] alpha asked beta's 'yellow':", alpha._single_slot_answer("color", "yellow"))
    print("         beta asked alpha's 'cube':   ", beta._single_slot_answer("shape", "cube"))

    # Merge.
    merged = copy.deepcopy(alpha)
    merged.merge(beta)
    print(f"\n[merged] codebook={merged.codebook.size()} facts={merged.memory.size()}")

    # Both bodies of knowledge survive.
    print("\nrecall on ALL alpha+beta primitives after merging:")
    correct = 0; total = 0
    for src, data in (("alpha", AGENT_ALPHA_DATA), ("beta", AGENT_BETA_DATA)):
        for (slot, val), expected in data.items():
            ans, score = merged._single_slot_answer(slot, val)
            ok = (ans == expected)
            correct += ok; total += 1
            mark = "OK  " if ok else "MISS"
            print(f"  [{src:>5}] {mark}  {slot}={val!r:>10s} -> got {ans!r:>4s} (cos={score:.3f}), expected {expected!r}")
    print(f"\noverall: {correct}/{total} = {correct/total:.1%}")
    return 0 if correct == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
