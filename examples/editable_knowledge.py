"""Editable knowledge: surgically modify what RCK knows.

Three operations:
  forget(slot=val, OUT=output)  -- subtract this exact fact from memory
  rename(old, new)              -- the same HV now answers to a new symbol
  re-teach(slot, val, new_out)  -- store a new fact that supersedes the old

LLMs cannot do this without retraining the whole network. With VSA, these
are O(D) hypervector ops.

Run:
    python -m examples.editable_knowledge
"""
from __future__ import annotations

from rck.compose import CompositionalReasoner


def main() -> int:
    print("=" * 64)
    print(" RCK EDITABLE KNOWLEDGE SURGERY")
    print("=" * 64)

    cr = CompositionalReasoner(dim=4096, seed=0)
    cr.teach_pair("color", "red", "R")
    cr.teach_pair("color", "blue", "L")
    cr.teach_pair("shape", "ball", "o")
    cr.teach_pair("shape", "cube", "c")

    print("\nInitial knowledge:")
    for fact in cr.memory.facts():
        print(f"  {fact}")

    # ---- FORGET --------------------------------------------------------
    print("\n[1] forget the (shape=cube -> 'c') fact:")
    cr.forget_fact({"shape": "cube"}, "c")
    ans, score = cr._single_slot_answer("shape", "cube")
    other = cr._single_slot_answer("shape", "ball")
    print(f"  ask shape=cube -> {ans!r} (cos={score:.3f})   (was 'c')")
    print(f"  ask shape=ball -> {other[0]!r} (cos={other[1]:.3f})   (unchanged)")
    print(f"  facts remaining: {cr.memory.size()}")

    # ---- RENAME --------------------------------------------------------
    print("\n[2] rename codebook atom 'L' (output for blue) to 'azure':")
    cr.rename("L", "azure")
    ans, score = cr._single_slot_answer("color", "blue")
    print(f"  ask color=blue -> {ans!r} (cos={score:.3f})   (was 'L')")
    other_ans, _ = cr._single_slot_answer("color", "red")
    print(f"  ask color=red  -> {other_ans!r}   (unchanged)")

    # ---- RE-TEACH ------------------------------------------------------
    print("\n[3] re-teach (color=red -> 'crimson') without forgetting first:")
    cr.teach_pair("color", "red", "crimson")
    # Bundle now has the original (red -> R) AND the new (red -> crimson).
    # Both compete via cosine; the more recent / dominant one wins.
    top3 = cr.memory.query(cr.codebook, {"color": "red"}, "OUT", top_k=3)
    print(f"  top-3 answers for 'red':")
    for sym, sc in top3:
        print(f"    {sym!r:>10s}  cos={sc:.3f}")

    # ---- CLEAN OVERWRITE ----------------------------------------------
    print("\n[4] forget the old (red -> R), keep only (red -> crimson):")
    cr.forget_fact({"color": "red"}, "R")
    ans, score = cr._single_slot_answer("color", "red")
    print(f"  ask color=red -> {ans!r} (cos={score:.3f})")

    print("\nFinal codebook:", sorted(cr.codebook.symbols()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
