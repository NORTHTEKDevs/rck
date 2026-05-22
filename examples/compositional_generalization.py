"""The killer demo: compose UNSEEN combinations of trained primitives.

Trains a RCK CompositionalReasoner on primitive (slot=value -> output)
mappings only -- never on full multi-slot compositions. At test time,
queries multi-slot compositions that DO NOT appear in training. A
char-level LLM trained on the same corpus has zero training signal for
these and must guess.

Run:
    python -m examples.compositional_generalization
"""
from __future__ import annotations

import itertools
import random

from rck.compose import CompositionalReasoner


# ---- primitive vocabulary ---------------------------------------------------

COLORS = {"red": "R", "blue": "L", "green": "G", "yellow": "Y"}
SHAPES = {"ball": "o", "cube": "c", "ring": "i", "pyramid": "p"}
QUANTITIES = {"one": "1", "two": "2", "three": "3", "four": "4"}


def main() -> int:
    print("=" * 64)
    print(" RCK COMPOSITIONAL GENERALIZATION")
    print("=" * 64)

    cr = CompositionalReasoner(dim=4096, seed=0)

    # ---- training: primitives only -----------------------------------------
    # NO MULTI-SLOT EXAMPLES are taught.
    for name, sym in COLORS.items():
        cr.teach_pair("color", name, sym)
    for name, sym in SHAPES.items():
        cr.teach_pair("shape", name, sym)
    for name, sym in QUANTITIES.items():
        cr.teach_pair("quantity", name, sym)

    n_primitives = len(COLORS) + len(SHAPES) + len(QUANTITIES)
    print(f"\nTrained {n_primitives} primitives:")
    print(f"  colours:    {list(COLORS)}")
    print(f"  shapes:     {list(SHAPES)}")
    print(f"  quantities: {list(QUANTITIES)}")
    print(f"  codebook size: {cr.codebook.size()}  memory facts: {cr.memory.size()}")

    # ---- single-slot recall (sanity) --------------------------------------
    print("\n[1] Single-slot recall (must be 100% -- these were trained):")
    correct = 0; total = 0
    for slot, vocab in (("color", COLORS), ("shape", SHAPES), ("quantity", QUANTITIES)):
        for name, expected in vocab.items():
            ans, score = cr._single_slot_answer(slot, name)
            ok = (ans == expected)
            correct += ok; total += 1
            if not ok:
                print(f"  MISS  {slot}={name!r} -> expected {expected!r}, got {ans!r}")
    print(f"  primitive accuracy: {correct}/{total}  ({correct / total:.1%})")

    # ---- compositional generalisation -------------------------------------
    # All 4*4*4 = 64 multi-slot combinations. NONE were trained.
    print("\n[2] Compositional generalisation -- 64 UNSEEN combinations of {quantity, colour, shape}:")
    print("    Render rule: concat single-slot outputs as 'quantity colour shape'")
    correct = 0; total = 0
    misses = []
    for c, s, q in itertools.product(COLORS, SHAPES, QUANTITIES):
        # Slots are passed in the rendering order we expect; compose emits
        # them in that same iteration order.
        slots = {"quantity": q, "color": c, "shape": s}
        _, rendered, score = cr.compose(slots)
        expected = f"{QUANTITIES[q]} {COLORS[c]} {SHAPES[s]}"
        ok = (rendered == expected)
        correct += ok; total += 1
        if not ok and len(misses) < 5:
            misses.append((slots, rendered, expected))
    acc = correct / total
    print(f"  composition accuracy: {correct}/{total}  ({acc:.1%})")
    print(f"  (LLM trained on the same primitives-only corpus would see 0 training examples")
    print(f"   for any of these combinations, so its baseline expectation is essentially")
    print(f"   uniform across the codebook -- well under {1/cr.codebook.size():.1%}.)")
    if misses:
        print("  example misses:")
        for slots, got, exp in misses:
            print(f"    {slots}  got={got!r}  expected={exp!r}")

    # ---- 4-slot extension --------------------------------------------------
    SIZES = {"tiny": "T", "huge": "H"}
    for name, sym in SIZES.items():
        cr.teach_pair("size", name, sym)
    print(f"\n[3] After learning {len(SIZES)} NEW primitive size atoms (one-shot each),")
    print(f"    test the 4-way composition (size x quantity x colour x shape)...")

    correct = 0; total = 0
    misses = []
    for sz, q, c, s in itertools.product(SIZES, QUANTITIES, COLORS, SHAPES):
        slots = {"size": sz, "quantity": q, "color": c, "shape": s}
        _, rendered, _ = cr.compose(slots)
        expected = f"{SIZES[sz]} {QUANTITIES[q]} {COLORS[c]} {SHAPES[s]}"
        ok = (rendered == expected)
        correct += ok; total += 1
        if not ok and len(misses) < 8:
            misses.append((slots, rendered, expected))
    acc = correct / total
    print(f"  4-way composition accuracy: {correct}/{total}  ({acc:.1%})")
    if misses:
        print("  example misses:")
        for slots, got, exp in misses:
            print(f"    {slots}  got={got!r}  expected={exp!r}")

    # ---- 5-slot stress test ------------------------------------------------
    MATERIALS = {"wood": "W", "metal": "M", "glass": "S"}
    for name, sym in MATERIALS.items():
        cr.teach_pair("material", name, sym)
    print(f"\n[4] 5-slot stress test (material x size x quantity x colour x shape) "
          f"= {len(MATERIALS) * len(SIZES) * len(QUANTITIES) * len(COLORS) * len(SHAPES)} combos:")
    correct = 0; total = 0
    for mat, sz, q, c, s in itertools.product(MATERIALS, SIZES, QUANTITIES, COLORS, SHAPES):
        slots = {"material": mat, "size": sz, "quantity": q, "color": c, "shape": s}
        _, rendered, _ = cr.compose(slots)
        expected = f"{MATERIALS[mat]} {SIZES[sz]} {QUANTITIES[q]} {COLORS[c]} {SHAPES[s]}"
        correct += (rendered == expected); total += 1
    acc = correct / total
    print(f"  5-way composition accuracy: {correct}/{total}  ({acc:.1%})")

    print("\n[5] Summary:")
    print(f"  primitives trained: {cr.memory.size()}")
    print(f"  3-slot compositions tested (all unseen): 64")
    print(f"  4-slot compositions tested (all unseen): {len(SIZES) * len(QUANTITIES) * len(COLORS) * len(SHAPES)}")
    print(f"  5-slot compositions tested (all unseen): {len(MATERIALS) * len(SIZES) * len(QUANTITIES) * len(COLORS) * len(SHAPES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
