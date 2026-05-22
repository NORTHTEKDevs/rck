"""Empirical capacity study for sparse-binary HRR substrate.

Maps recall@1 vs (D, k, n_facts) for the sparse KB substrate. Produces
the operational target_max_fill table the analog of dense's
capacity_study.json -- enabling shard_sizing to auto-tune SPARSE KBs.

Output:
  data/sparse_capacity_study.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from string import ascii_lowercase

# Make `rck` importable when running this script directly.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rck.sparse_relational import SparseRelationalMemory
from rck.sparse_hrr import SparseCodebook


def _gen_symbols(n: int) -> list[str]:
    """Generate n distinct deterministic symbols."""
    if n <= 26:
        return [f"sym_{c}" for c in ascii_lowercase[:n]]
    out: list[str] = []
    i = 0
    while len(out) < n:
        out.append(f"sym_{i:06d}")
        i += 1
    return out


def measure_recall(dim: int, k: int, n_facts: int, *, seed: int = 0,
                   n_relations: int = 4) -> dict:
    """Store `n_facts` random triples and measure recall@1 on each."""
    cb = SparseCodebook(dim=dim, k=k, seed=seed)
    mem = SparseRelationalMemory(
        dim=dim, k=k, seed=seed, role_names=("S", "R", "O"),
    )

    # Generate distinct subjects/objects; reuse a small relation set.
    subjects = _gen_symbols(n_facts)
    objects = [f"obj_{i:06d}" for i in range(n_facts)]
    relations = [f"rel_{i}" for i in range(n_relations)]

    triples: list[tuple[str, str, str]] = []
    for i in range(n_facts):
        s = subjects[i]
        r = relations[i % n_relations]
        o = objects[i]
        triples.append((s, r, o))
        cb.encode(s); cb.encode(r); cb.encode(o)
        mem.store(cb, {"S": s, "R": r, "O": o})

    t0 = time.perf_counter()
    correct = 0
    for s, r, o in triples:
        ans, _ = mem.answer(cb, {"S": s, "R": r}, "O")
        if ans == o:
            correct += 1
    elapsed = time.perf_counter() - t0

    return {
        "dim": dim, "k": k, "n_facts": n_facts,
        "recall_at_1": correct / max(1, n_facts),
        "correct": correct,
        "elapsed_s": elapsed,
        "memory_bytes": int(mem.memory_bytes()),
        "codebook_atoms": cb.size(),
    }


def find_cliff(dim: int, k: int, *,
               targets: list[int] | None = None,
               min_recall: float = 0.90) -> int:
    """Largest n_facts that still hits recall >= min_recall."""
    if targets is None:
        targets = [10, 20, 40, 80, 120, 160, 200, 240, 280, 320, 400, 500]
    last_good = 0
    for n in targets:
        r = measure_recall(dim, k, n)
        print(f"   D={dim} k={k} n={n:>4}  recall={r['recall_at_1']:.2%}  "
              f"({r['elapsed_s']:.2f}s)")
        if r["recall_at_1"] >= min_recall:
            last_good = n
        else:
            break
    return last_good


def main() -> int:
    print("=" * 70)
    print(" SPARSE HRR CAPACITY STUDY")
    print("=" * 70)
    print()

    configurations = [
        # (dim, k)
        (4096, 80),
        (4096, 160),
        (8192, 160),
        (8192, 320),
        (16384, 320),
    ]

    results: list[dict] = []
    cliffs: dict[str, int] = {}
    for dim, k in configurations:
        print(f"\n[D={dim}, k={k}]")
        cliff = find_cliff(dim, k)
        cliffs[f"D{dim}_k{k}"] = cliff
        # Record one detailed datapoint at the cliff.
        detail = measure_recall(dim, k, cliff if cliff > 0 else 10)
        results.append(detail)

    print("\n" + "=" * 70)
    print(" CAPACITY CLIFFS (max n_facts at recall >= 90%)")
    print("=" * 70)
    for key, cliff in cliffs.items():
        print(f"  {key}: {cliff} facts")

    # Memory comparison vs dense.
    print("\n" + "=" * 70)
    print(" MEMORY COMPARISON vs dense bipolar (per fact)")
    print("=" * 70)
    for dim, k in configurations:
        # Dense bipolar: D bytes per HV (int8). A fact takes O(D) since memory
        # is one D-length tensor regardless of count.
        # Sparse: D * 4 bytes for counts tensor (int32) regardless of count.
        # Per-fact RAM cost dominates by codebook+overhead for tiny n_facts.
        dense_per_atom = dim       # int8 bipolar
        sparse_per_atom = k * 4    # int32 indices
        ratio = dense_per_atom / sparse_per_atom
        print(f"  D={dim} k={k}: dense atom = {dense_per_atom}B, "
              f"sparse atom = {sparse_per_atom}B, ratio = {ratio:.2f}x")

    out = {
        "configurations": [{"dim": d, "k": k} for d, k in configurations],
        "cliffs_at_90pct_recall": cliffs,
        "details": results,
    }
    out_path = Path("data/sparse_capacity_study.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
