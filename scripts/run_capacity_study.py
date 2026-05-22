"""Run the capacity study and write findings to disk.

This is the actual experiment that informs the v12 docs. We sweep
D x n_shards x n_facts and report:
  * recall@1 (the hard metric)
  * recall@3 (does it ever come back in top-K)
  * mean top-1 cosine (signal strength)
  * shard fill stats (load balance)
  * timing (load + query)

Output: data/capacity_study.json with the full data + a markdown
summary in docs/design/CAPACITY-STUDY-2026-05-21.md.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from rck.capacity_profiler import profile, sweep, shard_sweep, dim_sweep


def main() -> int:
    t0 = time.time()

    results: dict = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": [],
    }

    print("=" * 64)
    print(" HRR CAPACITY STUDY")
    print("=" * 64)

    # --- Experiment 1: vary n_facts at D=4096, 64 shards
    print("\n[1] Sweep n_facts at D=4096, 64 shards ...")
    n_list = [500, 1000, 2000, 4000, 8000, 16000]
    curve = sweep(n_list, dim=4096, n_shards=64)
    for r in curve:
        print(f"   {r.n_facts:>6}  r@1={r.recall_at_1:.3f}  r@3={r.recall_at_3:.3f}  "
              f"mean_cos={r.mean_top1_cos:.3f}  p10={r.p10_top1_cos:.3f}  "
              f"shard_max={r.fill_per_shard_max}")
    results["experiments"].append({
        "name": "n_facts at D=4096 shards=64",
        "results": [asdict(r) for r in curve],
    })

    # --- Experiment 2: vary shards at D=4096, n=8000 facts
    print("\n[2] Sweep n_shards at D=4096, n=8000 facts ...")
    s_list = [4, 8, 16, 32, 64, 128, 256, 512]
    curve = shard_sweep(8000, dim=4096, shards_list=s_list)
    for r in curve:
        print(f"   {r.n_shards:>4}  r@1={r.recall_at_1:.3f}  "
              f"mean_cos={r.mean_top1_cos:.3f}  "
              f"max_fill={r.fill_per_shard_max}")
    results["experiments"].append({
        "name": "n_shards at D=4096 n_facts=8000",
        "results": [asdict(r) for r in curve],
    })

    # --- Experiment 3: vary D at fixed n=4000, shards=64
    print("\n[3] Sweep D at n=4000, 64 shards ...")
    d_list = [512, 1024, 2048, 4096, 8192]
    curve = dim_sweep(4000, n_shards=64, dim_list=d_list)
    for r in curve:
        print(f"   D={r.dim:>5}  r@1={r.recall_at_1:.3f}  "
              f"mean_cos={r.mean_top1_cos:.3f}")
    results["experiments"].append({
        "name": "D at n_facts=4000 shards=64",
        "results": [asdict(r) for r in curve],
    })

    # --- Experiment 4: 10k facts -- common operational config
    print("\n[4] Spot-check: 10000 facts at D=4096, 128 shards ...")
    r10k = profile(10000, dim=4096, n_shards=128)
    print(f"   r@1={r10k.recall_at_1:.3f}  r@3={r10k.recall_at_3:.3f}  "
          f"load={r10k.elapsed_load_s:.2f}s  query={r10k.elapsed_query_s:.2f}s")
    results["experiments"].append({
        "name": "10k facts at D=4096 shards=128",
        "results": [asdict(r10k)],
    })

    # --- Experiment 5: high-density config: 20k facts at 256 shards
    print("\n[5] High-density: 20000 facts at D=4096, 256 shards ...")
    r20k = profile(20000, dim=4096, n_shards=256)
    print(f"   r@1={r20k.recall_at_1:.3f}  r@3={r20k.recall_at_3:.3f}  "
          f"load={r20k.elapsed_load_s:.2f}s  query={r20k.elapsed_query_s:.2f}s")
    results["experiments"].append({
        "name": "20k facts at D=4096 shards=256",
        "results": [asdict(r20k)],
    })

    results["elapsed_s"] = time.time() - t0
    out = Path("data/capacity_study.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote raw data to {out}")
    print(f"Total elapsed: {results['elapsed_s']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
