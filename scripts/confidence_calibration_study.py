"""Confidence calibration study: what is a cosine actually worth?

Raw HRR cleanup cosines are similarity features, not probabilities --
a clean retrieval at ~0.7 cosine is nearly always correct. This study
measures the empirical mapping cosine -> P(correct), fits a monotone
calibrator (isotonic regression / PAV), and shows that plain
multiplication of CALIBRATED per-hop probabilities restores honest,
length-sensitive chain confidence (the `calibrated_product` rule).

Protocol:
  1. TRAIN: synthetic KBs at five bundle loads (clean -> saturated),
     querying stored facts back (label: top-1 in the valid-object set)
     plus never-stored probes (label: False). Deterministic seeds.
  2. FIT: pool-adjacent-violators isotonic regression.
  3. EVAL (held out): the real commonsense KB, same labeling. Report a
     reliability table and Brier scores (raw cosine as probability vs
     calibrated probability).
  4. DEPTH: re-run the 50-hop linear-chain study under
     calibrated_product, alongside geometric_mean for contrast.

Output: data/confidence_calibration_study.json (includes the fitted
calibrator, loadable via ScoreCalibrator.from_dict(doc["calibrator"])).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from rck.bulk_ingest import bulk_load_triples
from rck.capacity_profiler import _synthetic_triples
from rck.chain_walker import Hop, walk_chain
from rck.confidence_propagation import PropagationConfig
from rck.knowledge_base import ShardedKnowledgeBase
from rck.score_calibration import ScoreCalibrator
from rck.shard_sizing import auto_shard_for_kb

TRAIN_LOADS = [500, 2000, 4000, 8000, 12000]  # facts at 64 shards, D=4096
MAX_POSITIVE_QUERIES = 800
N_ABSENT_PROBES = 200
DEPTHS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]


def valid_object_sets(kb: ShardedKnowledgeBase) -> dict[tuple[str, str], set[str]]:
    """Ground truth from the shards' fact logs: every valid O per (S, R).
    Multi-valued relations (dog has fur/legs/tail) make set membership,
    not equality, the correct label."""
    valid: dict[tuple[str, str], set[str]] = defaultdict(set)
    for shard in kb._shards:
        for f in shard.facts():
            valid[(str(f["S"]), str(f["R"]))].add(str(f["O"]))
    return dict(valid)


def collect_samples(kb: ShardedKnowledgeBase, *, seed: int,
                    max_positive: int, n_absent: int
                    ) -> tuple[list[float], list[bool]]:
    rng = np.random.default_rng(seed)
    valid = valid_object_sets(kb)
    keys = sorted(valid.keys())
    if len(keys) > max_positive:
        idx = rng.choice(len(keys), size=max_positive, replace=False)
        keys = [keys[i] for i in sorted(idx)]

    scores: list[float] = []
    correct: list[bool] = []
    for s, r in keys:
        ans, score = kb.answer({"S": s, "R": r}, "O")
        scores.append(float(score))
        correct.append(ans is not None and str(ans) in valid[(s, r)])

    subjects = sorted({s for s, _ in valid.keys()})
    for i in range(n_absent):
        s = subjects[int(rng.integers(0, len(subjects)))]
        ans, score = kb.answer({"S": s, "R": f"neverstored_{i}"}, "O")
        scores.append(float(score))
        correct.append(False)  # nothing stored -> any answer is wrong
    return scores, correct


def brier(scores: list[float], correct: list[bool],
          prob_fn) -> float:
    ps = np.array([prob_fn(s) for s in scores])
    ys = np.array(correct, dtype=np.float64)
    return float(np.mean((ps - ys) ** 2))


def main() -> int:
    print("=" * 70)
    print(" CONFIDENCE CALIBRATION STUDY (cosine -> P(correct))")
    print("=" * 70)

    # --- 1. training samples across bundle loads ------------------------
    train_scores: list[float] = []
    train_correct: list[bool] = []
    for n_facts in TRAIN_LOADS:
        kb = ShardedKnowledgeBase(dim=4096, n_shards=64, seed=0)
        bulk_load_triples(kb, _synthetic_triples(n_facts, seed=n_facts))
        s, c = collect_samples(kb, seed=n_facts,
                               max_positive=MAX_POSITIVE_QUERIES,
                               n_absent=N_ABSENT_PROBES)
        acc = sum(c) / len(c)
        print(f"  load {n_facts:>6} facts: {len(s)} samples, "
              f"accuracy {acc:.1%}, "
              f"mean cosine {np.mean(s):.3f}")
        train_scores += s
        train_correct += c

    # --- 2. fit ----------------------------------------------------------
    cal = ScoreCalibrator.fit(train_scores, train_correct)
    print(f"\nFitted PAV calibrator on {cal.n_samples} samples "
          f"({len(cal.probs)} monotone blocks)")

    # --- 3. held-out eval on the real commonsense KB ---------------------
    triples = []
    with open("data/commonsense_kb.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            triples.append((rec["s"], rec["r"], rec["o"]))
    probe_kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(probe_kb, triples)
    n_shards = auto_shard_for_kb(probe_kb.size())
    kb = ShardedKnowledgeBase(dim=4096, n_shards=n_shards, seed=0)
    bulk_load_triples(kb, triples)
    held_scores, held_correct = collect_samples(
        kb, seed=7, max_positive=10_000, n_absent=150)

    b_raw = brier(held_scores, held_correct,
                  lambda s: max(0.0, min(1.0, s)))
    b_cal = brier(held_scores, held_correct, cal.prob)
    print(f"\nHeld-out (commonsense KB, {len(held_scores)} samples):")
    print(f"  Brier raw-cosine-as-probability: {b_raw:.4f}")
    print(f"  Brier calibrated:                {b_cal:.4f} "
          f"({(1 - b_cal / b_raw):.0%} lower)")

    reliability = cal.reliability_table(held_scores, held_correct, n_bins=10)
    print(f"\n  {'mean cos':>9} {'empirical':>10} {'calibrated':>11} {'n':>5}")
    for row in reliability:
        print(f"  {row['mean_score']:>9.3f} "
              f"{row['empirical_accuracy']:>10.1%} "
              f"{row['calibrated_prob']:>11.1%} {row['count']:>5}")

    # --- 4. chain-depth table under calibrated_product -------------------
    print("\nChain-depth table (50-node linear chain, relation `next`):")
    chain_kb = ShardedKnowledgeBase(dim=4096, n_shards=64, seed=0)
    nodes = [f"a_{i:04d}" for i in range(max(DEPTHS) + 1)]
    bulk_load_triples(chain_kb,
                      [(nodes[i], "next", nodes[i + 1])
                       for i in range(max(DEPTHS))])
    depth_rows: list[dict] = []
    configs = {
        "geometric_mean": PropagationConfig(rule="geometric_mean",
                                            chain_decay=0.95),
        "calibrated_product": PropagationConfig(rule="calibrated_product",
                                                calibrator=cal),
    }
    print(f"  {'rule':>19} {'depth':>5} {'correct':>7} {'conf':>7} {'hedge':>10}")
    for rule, cfg in configs.items():
        for depth in DEPTHS:
            res = walk_chain(chain_kb, nodes[0],
                             [Hop("next") for _ in range(depth)], config=cfg)
            row = {
                "rule": rule, "depth": depth,
                "correct": bool(res.answer == nodes[depth]),
                "confidence": float(res.confidence),
                "hedge": res.hedge,
            }
            depth_rows.append(row)
            print(f"  {rule:>19} {depth:>5} {str(row['correct']):>7} "
                  f"{row['confidence']:>7.3f} {row['hedge']:>10}")

    out_path = Path("data/confidence_calibration_study.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "train_loads": TRAIN_LOADS,
        "n_train_samples": cal.n_samples,
        "heldout_kb": "commonsense",
        "n_heldout_samples": len(held_scores),
        "brier_raw_cosine": b_raw,
        "brier_calibrated": b_cal,
        "reliability_heldout": reliability,
        "depth_table": depth_rows,
        "calibrator": cal.to_dict(),
    }, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
