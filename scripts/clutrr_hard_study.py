"""The CLUTRR-style study, run under conditions that can actually fail.

`scripts/clutrr_style_study.py` scores RCK at 100% for every chain length
k=2..6. That result is real but nearly vacuous, and the reason is structural:
it builds a **fresh agent per example containing only that example's 2-6
edges**. Finding a path between the two endpoints of a 6-edge chain is not a
reasoning task -- there is no alternative path to be wrong about, no
distractor, and no bundle crosstalk, because the knowledge base contains
nothing but the answer.

This study changes exactly one thing: **every example's edges go into ONE
cumulative knowledge base**, then each query is asked against that shared KB.
Now chain discovery has to find the right path among thousands of competing
ones, entity names recur across families, and the KB is relation-heavy -- the
conditions under which RCK's own paper (section 5.5) measured discovery rate
falling to 73%.

The symbolic control composes over the same shared edge set and should stay at
or near 100%. If it drops, the harness is wrong, not RCK.

Reproduce:
    python scripts/clutrr_hard_study.py
    python scripts/clutrr_hard_study.py --isolated   # reproduces the easy mode
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "clutrr_hard_study.json"

# Reuse the generator, kinship table and symbolic control verbatim so the two
# studies differ ONLY in KB construction.
_spec = importlib.util.spec_from_file_location(
    "clutrr_style_study", ROOT / "scripts" / "clutrr_style_study.py")
_mod = importlib.util.module_from_spec(_spec)
# Register before exec: @dataclass resolves cls.__module__ through
# sys.modules, and fails with AttributeError if the module is absent.
sys.modules["clutrr_style_study"] = _mod
_spec.loader.exec_module(_mod)

generate_dataset = _mod.generate_dataset
symbolic_infer = _mod.symbolic_infer
term_from_up_down = _mod.term_from_up_down


def compose_path(edges):
    """Compose a relation path through the same external kinship table the
    control uses. Returns the implied term, or None."""
    return symbolic_infer(edges)


def run(isolated: bool) -> dict:
    from rck.conscious_agent import ConsciousAgent

    examples = generate_dataset()
    by_k = defaultdict(list)
    for ex in examples:
        by_k[ex.k].append(ex)

    # ---- build the knowledge base -------------------------------------
    if isolated:
        shared = None
        kb_facts = None
    else:
        t0 = time.perf_counter()
        shared = ConsciousAgent(install_self=False,
                                expected_facts=sum(len(e.edges) for e in examples) * 3)
        seen = set()
        for ex in examples:
            for s, r, o in ex.edges:
                if (s, r, o) not in seen:
                    seen.add((s, r, o))
                    shared.tell(s, r, o)
        build_s = time.perf_counter() - t0
        kb_facts = shared.knowledge.size()
        print(f"  shared KB: {len(seen)} unique edges -> {kb_facts} stored facts "
              f"in {shared.knowledge.n_shards} shards ({build_s:.1f}s)")

    rows = []
    agg = defaultdict(lambda: {"n": 0, "sym": 0, "rck": 0, "found": 0})

    for ex in examples:
        truth = ex.pattern

        # symbolic control over the example's own edges (unchanged)
        sym = symbolic_infer(ex.edges)

        if isolated:
            agent = ConsciousAgent(install_self=False)
            for s, r, o in ex.edges:
                agent.tell(s, r, o)
        else:
            agent = shared

        spec = agent.discover(ex.start, ex.end, max_depth=ex.k)
        found = bool(spec and spec.get("relations"))
        rck_term = None
        if found:
            # Rebuild the (s,r,o) walk RCK claims, then compose it through
            # the SAME table the control uses. RCK is never told the answer.
            # discover() returns the walk in `trace` as (s, r, o, conf)
            # tuples -- there is no `nodes`/`path` key.
            trace = spec.get("trace") or []
            walk = [(t[0], t[1], t[2]) for t in trace if len(t) >= 3]
            if len(walk) == len(spec["relations"]):
                rck_term = compose_path(walk)

        a = agg[ex.k]
        a["n"] += 1
        a["sym"] += (sym == truth)
        a["found"] += found
        a["rck"] += (rck_term == truth)
        rows.append({"id": ex.id, "k": ex.k, "truth": truth,
                     "symbolic": sym, "rck": rck_term, "discovered": found})

    return {"mode": "isolated" if isolated else "cumulative",
            "kb_facts": kb_facts, "n": len(examples),
            "by_k": {str(k): dict(v) for k, v in sorted(agg.items())},
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--isolated", action="store_true",
                    help="reproduce the easy per-example-agent mode")
    args = ap.parse_args()

    mode = "ISOLATED (one agent per example -- the easy mode)" if args.isolated \
        else "CUMULATIVE (one shared KB for every example)"
    print(f"\nCLUTRR-style, locally generated. NOT official CLUTRR.")
    print(f"Mode: {mode}\n")

    res = run(args.isolated)

    print(f"\n  {'k':>2} {'n':>5} {'symbolic':>10} {'RCK found':>11} {'RCK correct':>12}")
    for k, v in res["by_k"].items():
        n = v["n"]
        print(f"  {k:>2} {n:>5} {v['sym']/n*100:>9.1f}% "
              f"{v['found']/n*100:>10.1f}% {v['rck']/n*100:>11.1f}%")

    tot = res["n"]
    s = sum(v["sym"] for v in res["by_k"].values())
    f = sum(v["found"] for v in res["by_k"].values())
    r = sum(v["rck"] for v in res["by_k"].values())
    print(f"\n  symbolic control : {s}/{tot} = {s/tot*100:.1f}%   (harness check)")
    print(f"  RCK found a path : {f}/{tot} = {f/tot*100:.1f}%")
    print(f"  RCK correct term : {r}/{tot} = {r/tot*100:.1f}%")

    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
