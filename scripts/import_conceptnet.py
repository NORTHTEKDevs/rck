"""Run-once import script for ConceptNet 5.7.

Steps:
  1. Download the data:
       https://github.com/commonsense/conceptnet5/wiki/Downloads
       (file: conceptnet-assertions-5.7.0.csv.gz, ~700MB)
  2. Decompress:
       gunzip conceptnet-assertions-5.7.0.csv.gz
  3. Run this script:
       python scripts/import_conceptnet.py \\
           --tsv conceptnet-assertions-5.7.0.csv \\
           --out data/conceptnet_imported.jsonl \\
           --language en --min-weight 1.5

The script:
  - parses the TSV, maps relations, filters to English
  - writes a JSONL of (s, r, o) for portability
  - optionally loads directly into a sharded KB and saves it

Output: ~1-2M English assertions for `--min-weight 1.5` (the higher the
weight the more reliable the assertion).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rck.conceptnet_loader import parse_conceptnet_tsv


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="import_conceptnet")
    p.add_argument("--tsv", required=True,
                   help="path to conceptnet-assertions-5.7.0.csv")
    p.add_argument("--out", required=True,
                   help="JSONL output path")
    p.add_argument("--language", default="en")
    p.add_argument("--min-weight", type=float, default=1.0)
    p.add_argument("--max-rows", type=int, default=None,
                   help="cap total rows to read")
    args = p.parse_args(argv)

    tsv_path = Path(args.tsv)
    if not tsv_path.exists():
        print(f"error: {tsv_path} does not exist", file=sys.stderr)
        print("Download from https://github.com/commonsense/conceptnet5/wiki/Downloads",
              file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out_path, "w", encoding="utf-8") as outf:
        for s, r, o, weight in parse_conceptnet_tsv(
            tsv_path, language=args.language,
            min_weight=args.min_weight, max_rows=args.max_rows,
        ):
            outf.write(json.dumps({"s": s, "r": r, "o": o,
                                    "w": weight}) + "\n")
            n += 1
            if n % 100_000 == 0:
                print(f"  wrote {n:,} so far")
    print(f"\nDone. {n:,} triples written to {out_path}")
    print(f"Next: load with `bulk_load_jsonl(kb, {out_path!s}, ...)` ")
    print(f"  or  `bulk_load_jsonl(agent.knowledge, {out_path!s})`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
