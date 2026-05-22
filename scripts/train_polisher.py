"""Training entry point for the v7 distilled polisher.

This script generates the synthetic training corpus from the existing
RCK KB and trains a small (50-100M-param) transformer to map
(template draft) -> (polished paraphrase). The trained model becomes
the v7 fluency layer for the Inverted Architecture.

USAGE -- this is a SKETCH because actually running it needs PyTorch +
transformers + a GPU; we ship the structure here, you (or a CI job)
run the actual fit.

  # 1. Generate the corpus from a loaded KB.
  python scripts/train_polisher.py generate \\
      --out data/polisher_corpus.jsonl \\
      --examples-per-triple 5

  # 2. Train the polisher (requires torch + transformers + GPU).
  python scripts/train_polisher.py train \\
      --corpus data/polisher_corpus.jsonl \\
      --out checkpoints/polisher_v7.pt \\
      --model-size 80M --epochs 3

  # 3. Use it.
  python -c "from rck.polisher_training import NeuralPolisher; \\
             p = NeuralPolisher('checkpoints/polisher_v7.pt'); \\
             print(p.polish('the dog is a mammal'))"

Estimated cost: ~$50-150 on rented A100 for the 80M model and a
1B-token corpus.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rck.bulk_ingest import bulk_load_jsonl
from rck.knowledge_base import ShardedKnowledgeBase
from rck.polisher_training import write_corpus_jsonl


def cmd_generate(args: argparse.Namespace) -> int:
    kb = ShardedKnowledgeBase(dim=4096, n_shards=128, seed=0)
    for fp in args.kb:
        bulk_load_jsonl(kb, fp, symmetrize=True)
    print(f"Loaded {kb.size():,} facts. Generating corpus -> {args.out}")
    stats = write_corpus_jsonl(
        kb, args.out,
        examples_per_triple=args.examples_per_triple,
        max_triples=args.max_triples,
    )
    print(f"Wrote {stats['examples']:,} examples to {stats['path']}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Stub. Real training calls into transformers.Trainer.

    For now we print the planned training command. v7's first real
    deliverable is to make this actually run on a rented GPU.
    """
    print(
        f"[stub] Would train a {args.model_size}-param transformer:"
        f"\n  corpus  = {args.corpus}"
        f"\n  out     = {args.out}"
        f"\n  epochs  = {args.epochs}"
        f"\n  batch   = {args.batch_size}"
        f"\n  lr      = {args.lr}"
        f"\nReal training requires torch + transformers and a GPU."
        f"\nThe model is a small GPT-style decoder trained on (draft -> target)"
        f"\npairs. After training the polisher loads with"
        f"\n    from rck.polisher_training import NeuralPolisher"
        f"\n    p = NeuralPolisher('{args.out}')"
        f"\nand replaces the v4 RuleBasedPolisher in the Inverted Architecture."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="train_polisher")
    sub = p.add_subparsers(dest="cmd", required=True)

    gp = sub.add_parser("generate", help="produce synthetic corpus")
    gp.add_argument("--out", required=True)
    gp.add_argument("--kb", nargs="+", default=[
        "data/commonsense_kb.jsonl",
        "data/extended_kb.jsonl",
        "data/massive_kb.jsonl",
    ])
    gp.add_argument("--examples-per-triple", type=int, default=3)
    gp.add_argument("--max-triples", type=int, default=None)
    gp.set_defaults(func=cmd_generate)

    tp = sub.add_parser("train", help="train the polisher (stub)")
    tp.add_argument("--corpus", required=True)
    tp.add_argument("--out", required=True)
    tp.add_argument("--model-size", default="80M")
    tp.add_argument("--epochs", type=int, default=3)
    tp.add_argument("--batch-size", type=int, default=64)
    tp.add_argument("--lr", type=float, default=3e-4)
    tp.set_defaults(func=cmd_train)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
