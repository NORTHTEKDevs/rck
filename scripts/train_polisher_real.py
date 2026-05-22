"""Production training entry point for the v7 polisher.

This is the script you run on a GPU to produce the real polisher
checkpoint. Loads the synthetic corpus, builds vocab, builds a model
at the requested size, trains for the requested step count, saves.

Use scripts/smoke_train.py for the CPU smoke test first.

Run:
    python scripts/train_polisher_real.py \\
        --corpus data/training_corpus.jsonl \\
        --out checkpoints/polisher_v7 \\
        --size small \\
        --steps 5000 \\
        --batch-size 32 \\
        --device cuda

See docs/TRAINING.md for detailed runbook.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rck.polisher import (
    PairDataset, PolisherConfig, PolisherModel, PolisherTokenizer,
    train_polisher,
)
from rck.polisher.training import TrainConfig, save_checkpoint


SIZE_FACTORY = {
    "tiny":   PolisherConfig.tiny,
    "small":  PolisherConfig.small,
    "medium": PolisherConfig.medium,
    "large":  PolisherConfig.large,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="train_polisher_real")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", choices=list(SIZE_FACTORY), default="small")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--max-vocab", type=int, default=16000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if not Path(args.corpus).exists():
        print(f"error: corpus not found at {args.corpus}", file=sys.stderr)
        return 1

    # 1. Read corpus + build vocab.
    print(f"Reading corpus ...")
    pairs: list[tuple[str, str]] = []
    vocab_texts: list[str] = []
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            if args.max_examples and len(pairs) >= args.max_examples:
                break
            rec = json.loads(line)
            pairs.append((rec["draft"], rec["target"]))
            vocab_texts.append(rec["draft"])
            vocab_texts.append(rec["target"])
    print(f"  {len(pairs):,} pairs loaded")

    print(f"Building vocab (max {args.max_vocab}) ...")
    tokenizer = PolisherTokenizer.from_corpus(
        vocab_texts, max_vocab=args.max_vocab, min_count=2,
    )
    print(f"  vocab: {tokenizer.vocab_size:,}")

    # 2. Dataset.
    print("Encoding dataset ...")
    dataset = PairDataset(pairs, tokenizer, max_seq_len=args.max_seq_len)
    print(f"  encoded: {len(dataset):,}")

    # 3. Model.
    config = SIZE_FACTORY[args.size](vocab_size=tokenizer.vocab_size)
    config.max_seq_len = args.max_seq_len
    model = PolisherModel(config)
    print(f"Model: {args.size} -- {model.num_parameters() / 1e6:.2f}M params, "
          f"d_model={config.d_model}, layers={config.n_layers}")

    # 4. Train.
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr, warmup_steps=args.warmup_steps,
        max_steps=args.steps, log_every=max(1, args.steps // 50),
        device=args.device, seed=args.seed,
    )

    def on_log(info):
        print(f"  step {info['step']:>6}  loss={info['loss']:.4f}  "
              f"lr={info['lr']:.5f}  t={info['elapsed_s']:.1f}s")

    print(f"Training {args.steps} steps on {args.device} "
          f"with batch={args.batch_size} lr={args.lr} ...")
    result = train_polisher(model, dataset, tokenizer,
                             config=train_cfg, on_log=on_log)

    print(f"\nDone. Final loss: {result.final_loss:.4f}, "
          f"elapsed: {result.elapsed_s:.1f}s.")

    # 5. Save.
    print(f"Saving checkpoint to {args.out} ...")
    save_checkpoint(model, tokenizer, args.out)

    print(f"\nDone. Load via:")
    print(f"  from rck.polisher import NeuralPolisher")
    print(f"  p = NeuralPolisher('{args.out}', device='{args.device}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
