"""Smoke test of the v7 training pipeline.

Trains a TINY polisher (~1M params) for 200 steps on a sampled subset
of the corpus. Verifies:
  * Tokenizer builds from corpus
  * Model forward + backward works
  * Loss decreases
  * Inference path produces output

Run:
    python scripts/smoke_train.py

Time: ~2-5 minutes on CPU. ~30s on GPU.

This is NOT the production training run. For production, use
scripts/train_polisher_real.py with `--size small` or `--size medium`
on a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rck.polisher import (
    NeuralPolisher, PairDataset, PolisherConfig, PolisherModel,
    PolisherTokenizer, train_polisher,
)
from rck.polisher.training import TrainConfig, save_checkpoint


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="smoke_train")
    p.add_argument("--corpus", default="data/training_corpus.jsonl")
    p.add_argument("--out", default="checkpoints/polisher_smoke")
    p.add_argument("--max-examples", type=int, default=2000)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    if not Path(args.corpus).exists():
        print(f"error: {args.corpus} not found. Run build_training_corpus.py first.",
              file=sys.stderr)
        return 1

    # ---- 1. Build vocab from a sample of the corpus -------------------
    print(f"Reading corpus from {args.corpus} ...")
    pairs: list[tuple[str, str]] = []
    texts_for_vocab: list[str] = []
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            if len(pairs) >= args.max_examples:
                break
            rec = json.loads(line)
            pairs.append((rec["draft"], rec["target"]))
            texts_for_vocab.append(rec["draft"])
            texts_for_vocab.append(rec["target"])

    print(f"  loaded {len(pairs):,} (draft, target) pairs")

    print("Building vocab ...")
    tokenizer = PolisherTokenizer.from_corpus(
        texts_for_vocab, max_vocab=8000, min_count=1,
    )
    print(f"  vocab size: {tokenizer.vocab_size:,}")

    # ---- 2. Build dataset ---------------------------------------------
    print("Encoding dataset ...")
    dataset = PairDataset(pairs, tokenizer, max_seq_len=128)
    print(f"  encoded {len(dataset):,} usable examples")

    # ---- 3. Build tiny model ------------------------------------------
    config = PolisherConfig.tiny(vocab_size=tokenizer.vocab_size)
    config.max_seq_len = 128
    model = PolisherModel(config)
    n_params = model.num_parameters()
    print(f"Model: tiny (~{n_params/1e6:.2f}M params)")

    # ---- 4. Train -----------------------------------------------------
    print(f"Training {args.steps} steps on {args.device} ...")
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        max_steps=args.steps,
        log_every=25,
        warmup_steps=20,
        lr=3e-3,
        device=args.device,
    )

    log_lines = []
    def on_log(info):
        log_lines.append(info)
        print(f"  step {info['step']:>4}  loss={info['loss']:.4f}  "
              f"lr={info['lr']:.5f}  t={info['elapsed_s']:.1f}s")

    result = train_polisher(model, dataset, tokenizer,
                            config=train_cfg, on_log=on_log)

    # ---- 5. Sanity check: loss decreased? -----------------------------
    if len(result.losses) > 30:
        early = sum(result.losses[:10]) / 10
        late = sum(result.losses[-10:]) / 10
        print(f"\nLoss reduction: {early:.3f} -> {late:.3f}  "
              f"(delta {early - late:+.3f})")
        if late >= early:
            print("  WARNING: loss did not decrease. Hyperparameters?")
        else:
            print("  OK: model is learning.")

    # ---- 6. Save checkpoint -------------------------------------------
    print(f"\nSaving checkpoint to {args.out} ...")
    save_checkpoint(model, tokenizer, args.out)

    # ---- 7. Test inference --------------------------------------------
    print("\nTesting inference ...")
    polisher = NeuralPolisher(weights_path=args.out, device=args.device,
                               max_new_tokens=20, temperature=0.8)
    for draft in [
        "the dog is a mammal",
        "the capital of france is paris",
        "rain causes wetness",
    ]:
        out = polisher.polish(draft)
        print(f"  draft:    {draft}")
        print(f"  polished: {out}")

    print(f"\nSmoke test complete. Final loss: {result.final_loss:.4f}, "
          f"elapsed: {result.elapsed_s:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
