"""Dataset for the polisher.

Loads (draft, target) JSONL produced by
`rck.polisher_training.write_corpus_jsonl` or
`scripts/build_training_corpus.py`. Each example is encoded as

    <bos> draft <sep> target <eos>

Targets are the same tokens shifted by 1 (next-token loss). A loss
MASK zeroes the loss over the draft and separator tokens, so the model
only learns to predict the polished half.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from rck.polisher.tokenizer import (
    PAD_ID, BOS_ID, EOS_ID, SEP_ID, PolisherTokenizer,
)


class PairDataset(Dataset):
    """Holds encoded (draft, target) pairs and produces training tensors."""

    def __init__(self, pairs: list[tuple[str, str]],
                 tokenizer: PolisherTokenizer,
                 max_seq_len: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self._examples: list[tuple[list[int], int]] = []
        for draft, target in pairs:
            ids, sep_idx = tokenizer.encode_pair(draft, target)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
                if sep_idx >= max_seq_len:
                    continue  # draft alone overflows; skip
            self._examples.append((ids, sep_idx))

    @classmethod
    def from_jsonl(cls, path: str | Path,
                   tokenizer: PolisherTokenizer,
                   max_seq_len: int = 256,
                   max_examples: int | None = None) -> "PairDataset":
        pairs: list[tuple[str, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                draft = rec.get("draft") or rec.get("input")
                target = rec.get("target") or rec.get("output")
                if draft is None or target is None:
                    continue
                pairs.append((str(draft), str(target)))
                if max_examples is not None and len(pairs) >= max_examples:
                    break
        return cls(pairs, tokenizer, max_seq_len=max_seq_len)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict:
        ids, sep_idx = self._examples[idx]
        input_ids = ids[:-1]
        targets = ids[1:]
        # Loss mask: 1 for positions where we want loss (target tokens),
        # 0 for positions inside the draft.
        loss_mask = [0] * len(input_ids)
        for i in range(len(input_ids)):
            # The target at position i predicts ids[i+1]. We want loss only
            # when the position is INSIDE the target half (i.e. i >= sep_idx).
            if i >= sep_idx:
                loss_mask[i] = 1
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "targets":   torch.tensor(targets, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
        }


def collate_pad(batch: list[dict], pad_id: int = PAD_ID) -> dict:
    """Pad a batch of variable-length examples to the longest in the batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    targets = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        L = item["input_ids"].size(0)
        input_ids[i, :L] = item["input_ids"]
        targets[i, :L] = item["targets"]
        loss_mask[i, :L] = item["loss_mask"]
    return {"input_ids": input_ids, "targets": targets,
            "loss_mask": loss_mask}
