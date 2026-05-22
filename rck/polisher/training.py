"""Real PyTorch training loop for the polisher.

This is not a stub: invoke it with a corpus path and you get a trained
checkpoint. Designed to scale from a 5-step smoke test on CPU to a
multi-day production run on an A100.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rck.polisher.dataset import PairDataset, collate_pad
from rck.polisher.model import PolisherConfig, PolisherModel
from rck.polisher.tokenizer import PolisherTokenizer


@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    log_every: int = 25
    eval_every: int | None = None
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"


@dataclass
class TrainResult:
    final_loss: float
    final_step: int
    losses: list[float] = field(default_factory=list)
    elapsed_s: float = 0.0


def _lr_at(step: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr  # constant after warmup -- simple and adequate at our scale


def train_polisher(model: PolisherModel, dataset: PairDataset,
                   tokenizer: PolisherTokenizer,
                   *, config: TrainConfig | None = None,
                   on_log: callable | None = None) -> TrainResult:
    """Train `model` on `dataset` for up to `config.max_steps` steps."""
    config = config or TrainConfig()
    torch.manual_seed(config.seed)

    device = torch.device(config.device)
    model = model.to(device)
    model.train()

    loader = DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_pad,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr, weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    losses: list[float] = []
    step = 0
    t0 = time.time()
    last_loss = math.inf

    # Loop until max_steps; iterate the loader as many times as needed.
    while step < config.max_steps:
        for batch in loader:
            if step >= config.max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            # LR schedule.
            lr = _lr_at(step, config.lr, config.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            out = model(input_ids, targets=targets, loss_mask=loss_mask)
            loss = out["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            last_loss = float(loss.item())
            losses.append(last_loss)
            step += 1

            if (step % max(1, config.log_every)) == 0 and on_log is not None:
                on_log({"step": step, "loss": last_loss, "lr": lr,
                        "elapsed_s": time.time() - t0})

    return TrainResult(
        final_loss=last_loss, final_step=step,
        losses=losses, elapsed_s=time.time() - t0,
    )


def save_checkpoint(model: PolisherModel, tokenizer: PolisherTokenizer,
                    path: str | Path) -> None:
    """Save model weights + tokenizer + config to a directory."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p / "model.pt")
    tokenizer.save(p / "tokenizer.json")
    # Save config so we can reconstruct.
    import json
    config_dict = {
        k: v for k, v in model.config.__dict__.items()
        if not k.startswith("_")
    }
    (p / "config.json").write_text(json.dumps(config_dict))


def load_checkpoint(path: str | Path,
                    map_location: str = "cpu") -> tuple[PolisherModel, PolisherTokenizer]:
    """Load model + tokenizer from a directory."""
    import json
    p = Path(path)
    config_dict = json.loads((p / "config.json").read_text())
    config = PolisherConfig(**config_dict)
    model = PolisherModel(config)
    state = torch.load(p / "model.pt", map_location=map_location,
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()
    tokenizer = PolisherTokenizer.load(p / "tokenizer.json")
    return model, tokenizer
