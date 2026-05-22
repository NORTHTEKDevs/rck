"""RCK fluency polisher -- real PyTorch implementation.

This package replaces the v4 RuleBasedPolisher / v7 NeuralPolisher stub
with a real trainable transformer-based model. The model is small
(default ~5M params; scales to ~80M for production), trains in hours
on a single GPU, and serves as the surface-polish layer in the Inverted
Architecture.

Public API:
    from rck.polisher import (
        PolisherTokenizer, PolisherModel, PolisherConfig,
        PairDataset, train_polisher, NeuralPolisher,
    )

The trained model is the v7 deliverable. The training script in
`scripts/train_polisher_real.py` produces it; `NeuralPolisher` loads
and serves it.
"""
from rck.polisher.tokenizer import PolisherTokenizer
from rck.polisher.model import PolisherConfig, PolisherModel
from rck.polisher.dataset import PairDataset
from rck.polisher.training import train_polisher
from rck.polisher.inference import NeuralPolisher

__all__ = [
    "PolisherTokenizer",
    "PolisherConfig",
    "PolisherModel",
    "PairDataset",
    "train_polisher",
    "NeuralPolisher",
]
