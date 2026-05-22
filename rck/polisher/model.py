"""Small transformer decoder for the polisher.

Decoder-only (GPT-style). Trained on `<bos> draft <sep> target <eos>`
sequences with causal masking. At inference, prefix is `<bos> draft <sep>`
and we sample tokens until `<eos>`.

Sizes (configurable):
    tiny:   1M params  (d=64, layers=2, heads=4) -- smoke test
    small:  5M params  (d=128, layers=4, heads=4)
    medium: 25M params (d=256, layers=6, heads=8)
    large:  80M params (d=512, layers=8, heads=8) -- production
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PolisherConfig:
    vocab_size: int = 8000
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int | None = None       # defaults to 4 * d_model
    max_seq_len: int = 256
    dropout: float = 0.1
    pad_id: int = 0

    @classmethod
    def tiny(cls, vocab_size: int) -> "PolisherConfig":
        return cls(vocab_size=vocab_size, d_model=64, n_layers=2,
                   n_heads=4, max_seq_len=128, dropout=0.0)

    @classmethod
    def small(cls, vocab_size: int) -> "PolisherConfig":
        return cls(vocab_size=vocab_size, d_model=128, n_layers=4,
                   n_heads=4, max_seq_len=256, dropout=0.1)

    @classmethod
    def medium(cls, vocab_size: int) -> "PolisherConfig":
        return cls(vocab_size=vocab_size, d_model=256, n_layers=6,
                   n_heads=8, max_seq_len=384, dropout=0.1)

    @classmethod
    def large(cls, vocab_size: int) -> "PolisherConfig":
        return cls(vocab_size=vocab_size, d_model=512, n_layers=8,
                   n_heads=8, max_seq_len=512, dropout=0.1)

    @property
    def feedforward_dim(self) -> int:
        return self.d_ff if self.d_ff is not None else 4 * self.d_model


class CausalSelfAttention(nn.Module):
    def __init__(self, config: PolisherConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.drop = nn.Dropout(config.dropout)
        # Static causal mask, registered as buffer (no gradient).
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
            1, 1, config.max_seq_len, config.max_seq_len)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)   # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)
        # split heads
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0,
                              float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v   # (B, n_heads, T, d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, config: PolisherConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.feedforward_dim),
            nn.GELU(),
            nn.Linear(config.feedforward_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PolisherModel(nn.Module):
    """Small GPT-style decoder for surface-fluency polish."""

    def __init__(self, config: PolisherConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model,
                                      padding_idx=config.pad_id)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config)
                                     for _ in range(config.n_layers)])
        self.ln_final = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Tie weights for parameter efficiency.
        self.head.weight = self.token_emb.weight

        # init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor,
                targets: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None) -> dict:
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f"sequence too long: {T} > {self.config.max_seq_len}"
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.head(x)  # (B, T, V)

        out: dict = {"logits": logits}
        if targets is not None:
            # Cross-entropy ignoring pad positions and (optionally) the
            # draft side of the sequence.
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=self.config.pad_id,
                reduction="none",
            ).view(B, T)
            if loss_mask is not None:
                loss = loss * loss_mask.float()
                denom = loss_mask.float().sum().clamp(min=1.0)
            else:
                denom = (targets != self.config.pad_id).float().sum().clamp(min=1.0)
            out["loss"] = loss.sum() / denom
        return out
