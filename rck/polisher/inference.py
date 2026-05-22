"""Inference path for a trained PolisherModel.

`NeuralPolisher` implements the `FluencyPolisher` interface from
`rck.inverted_lm` so it slots in as a drop-in replacement for the
v4 RuleBasedPolisher.

Supports both blocking (`polish`) and streaming (`stream_polish`)
generation. Streaming yields one decoded word at a time as it's
generated, useful for chat UIs that want to show typing.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch

from rck.polisher.model import PolisherConfig, PolisherModel
from rck.polisher.tokenizer import (
    BOS_ID, EOS_ID, PAD_ID, SEP_ID, PolisherTokenizer, basic_tokenize,
)
from rck.polisher.training import load_checkpoint


@dataclass
class NeuralPolisher:
    """Drop-in `FluencyPolisher` that uses a trained transformer."""

    weights_path: str | Path
    device: str = "cpu"
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_k: int = 40

    _model: PolisherModel = field(default=None, init=False)
    _tokenizer: PolisherTokenizer = field(default=None, init=False)

    def __post_init__(self) -> None:
        p = Path(self.weights_path)
        if not p.exists() or not (p / "model.pt").exists():
            raise FileNotFoundError(
                f"polisher checkpoint not found at {p}. "
                f"Train via scripts/train_polisher_real.py first."
            )
        self._model, self._tokenizer = load_checkpoint(p, map_location=self.device)
        self._model.eval()

    # ---- FluencyPolisher protocol ---------------------------------------

    def polish(self, draft: str, context: dict | None = None) -> str:
        if not draft.strip():
            return draft
        return self._generate(draft)

    # ---- streaming generation ------------------------------------------

    def stream_polish(self, draft: str,
                      context: dict | None = None) -> Iterator[str]:
        """Yield decoded tokens one at a time as they are generated.

        Useful for chat UIs that want a "typing" effect. Each yielded
        string is a single word (already detokenized with appropriate
        spacing/punctuation).
        """
        if not draft.strip():
            return
        yield from self._generate_stream(draft)

    @torch.no_grad()
    def _generate_stream(self, draft: str) -> Iterator[str]:
        device = torch.device(self.device)
        ids = [BOS_ID]
        ids.extend(self._tokenizer.token_to_id.get(t, 1)
                   for t in basic_tokenize(draft))
        ids.append(SEP_ID)
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        generated_so_far: list[int] = []
        max_len = self._model.config.max_seq_len
        last_text = ""
        for _ in range(self.max_new_tokens):
            if input_ids.size(1) >= max_len:
                break
            out = self._model(input_ids)
            logits = out["logits"][:, -1, :] / max(self.temperature, 1e-6)
            if self.top_k > 0:
                v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            tok_id = int(next_tok.item())
            if tok_id == EOS_ID or tok_id == PAD_ID:
                break
            generated_so_far.append(tok_id)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
            # Detokenize incrementally and yield only the new substring.
            full_text = self._tokenizer.decode(generated_so_far,
                                                strip_specials=True)
            delta = full_text[len(last_text):]
            last_text = full_text
            if delta:
                yield delta

    # ---- generation -----------------------------------------------------

    @torch.no_grad()
    def _generate(self, draft: str) -> str:
        device = torch.device(self.device)
        # Encode prefix: <bos> draft <sep>
        ids = [BOS_ID]
        ids.extend(self._tokenizer.token_to_id.get(t, 1)
                   for t in basic_tokenize(draft))
        ids.append(SEP_ID)
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        generated: list[int] = []
        max_len = self._model.config.max_seq_len
        for _ in range(self.max_new_tokens):
            if input_ids.size(1) >= max_len:
                break
            out = self._model(input_ids)
            logits = out["logits"][:, -1, :] / max(self.temperature, 1e-6)
            if self.top_k > 0:
                v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            tok_id = int(next_tok.item())
            if tok_id == EOS_ID or tok_id == PAD_ID:
                break
            generated.append(tok_id)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return self._tokenizer.decode(generated, strip_specials=True)
