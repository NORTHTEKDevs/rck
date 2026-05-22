"""Word-level tokenizer for the polisher.

Deliberately simple: split on whitespace + punctuation, lowercase,
build a vocab from the training corpus. No BPE -- at our scale
(50k-100k vocab) word-level is plenty and the architecture stays
inspectable.

Special tokens:
    <pad>  index 0  -- padding
    <unk>  index 1  -- unknown (rare in our domain)
    <bos>  index 2  -- beginning of sequence
    <eos>  index 3  -- end of sequence
    <sep>  index 4  -- separates draft from target during training
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# Match words (alphanumeric runs, with apostrophes for English) OR single
# punctuation. Same regex shape as rck/tokenizer.py for consistency.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:'[A-Za-z]+)?|[^\sA-Za-z0-9_]")

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID, SEP_ID = range(5)


def basic_tokenize(text: str) -> list[str]:
    """Lowercase + regex tokenization. Returns list of token strings."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class PolisherTokenizer:
    """Vocab-backed word tokenizer with reserved special tokens."""

    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id_to_token:
            self.id_to_token = list(SPECIAL_TOKENS)
            self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    # ---- build from a corpus ----------------------------------------------

    @classmethod
    def from_corpus(cls, texts: Iterable[str], *,
                    max_vocab: int = 50_000,
                    min_count: int = 2) -> "PolisherTokenizer":
        """Build a vocab by counting tokens across `texts`."""
        tok = cls()
        counts: Counter[str] = Counter()
        for txt in texts:
            counts.update(basic_tokenize(txt))
        # Keep tokens above min_count, capped at max_vocab.
        ordered = [t for t, c in counts.most_common() if c >= min_count]
        ordered = ordered[:max(0, max_vocab - len(SPECIAL_TOKENS))]
        for t in ordered:
            tok.token_to_id[t] = len(tok.id_to_token)
            tok.id_to_token.append(t)
        return tok

    # ---- encode / decode -------------------------------------------------

    def encode(self, text: str, *, add_bos: bool = False,
               add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(BOS_ID)
        for t in basic_tokenize(text):
            ids.append(self.token_to_id.get(t, UNK_ID))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def encode_pair(self, draft: str, target: str) -> tuple[list[int], int]:
        """Encode `<bos> draft <sep> target <eos>` and return the index
        of the separator (so the training loop can mask the draft side
        out of the loss).
        """
        ids = [BOS_ID]
        ids.extend(self.token_to_id.get(t, UNK_ID) for t in basic_tokenize(draft))
        sep_idx = len(ids)
        ids.append(SEP_ID)
        ids.extend(self.token_to_id.get(t, UNK_ID) for t in basic_tokenize(target))
        ids.append(EOS_ID)
        return ids, sep_idx

    def decode(self, ids: Iterable[int], *, strip_specials: bool = True) -> str:
        out: list[str] = []
        for i in ids:
            if i < 0 or i >= len(self.id_to_token):
                continue
            tok = self.id_to_token[i]
            if strip_specials and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        # Reattach punctuation to preceding word.
        text = ""
        for t in out:
            if not text:
                text = t
            elif t in ".,;:!?)]}":
                text += t
            elif t in "([{":
                text += " " + t
            else:
                text += " " + t
        return text.strip()

    # ---- persistence ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "id_to_token": self.id_to_token,
        }, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "PolisherTokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls(token_to_id={}, id_to_token=[])
        tok.id_to_token = list(data["id_to_token"])
        tok.token_to_id = {t: i for i, t in enumerate(tok.id_to_token)}
        return tok
