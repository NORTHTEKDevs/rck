"""Online, continual training utilities.

There are no epochs, no batches, no train/test split. You stream symbols
through an RCKAgent and it learns. This module just bundles a few
convenience entry points for the CLI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from rck.agent import RCKAgent, StepTrace


def stream_text(path: str | Path, *, lowercase: bool = False) -> Iterable[str]:
    """Yield characters from a text file. Newlines included as symbols."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    if lowercase:
        text = text.lower()
    yield from text


def train_on_text(
    agent: RCKAgent,
    text: Iterable[str],
    max_chars: int | None = None,
    log_every: int = 500,
    on_log=None,
) -> list[StepTrace]:
    """Stream characters through the agent. Returns the trace history."""
    history: list[StepTrace] = []
    buf: list[str] = []
    for i, c in enumerate(text):
        if max_chars is not None and i >= max_chars:
            break
        buf.append(c)
    # Pre-materialising lets us pass teacher_next.
    traces = agent.observe(buf, learn=True)
    if on_log is not None:
        for i, t in enumerate(traces):
            if (i + 1) % log_every == 0:
                on_log(i + 1, t)
    history.extend(traces)
    return history


def recall_score(agent: RCKAgent, text: str, n_eval: int = 200) -> float:
    """Fraction of next-char predictions that match ground truth.

    No learning during eval. Resets temporal state first so it's a clean run.
    """
    agent.reset_temporal()
    chars = list(text[: n_eval + 1])
    correct = 0
    total = 0
    for i in range(len(chars) - 1):
        tr = agent.step(chars[i], learn=False, teacher_next=None)
        if tr.emitted_symbol == chars[i + 1]:
            correct += 1
        total += 1
    return correct / max(total, 1)
