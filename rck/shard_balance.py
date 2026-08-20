"""Per-shard load balancing report.

The sharded KB routes each (S, R) pair via blake2b hash; with
imbalanced fact counts a few shards can drift toward the capacity
cliff while others stay nearly empty. This module produces a
report:

  * fill per shard
  * overloaded shards (above target_fill cap)
  * sparse shards (< target_fill * sparse_ratio)
  * suggested action (reshard / prune)

Output integrates with `shard_sizing.recommend_shards` so the user
can compute a proposed n_shards for the current load.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rck.knowledge_base import ShardedKnowledgeBase
from rck.shard_sizing import TARGET_MAX_FILL_BY_DIM, recommend_shards
from rck.dict_knowledge_base import DictKnowledgeBase

if TYPE_CHECKING:
    pass


@dataclass
class ShardBalanceReport:
    n_shards: int
    dim: int
    fills: list[int] = field(default_factory=list)
    target_fill: int = 0
    overloaded: list[int] = field(default_factory=list)
    sparse: list[int] = field(default_factory=list)
    suggested_n_shards: int = 0
    suggested_action: str = ""

    @property
    def min_fill(self) -> int:
        return min(self.fills) if self.fills else 0

    @property
    def max_fill(self) -> int:
        return max(self.fills) if self.fills else 0

    @property
    def avg_fill(self) -> float:
        return (sum(self.fills) / max(1, len(self.fills)))

    def verbalize(self) -> str:
        lines = [f"Shard balance ({self.n_shards} shards, D={self.dim}):"]
        lines.append(f"  fill min/avg/max: {self.min_fill}/{self.avg_fill:.1f}/{self.max_fill}")
        lines.append(f"  target_fill: {self.target_fill}")
        lines.append(f"  overloaded shards: {len(self.overloaded)}"
                      + (f"  indices: {self.overloaded[:5]}"
                         + (" ..." if len(self.overloaded) > 5 else "")
                         if self.overloaded else ""))
        lines.append(f"  sparse shards: {len(self.sparse)}")
        if self.suggested_action:
            lines.append(f"  suggested: {self.suggested_action}")
        return "\n".join(lines)


def report(kb: "ShardedKnowledgeBase | DictKnowledgeBase", *,
            sparse_ratio: float = 0.1) -> ShardBalanceReport:
    """Build a ShardBalanceReport from the current KB.

    [Phase 2] The dict backend (DictKnowledgeBase) has exactly one
    pseudo-shard and an exact index -- there is no capacity cliff to
    partition around and no reshard that could relieve anything, so it
    is reported honestly: no overload, no sparse flag, no suggestion,
    regardless of how many facts that one shard holds.
    """
    fills = kb.shard_sizes()
    n_shards = kb.n_shards
    dim = kb.dim
    if isinstance(kb, DictKnowledgeBase):
        return ShardBalanceReport(
            n_shards=n_shards, dim=dim, fills=list(fills),
            target_fill=0, overloaded=[], sparse=[],
            suggested_n_shards=n_shards, suggested_action="",
        )
    target_fill = TARGET_MAX_FILL_BY_DIM.get(dim, max(20, dim // 50))
    overloaded = [i for i, f in enumerate(fills) if f > target_fill]
    sparse_threshold = max(1, int(target_fill * sparse_ratio))
    sparse_shards = [i for i, f in enumerate(fills) if f < sparse_threshold]

    suggested_n_shards = n_shards
    suggested_action = ""
    if overloaded:
        rec = recommend_shards(kb.size(), dim=dim)
        suggested_n_shards = rec.n_shards
        if suggested_n_shards > n_shards:
            suggested_action = (
                f"reshard to {suggested_n_shards} shards "
                f"({len(overloaded)}/{n_shards} above capacity cliff)"
            )
    elif sparse_shards and len(sparse_shards) >= n_shards * 0.75:
        # Most shards are nearly empty.
        suggested_action = f"prune or reduce n_shards"

    return ShardBalanceReport(
        n_shards=n_shards, dim=dim,
        fills=list(fills),
        target_fill=target_fill,
        overloaded=overloaded,
        sparse=sparse_shards,
        suggested_n_shards=suggested_n_shards,
        suggested_action=suggested_action,
    )
