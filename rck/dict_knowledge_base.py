"""Exact-index knowledge base -- the second backend.

`ShardedKnowledgeBase` proves the reasoning layer works over an HRR
substrate. This module proves the *reasoning layer itself* does not
depend on that substrate: `DictKnowledgeBase` implements the exact same
surface (`store`, `forget`, `query`, `query_union`, `answer`, `size`,
`shard_sizes`, `all_facts`, `relation_index`, `reshard`, plus `dim`,
`n_shards`, `seed`, `wal`, `codebook`, `_fact_count`) over a plain
Python exact index -- no HRR, no numpy, no crosstalk.

Query semantics: every match scores exactly 1.0 (it either is or is
not in the index); a miss returns `[]`. Multi-valued results are
returned in INSERTION order (the same tie-break `all_facts()` uses),
not any similarity ranking -- there is no similarity here to rank by.

The pseudo-shard (`_DictShard`) exposes `.facts()` / `._facts` /
`.merge()`, the exact surface `RelationalMemory` shards expose, so the
Phase 1 ALLOWED_EXCEPTIONS modules (`dreaming.py::compress_duplicates`,
`curiosity.py::detect_global_gaps`, `research.py::_related_entities`,
`subject_summary.py::summarize_subject`) keep working unmodified
against `kb._shards` -- see docs/plans/2026-08-19-dict-backend.md
section C.

[R2] Index invariant: `dreaming.compress_duplicates` reassigns
`shard._facts = keep` directly, bypassing `store()`/`forget()`. To
guarantee `query()` never serves a stale index after that kind of
external reassignment, `_facts` is a property: any assignment to it
rebuilds the derived query index synchronously, on the spot. (The
alternative the plan allows -- deriving the index fresh on every
`query()` call -- would cost O(n_facts) per query at KB sizes Task 5
measures in the tens of thousands; rebuilding only on the rare
reassignment path is the same correctness guarantee at O(1) amortized
query cost.)

`codebook` is `None` on this backend. Nothing in the reasoning layer
reads `kb.codebook` for the `knowledge` or `beliefs` KBs -- every hit
on `.codebook` outside this module belongs either to `ShardedKnowledgeBase`
itself or to the HRR-only generative subsystem's own `self.lm.codebook`,
never to a `ConsciousAgent.knowledge`/`.beliefs` codebook read by a
reasoning-layer module. See docs/plans/2026-08-19-dict-backend.md,
"Report explicitly".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable

from rck.wal import WriteAheadLog


class _DictShard:
    """Single pseudo-shard behind `DictKnowledgeBase`.

    Maintains an exact role-value inverted index (`(role, value) ->
    [fact indices]`) so `query()` is O(min matching set) rather than
    O(n_facts), for ANY combination of known roles and ANY unknown
    role -- this is what makes the backend generic enough to also
    serve the belief KB's 4-tuple (B, S, R, O) facts without special
    casing, matching `RelationalMemory`'s own role-name-agnostic
    design.
    """

    def __init__(self) -> None:
        self._facts_list: list[dict[str, Hashable]] = []
        # Sets, not lists: query()'s intersection step needs O(1)
        # membership on whichever bucket turns out larger. A common
        # relation (e.g. "isa") can appear in a large fraction of the
        # KB, so its bucket is large; storing lists made every query
        # against it re-materialize a fresh set from that large list
        # (measured: baseline_study.py's dict-backend query_median grew
        # from competitive to 3x HRR's between 10k and 100k facts before
        # this fix -- see docs/plans/2026-08-19-dict-backend.md Task 5).
        # Sets don't preserve insertion order; the final `sorted(common)`
        # in query() restores it by fact index, which IS insertion order.
        self._by_role_value: dict[tuple[str, Hashable], set[int]] = {}

    # `_facts` is a property (not a plain list attribute) so an
    # external direct reassignment -- `shard._facts = keep`, exactly
    # what dreaming.compress_duplicates does -- rebuilds the index
    # synchronously instead of leaving it stale. See module docstring.
    @property
    def _facts(self) -> list[dict[str, Hashable]]:
        return self._facts_list

    @_facts.setter
    def _facts(self, value: list[dict[str, Hashable]]) -> None:
        self._facts_list = list(value)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._by_role_value = {}
        for i, f in enumerate(self._facts_list):
            self._index_add(i, f)

    def _index_add(self, i: int, fact: dict[str, Hashable]) -> None:
        for role, value in fact.items():
            self._by_role_value.setdefault((role, value), set()).add(i)

    # ---- core ops -----------------------------------------------------

    def store(self, fact: dict[str, Hashable]) -> None:
        i = len(self._facts_list)
        f = dict(fact)
        self._facts_list.append(f)
        self._index_add(i, f)

    def forget(self, fact: dict[str, Hashable]) -> None:
        """Remove the FIRST exact match, mirroring RelationalMemory.forget."""
        for i, f in enumerate(self._facts_list):
            if f == fact:
                del self._facts_list[i]
                self._rebuild_index()  # indices are positional; must rebuild
                return

    def query(self, known: dict[str, Hashable], unknown_role: str,
              top_k: int = 3) -> list[tuple[Hashable, float]]:
        if unknown_role in known:
            raise ValueError("unknown_role cannot also be in known")
        if not known:
            candidate_idxs: Iterable[int] = range(len(self._facts_list))
        else:
            sets = [self._by_role_value.get((role, value), set())
                    for role, value in known.items()]
            if any(not s for s in sets):
                candidate_idxs = []
            else:
                sets.sort(key=len)
                # Already sets -- `&=` lets CPython iterate whichever
                # operand is smaller internally, so this never re-scans
                # a large bucket just because it appears on the right.
                common = set(sets[0])
                for s in sets[1:]:
                    common &= s
                    if not common:
                        break
                candidate_idxs = sorted(common)
        seen: set = set()
        out: list[tuple[Hashable, float]] = []
        for i in candidate_idxs:
            f = self._facts_list[i]
            if unknown_role not in f:
                continue
            v = f[unknown_role]
            if v in seen:
                continue
            seen.add(v)
            out.append((v, 1.0))
            if len(out) >= top_k:
                break
        return out

    # ---- introspection --------------------------------------------------

    def size(self) -> int:
        return len(self._facts_list)

    def facts(self) -> list[dict[str, Hashable]]:
        return list(self._facts_list)

    # ---- merge ------------------------------------------------------------

    def merge(self, other: "_DictShard") -> list[dict[str, Hashable]]:
        """Dedup-union `other`'s facts into this shard.

        The exact-index equivalent of `RelationalMemory.merge`'s bundle
        sum: instead of adding tensors, union the fact sets, skipping
        exact duplicates (HRR's bundle sum has no such dedup -- summing
        the same fact twice just over-weights it in the bundle; there is
        no bundle here to over-weight, so dedup is the correct exact-index
        analogue). [R2 decision B]

        Raises TypeError on a mixed-backend merge (summing an HRR
        tensor into an exact index, or vice versa, is not meaningful).

        Returns the list of facts actually newly added (NOT `other`'s
        full fact list) -- callers that WAL-log a merge (federated_merge.py)
        need this to log only what really changed; logging every one of
        `other`'s facts unconditionally would replay duplicates that this
        dedup-union never added.
        """
        if not isinstance(other, _DictShard):
            raise TypeError(
                f"cannot merge {type(other).__name__} into DictKnowledgeBase's "
                "pseudo-shard: mixed-backend merge is not meaningful "
                "(summing an HRR bundle into an exact index has no defined "
                "semantics)"
            )
        existing = {tuple(sorted(f.items())) for f in self._facts_list}
        added: list[dict[str, Hashable]] = []
        for f in other._facts_list:
            key = tuple(sorted(f.items()))
            if key in existing:
                continue
            existing.add(key)
            self.store(f)
            added.append(dict(f))
        return added


@dataclass
class DictKnowledgeBase:
    """Exact-index knowledge base: the surface `ShardedKnowledgeBase`
    exposes, backed by a plain Python index instead of HRR shards.

    `n_shards` is always 1 (one pseudo-shard) -- there is nothing to
    shard; an exact index has no capacity cliff to partition around.
    `dim` and `seed` are accepted for constructor-shape parity with
    `ShardedKnowledgeBase` (so `ConsciousAgent.__post_init__` can build
    either backend the same way) but do not affect indexing.
    """

    dim: int = 4096
    seed: int = 0
    wal: WriteAheadLog | None = None

    codebook: None = field(default=None, init=False)
    n_shards: int = field(default=1, init=False)
    _shards: list[_DictShard] = field(default_factory=list, init=False)
    _fact_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._shards = [_DictShard()]

    # ---- core ops -----------------------------------------------------

    def store(self, fact: dict[str, Hashable], *, _log: bool = True) -> None:
        self._shards[0].store(fact)
        self._fact_count += 1
        if self.wal is not None and _log:
            self.wal.append("store", dict(fact))

    def store_many(self, facts: Iterable[dict[str, Hashable]]) -> int:
        count = 0
        for f in facts:
            self.store(f)
            count += 1
        return count

    def forget(self, fact: dict[str, Hashable], *, _log: bool = True) -> None:
        self._shards[0].forget(fact)
        self._fact_count = max(0, self._fact_count - 1)
        if self.wal is not None and _log:
            self.wal.append("forget", dict(fact))

    def query(self, known: dict[str, Hashable], unknown_role: str,
              top_k: int = 3,
              shard_subset: Iterable[int] | None = None,
              cleanup: str = "local",
              ) -> list[tuple[Hashable, float]]:
        """`shard_subset` / `cleanup` are accepted for interface parity
        with `ShardedKnowledgeBase.query` and ignored beyond validating
        `shard_subset` (there is exactly one shard, index 0; `cleanup`
        has no effect because an exact index has no crosstalk to filter)."""
        if cleanup not in ("local", "global"):
            raise ValueError(
                f"cleanup must be 'local' or 'global', got {cleanup!r}")
        if shard_subset is not None:
            subset = {int(i) for i in shard_subset}
            bad = sorted(i for i in subset if i != 0)
            if bad:
                raise ValueError(
                    f"shard_subset indices out of range for "
                    f"n_shards=1: {bad}")
            if 0 not in subset:
                return []
        return self._shards[0].query(known, unknown_role, top_k=top_k)

    def query_union(self, known: dict[str, Hashable], unknown_role: str,
                    per_shard_top_k: int = 5,
                    min_score: float | None = None,
                    ) -> list[tuple[Hashable, float, int]]:
        results = self._shards[0].query(known, unknown_role, top_k=per_shard_top_k)
        return [(sym, score, 0) for sym, score in results]

    def answer(self, known: dict[str, Hashable],
               unknown_role: str) -> tuple[Hashable | None, float]:
        results = self.query(known, unknown_role, top_k=1)
        return results[0] if results else (None, 0.0)

    # ---- introspection --------------------------------------------------

    def size(self) -> int:
        return self._fact_count

    def all_facts(self) -> list[dict[str, Hashable]]:
        return self._shards[0].facts()

    def shard_sizes(self) -> list[int]:
        return [self._shards[0].size()]

    def relation_index(self):
        # RelationIndex.build() only touches kb._shards / shard.facts(),
        # so it works unchanged over the single pseudo-shard.
        from rck.knowledge_base import RelationIndex
        return RelationIndex.build(self)

    def reshard(self, n_shards: int | None = None) -> dict:
        """No-op: an exact index has no shard count to change and no
        capacity cliff to reshard away from. Same return shape as
        `ShardedKnowledgeBase.reshard`'s no-change branch, so callers
        like `session.load_session` (`agent.beliefs.reshard(...)`,
        return value discarded) don't need to special-case the backend."""
        return {"n_shards": self.n_shards, "facts": self._fact_count,
                "resharded": False}

    def utilization(self) -> dict:
        sizes = self.shard_sizes()
        return {
            "n_shards": self.n_shards,
            "total_facts": sum(sizes),
            "max_shard": max(sizes) if sizes else 0,
            "min_shard": min(sizes) if sizes else 0,
            "avg_shard": sum(sizes) / max(1, len(sizes)),
            "histogram_top4": sorted(sizes, reverse=True)[:4],
        }
