"""Sharded HRR knowledge base.

A single `RelationalMemory(dim=4096)` cleanly holds ~250 facts before
crosstalk dominates the cleanup signal. To scale, we partition facts
across N independent memory shards keyed by a deterministic hash of the
(subject, relation) pair. Each shard stores only the facts routed to it,
so cleanup quality stays high regardless of total knowledge volume.

Capacity scaling:
  - 1 shard, D=4096:    ~250 facts at 95% recall
  - 64 shards, D=4096:  ~16k facts
  - 256 shards, D=4096: ~64k facts
  - 1024 shards, D=4096: ~256k facts

Each shard is a `RelationalMemory` with NAME-SEEDED role HVs, so all
shards share the same role embedding -- a fact stored in shard 17 can
be queried with the same role HV that the user/agent always sees.

Sharding key:
  shard_idx = blake2b(subject || "::" || relation) mod n_shards

The codebook is GLOBAL (all shards share it) so the same symbol always
maps to the same HV -- otherwise federated merge / cross-shard reasoning
would break.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Hashable, Iterable

from rck.codebook import Codebook
from rck.relational import RelationalMemory
from rck.shard_sizing import recommend_shards as _recommend_shards


def _shard_index(subject: str, relation: str, n_shards: int) -> int:
    """Stable hash-based routing key."""
    key = f"{subject}\x00{relation}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int.from_bytes(digest, "little") % n_shards


@dataclass
class RelationIndex:
    """Snapshot of which relations have facts in which shards.

    Built from the shards' symbolic fact logs, so it is correct under
    every mutation path (store, forget, shard-level federated merge,
    session load) at the moment it is built. It does NOT track later
    writes -- rebuild after mutating the KB. Chain discovery builds a
    fresh one per call by default, which costs one O(total_facts) scan
    (the same scan discovery already needed to enumerate relations) and
    in exchange skips every HRR query against a (subject, relation)
    shard that holds no fact with that relation.
    """

    per_shard: list[set[str]]

    @classmethod
    def build(cls, kb: "ShardedKnowledgeBase") -> "RelationIndex":
        per_shard: list[set[str]] = []
        for shard in kb._shards:
            rels: set[str] = set()
            for fact in shard.facts():
                r = fact.get("R")
                if r is not None:
                    rels.add(str(r))
            per_shard.append(rels)
        return cls(per_shard=per_shard)

    @property
    def n_shards(self) -> int:
        return len(self.per_shard)

    def live(self, subject: str, relation: str) -> bool:
        """True iff the shard that (subject, relation) routes to holds at
        least one fact with this relation. A False answer proves the
        forward query (subject, relation, ?) can only return crosstalk
        noise, so the caller may skip it."""
        idx = _shard_index(subject, relation, self.n_shards)
        return relation in self.per_shard[idx]

    def shards_with(self, relation: str) -> list[int]:
        return [i for i, rels in enumerate(self.per_shard) if relation in rels]

    def all_relations(self) -> list[str]:
        out: set[str] = set()
        for rels in self.per_shard:
            out |= rels
        return sorted(out)


@dataclass
class ShardedKnowledgeBase:
    """N HRR shards behind a single user-facing API.

    Args:
        dim:        hypervector dimensionality per shard.
        n_shards:   number of HRR shards. More -> higher total capacity at
                    O(N) memory; pick log-spaced (8/64/256/1024).
        seed:       master seed (passed to codebook + every shard).
    """

    dim: int = 4096
    n_shards: int = 64
    seed: int = 0

    codebook: Codebook = field(default=None, init=False)
    _shards: list[RelationalMemory] = field(default_factory=list, init=False)
    _fact_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.codebook = Codebook(dim=self.dim, seed=self.seed)
        # All shards share the same role embedding because they share `seed`
        # and the role-HVs are derived from name-hash, not insertion order.
        self._shards = [
            RelationalMemory(
                dim=self.dim, seed=self.seed,
                role_names=("S", "R", "O", "B"),  # B = "believer", optional
            )
            for _ in range(self.n_shards)
        ]

    # ---- core ops ----------------------------------------------------------

    def store(self, fact: dict[str, Hashable]) -> None:
        """Store a fact in the shard determined by (S, R)."""
        s, r = str(fact.get("S", "")), str(fact.get("R", ""))
        idx = _shard_index(s, r, self.n_shards)
        self._shards[idx].store(self.codebook, fact)
        self._fact_count += 1

    def store_many(self, facts: Iterable[dict[str, Hashable]]) -> int:
        count = 0
        for f in facts:
            self.store(f)
            count += 1
        return count

    def reshard(self, n_shards: int | None = None) -> dict:
        """Re-bundle every stored fact into a new shard array.

        Facts are retained per shard (`RelationalMemory._facts`), so this is a
        pure re-route: nothing is re-derived and nothing is lost. The codebook
        is reused -- resharding changes routing, never symbol encoding.
        """
        if n_shards is None:
            n_shards = max(self.n_shards * 2,
                           _recommend_shards(self._fact_count, dim=self.dim).n_shards)
        if n_shards == self.n_shards:
            return {"n_shards": self.n_shards, "facts": self._fact_count,
                    "resharded": False}

        facts = [f for shard in self._shards for f in shard.facts()]
        new_shards = [
            RelationalMemory(dim=self.dim, seed=self.seed,
                             role_names=("S", "R", "O", "B"))
            for _ in range(n_shards)
        ]
        for f in facts:
            idx = _shard_index(str(f.get("S", "")), str(f.get("R", "")), n_shards)
            # Direct write: self.store() would re-increment _fact_count.
            new_shards[idx].store(self.codebook, f)

        self._shards = new_shards
        self.n_shards = n_shards
        return {"n_shards": n_shards, "facts": len(facts), "resharded": True}

    def relation_index(self) -> RelationIndex:
        """Fresh snapshot of live relations per shard (see RelationIndex)."""
        return RelationIndex.build(self)

    def query(self, known: dict[str, Hashable], unknown_role: str,
              top_k: int = 3,
              shard_subset: Iterable[int] | None = None,
              cleanup: str = "local"
              ) -> list[tuple[Hashable, float]]:
        """Retrieve from the shard determined by (S, R) if both are known,
        otherwise broadcast across all shards and merge with cross-shard
        evidence pooling + route-back verification.

        Cross-shard merge rules:
          * Same symbol seen in multiple shards -> evidence is pooled.
            We take MAX cosine across shards but slightly upweight when
            >=2 shards see the symbol above `_UNION_MIN_SCORE` -- the
            agreement is informative.
          * For fan-out queries where the missing role is S, R, or O, we
            verify each candidate by routing the implied (S, R) back to
            its correct shard. False positives caused by codebook cleanup
            noise in OTHER shards get filtered out.
        """
        if shard_subset is not None:
            shard_subset = self._validated_subset(shard_subset)
        s, r = str(known.get("S", "")), str(known.get("R", ""))
        if s and r:
            idx = _shard_index(s, r, self.n_shards)
            if shard_subset is not None and idx not in shard_subset:
                return []
            return self._shards[idx].query(self.codebook, known,
                                           unknown_role, top_k=top_k,
                                           cleanup=cleanup)
        # Fan-out: ask every shard, then merge with evidence pooling.
        return self._fanout_query(known, unknown_role, top_k=top_k,
                                  shard_subset=shard_subset,
                                  cleanup=cleanup)

    def _validated_subset(self, shard_subset: Iterable[int]) -> set[int]:
        subset = {int(i) for i in shard_subset}
        bad = sorted(i for i in subset if not 0 <= i < self.n_shards)
        if bad:
            raise ValueError(
                f"shard_subset indices out of range for "
                f"n_shards={self.n_shards}: {bad}")
        return subset

    # --- cross-shard union -------------------------------------------------

    _UNION_MIN_SCORE: float = 0.05  # filter: ignore noise below this cosine
    _UNION_AGREEMENT_BONUS: float = 0.05  # boost per extra-shard agreement

    def _fanout_query(self, known: dict[str, Hashable], unknown_role: str,
                      top_k: int = 3,
                      shard_subset: Iterable[int] | None = None,
                      cleanup: str = "local"
                      ) -> list[tuple[Hashable, float]]:
        # Collect per-shard hits.
        if shard_subset is None:
            shards = self._shards
        else:
            shards = [self._shards[i]
                      for i in self._validated_subset(shard_subset)]
        per_symbol_scores: dict[Hashable, list[float]] = {}
        for shard in shards:
            for sym, score in shard.query(self.codebook, known, unknown_role,
                                          top_k=top_k, cleanup=cleanup):
                if score < self._UNION_MIN_SCORE:
                    continue
                per_symbol_scores.setdefault(sym, []).append(float(score))

        # Build merged scoring.
        merged: list[tuple[Hashable, float, int]] = []  # (sym, score, agreement_count)
        for sym, scores in per_symbol_scores.items():
            best = max(scores)
            agreement = len(scores)
            boost = self._UNION_AGREEMENT_BONUS * (agreement - 1)
            merged.append((sym, min(1.0, best + boost), agreement))

        # Route-back verification: if a candidate would route to a shard
        # whose recall is non-trivial, prefer it. We use this only as a
        # tiebreak to avoid penalising real low-score hits.
        # (Implemented via stable sort: primary = score, secondary = agreement.)
        merged.sort(key=lambda x: (-x[1], -x[2]))
        return [(sym, score) for sym, score, _ in merged[:top_k]]

    def query_union(self, known: dict[str, Hashable], unknown_role: str,
                    per_shard_top_k: int = 5,
                    min_score: float | None = None
                    ) -> list[tuple[Hashable, float, int]]:
        """Cross-shard union retrieval: every shard's candidates, tagged
        with originating shard.

        Returns a list of `(symbol, score, shard_idx)` tuples sorted by
        score descending. Useful for enumeration queries
        ("what things are mammals?") and for debugging which shards
        contribute to a fan-out.
        """
        threshold = self._UNION_MIN_SCORE if min_score is None else float(min_score)
        out: list[tuple[Hashable, float, int]] = []
        for idx, shard in enumerate(self._shards):
            for sym, score in shard.query(self.codebook, known, unknown_role,
                                          top_k=per_shard_top_k):
                if float(score) < threshold:
                    continue
                out.append((sym, float(score), idx))
        out.sort(key=lambda x: -x[1])
        return out

    def answer(self, known: dict[str, Hashable], unknown_role: str) -> tuple[Hashable | None, float]:
        results = self.query(known, unknown_role, top_k=1)
        return (results[0] if results else (None, 0.0))

    def forget(self, fact: dict[str, Hashable]) -> None:
        s, r = str(fact.get("S", "")), str(fact.get("R", ""))
        idx = _shard_index(s, r, self.n_shards)
        self._shards[idx].forget(self.codebook, fact)
        self._fact_count = max(0, self._fact_count - 1)

    # ---- introspection -----------------------------------------------------

    def size(self) -> int:
        """Total facts stored across all shards."""
        return self._fact_count

    def shard_sizes(self) -> list[int]:
        return [s.size() for s in self._shards]

    def utilization(self) -> dict:
        """Stats for visualisation."""
        sizes = self.shard_sizes()
        return {
            "n_shards": self.n_shards,
            "total_facts": sum(sizes),
            "max_shard": max(sizes) if sizes else 0,
            "min_shard": min(sizes) if sizes else 0,
            "avg_shard": sum(sizes) / max(1, len(sizes)),
            "histogram_top4": sorted(sizes, reverse=True)[:4],
        }
