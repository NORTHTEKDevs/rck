"""Chain discovery -- BFS over the HRR knowledge base.

`chain_walker` EXECUTES a chain when you already know the relation
sequence. This module DISCOVERS the relation sequence: given a start
entity and a goal predicate, find a chain of relations that connects
them through the KB.

Why this matters: it closes the loop on multi-hop reasoning. Real
questions don't come with chain specs; they come as natural language.
With chain discovery the agent can answer questions of the form
"reach X from Y" without a hand-written template.

The search is HRR-native: each frontier expansion is a KB query
(`kb.query`) returning top-K candidates with cosine scores. We BFS
in score order (highest-confidence frontier first), bounded by
`max_depth` and `beam_width`.

Goals are specified with a `Goal` predicate:
  * `Goal.symbol("paris")`              -- reach exactly this atom
  * `Goal.relation_value("isa", "continent")` -- reach any atom where
    (atom, isa, continent) holds
  * `Goal.custom(callable)`              -- any predicate
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from rck.knowledge_base import RelationIndex, ShardedKnowledgeBase


@dataclass
class Goal:
    """Predicate identifying when a BFS frontier node is the goal."""
    match: Callable[[str, ShardedKnowledgeBase], bool]
    description: str

    @classmethod
    def symbol(cls, target: str) -> "Goal":
        t = target.lower()
        return cls(match=lambda node, kb: node == t,
                   description=f"node == {target!r}")

    @classmethod
    def relation_value(cls, relation: str, value: str,
                       min_score: float = 0.10) -> "Goal":
        r, v = relation.lower(), value.lower()
        def _m(node: str, kb: ShardedKnowledgeBase) -> bool:
            ans, score = kb.answer({"S": node, "R": r}, "O")
            return ans is not None and str(ans) == v and score >= min_score
        return cls(match=_m, description=f"({{node}}, {relation}, {value})")

    @classmethod
    def relation_present(cls, relation: str,
                         min_score: float = 0.10) -> "Goal":
        r = relation.lower()
        def _m(node: str, kb: ShardedKnowledgeBase) -> bool:
            ans, score = kb.answer({"S": node, "R": r}, "O")
            return ans is not None and score >= min_score
        return cls(match=_m, description=f"({{node}}, {relation}, ?)")

    @classmethod
    def custom(cls, predicate: Callable[[str, ShardedKnowledgeBase], bool],
               description: str = "custom") -> "Goal":
        return cls(match=predicate, description=description)


@dataclass(order=True)
class _Frontier:
    """Heap entry. Negative score (min-heap pops the highest score)."""
    neg_score: float
    depth: int
    node: str = field(compare=False)
    path: list[tuple[str, str, str, float]] = field(compare=False)
    dirs: list[str] = field(default_factory=list, compare=False)


@dataclass
class DiscoveredChain:
    relations: list[str]
    directions: list[str]
    trace: list[tuple[str, str, str, float]]
    confidence: float
    goal_description: str

    def signature(self) -> tuple:
        return tuple(zip(self.relations, self.directions))


def discover_chains(kb: ShardedKnowledgeBase, start: str, goal: Goal,
                    *, max_depth: int = 4, beam_width: int = 4,
                    relations: Iterable[str] | None = None,
                    min_link_score: float = 0.10,
                    top_n: int = 3,
                    allow_reverse: bool = False,
                    skills_prior=None,
                    relation_index: RelationIndex | None = None,
                    use_relation_index: bool = True) -> list[DiscoveredChain]:
    """Search the KB for chains starting at `start` that satisfy `goal`.

    Args:
        relations: if given, restrict edge relations to this set. Default
            None means "try every relation in the KB's codebook" -- which
            is what chains do in practice.
        beam_width: top-K candidates explored per frontier expansion.
        top_n: number of best chains to return.
        skills_prior: optional SkillLibrary. When provided, the search
            REORDERS its per-relation expansion so that relations
            appearing in high-confidence skill patterns are tried first.
            This is a pure speedup: the same chains are findable; we
            just hit them earlier in the search.
        relation_index: optional prebuilt RelationIndex snapshot. Pass one
            when running many discoveries against an unchanged KB to skip
            the per-call rebuild. Must be rebuilt after KB writes.
        use_relation_index: when True (default), skip the HRR query for
            any (node, relation) whose routed shard holds no fact with
            that relation, and restrict reverse fan-outs to shards where
            the relation is live. Pruned queries could only ever return
            bundle crosstalk, so real chains are unaffected; edges that
            existed purely as cleanup noise are no longer followed. Set
            False to reproduce the pre-index (v15.1) search exactly.

    Returns up to `top_n` chains, sorted by total confidence.
    """
    start = start.lower()
    visited: set[tuple[str, int]] = set()
    # Min-heap by -score so popping gives highest-confidence frontier.
    heap: list[_Frontier] = [_Frontier(neg_score=-1.0, depth=0,
                                        node=start, path=[], dirs=[])]
    found: list[DiscoveredChain] = []
    seen_paths: set[tuple] = set()

    index: RelationIndex | None = None
    if use_relation_index:
        index = relation_index if relation_index is not None \
            else RelationIndex.build(kb)

    # The relations we'll try to expand on. If unconstrained, use the
    # KB's known relations (collected from prior queries).
    if relations is None:
        relations_list = index.all_relations() if index is not None \
            else _enumerate_relations(kb)
    else:
        relations_list = [r.lower() for r in relations]
    if not relations_list:
        return []

    # Skill-prior reordering: relations that appear in high-success
    # skill patterns get scored higher and tried first. The set of
    # relations explored is unchanged -- this is purely a heuristic
    # ordering for faster discovery.
    if skills_prior is not None:
        relation_priority = _relation_priority_from_skills(skills_prior)
        relations_list.sort(
            key=lambda r: (-relation_priority.get(r, 0.0), r),
        )

    while heap:
        cur = heapq.heappop(heap)
        node, depth, path, dirs = cur.node, cur.depth, cur.path, cur.dirs
        if depth > 0 and goal.match(node, kb):
            chain = DiscoveredChain(
                relations=[r for _, r, _, _ in path],
                directions=list(dirs),
                trace=list(path),
                confidence=_propagated(path),
                goal_description=goal.description,
            )
            sig = chain.signature()
            if sig not in seen_paths:
                seen_paths.add(sig)
                found.append(chain)
                if len(found) >= top_n:
                    break
            continue
        if depth >= max_depth:
            continue
        key = (node, depth)
        if key in visited:
            continue
        visited.add(key)

        for rel in relations_list:
            # Forward edges: (node, rel, ?)
            if index is not None and not index.live(node, rel):
                candidates = []
            else:
                candidates = kb.query({"S": node, "R": rel}, "O",
                                      top_k=beam_width)
            for sym, score in candidates:
                if score < min_link_score:
                    continue
                new_path = path + [(node, rel, str(sym), float(score))]
                if any(p[0] == str(sym) for p in path) or str(sym) == start:
                    continue
                propagated = _propagated(new_path)
                heapq.heappush(heap, _Frontier(
                    neg_score=-propagated, depth=depth + 1,
                    node=str(sym).lower(), path=new_path,
                    dirs=dirs + ["forward"],
                ))
            # Reverse edges: (?, rel, node) -- only when explicitly enabled.
            if allow_reverse:
                rev_subset = index.shards_with(rel) if index is not None \
                    else None
                if rev_subset == []:
                    continue
                rev_candidates = kb.query({"R": rel, "O": node}, "S",
                                          top_k=beam_width,
                                          shard_subset=rev_subset)
                for sym, score in rev_candidates:
                    if score < min_link_score:
                        continue
                    new_path = path + [(str(sym), rel, node, float(score))]
                    if any(p[0] == str(sym) for p in path) or str(sym) == start:
                        continue
                    propagated = _propagated(new_path)
                    heapq.heappush(heap, _Frontier(
                        neg_score=-propagated, depth=depth + 1,
                        node=str(sym).lower(), path=new_path,
                        dirs=dirs + ["reverse"],
                    ))

    found.sort(key=lambda c: -c.confidence)
    return found[:top_n]


def _relation_priority_from_skills(skills_prior) -> dict[str, float]:
    """Build a per-relation utility score from a SkillLibrary.

    For each (role, relation) edge in each skill's pattern, contribute
    `skill.success_count * skill.confidence` to that relation. Higher
    aggregates -> tried first by the BFS expansion.
    """
    priority: dict[str, float] = {}
    for skill in skills_prior.skills.values():
        weight = skill.success_count * max(skill.confidence, 0.01)
        for _role, relation in skill.pattern:
            priority[relation] = priority.get(relation, 0.0) + float(weight)
    return priority


def _enumerate_relations(kb: ShardedKnowledgeBase) -> list[str]:
    """Read every symbol the codebook knows and filter for relation-like
    names. Without explicit metadata we use a heuristic: any symbol that
    appears as the R-slot in stored facts."""
    # The codebook holds atoms but doesn't separate roles from fillers.
    # We use the stored facts list per shard.
    relations: set[str] = set()
    for fact in kb.all_facts():
        r = fact.get("R")
        if r is not None:
            relations.add(str(r))
    return sorted(relations)


def _propagated(path: list[tuple[str, str, str, float]]) -> float:
    """Geometric mean of link scores (matches v13 default propagation)."""
    if not path:
        return 0.0
    import math
    log_sum = 0.0
    for _, _, _, s in path:
        log_sum += math.log(max(s, 1e-9))
    return math.exp(log_sum / len(path))
