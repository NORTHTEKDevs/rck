"""Causal chain reasoning.

Specialised chain walker for `causes` / `causedby` chains. Given a
starting cause, walk the causal graph to find all downstream effects.
Given a starting effect, walk backwards to find root causes.

This is a domain-specific instantiation of chain_walker with two
properties baked in:
  * the relation is fixed to `causes` (forward) or `causedby` (reverse)
  * the walker is BREADTH-FIRST rather than confidence-prioritised
    so we enumerate ALL effects/causes within max_depth, not just
    the highest-confidence chain
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from rck.knowledge_base import ShardedKnowledgeBase


@dataclass
class CausalNode:
    entity: str
    depth: int
    score: float
    via: list[tuple[str, str]] = field(default_factory=list)  # (prev, current)


def downstream_effects(kb: ShardedKnowledgeBase, cause: str,
                        *, max_depth: int = 3,
                        beam_width: int = 5,
                        min_score: float = 0.10) -> list[CausalNode]:
    """All entities reachable from `cause` via `causes` chains."""
    return _walk(kb, cause, relation="causes",
                  unknown_role="O", neighbour_role_for_query="S",
                  max_depth=max_depth, beam_width=beam_width,
                  min_score=min_score)


def root_causes(kb: ShardedKnowledgeBase, effect: str,
                 *, max_depth: int = 3,
                 beam_width: int = 5,
                 min_score: float = 0.10) -> list[CausalNode]:
    """All entities that reach `effect` via `causes` chains, walked
    backwards."""
    return _walk(kb, effect, relation="causes",
                  unknown_role="S", neighbour_role_for_query="O",
                  max_depth=max_depth, beam_width=beam_width,
                  min_score=min_score)


def _walk(kb: ShardedKnowledgeBase, start: str, *, relation: str,
          unknown_role: str, neighbour_role_for_query: str,
          max_depth: int, beam_width: int,
          min_score: float) -> list[CausalNode]:
    """Internal BFS over the relation graph.

    unknown_role: which role of the new edge is unknown
      ("O" for forward, "S" for backward).
    neighbour_role_for_query: which role we BIND to current
      ("S" for forward, "O" for backward).
    """
    start = start.lower()
    visited: set[str] = {start}
    frontier: deque = deque([(start, 0, 1.0, [])])
    nodes: list[CausalNode] = []
    while frontier:
        node, depth, score, via = frontier.popleft()
        if depth >= max_depth:
            continue
        known = {neighbour_role_for_query: node, "R": relation}
        candidates = kb.query(known, unknown_role, top_k=beam_width)
        for sym, edge_score in candidates:
            if float(edge_score) < min_score:
                continue
            nxt = str(sym).lower()
            if nxt in visited or nxt == start:
                continue
            visited.add(nxt)
            new_score = score * float(edge_score)
            new_via = via + [(node, nxt)]
            nodes.append(CausalNode(
                entity=nxt, depth=depth + 1,
                score=new_score, via=new_via,
            ))
            frontier.append((nxt, depth + 1, new_score, new_via))
    nodes.sort(key=lambda n: (n.depth, -n.score))
    return nodes
