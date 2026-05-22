"""Spatial reasoning -- coordinate-free directional + containment queries.

Two query types:
  1. Containment chains: "Is Paris in Europe?" -> walk locatedin chain.
  2. Cardinal direction: stored as facts (germany, north_of, italy).

For coordinate-based reasoning we'd need (lat, lon) per entity; that's a
v2.x problem. This module covers the qualitative spatial relations
that come up in natural conversation.
"""
from __future__ import annotations

from rck.knowledge_base import ShardedKnowledgeBase


def is_inside(kb: ShardedKnowledgeBase, container: str, contained: str,
              max_depth: int = 6) -> dict:
    """Walk the locatedin chain to see if `contained` is inside `container`.

    At each hop we look at top-K candidates (not just top-1) because the
    HRR cleanup can return cross-bound entities at high cosine when facts
    in the same shard share a slot value. The walk picks the best fresh
    candidate -- one that isn't already in the chain.
    """
    contained = contained.lower(); container = container.lower()
    seen = {contained}
    chain = [contained]
    cursor = contained
    for _ in range(max_depth):
        results = kb.query({"S": cursor, "R": "locatedin"}, "O", top_k=4)
        # Pick the highest-cosine candidate not already in the chain.
        nxt = None
        for sym, score in results:
            sym_s = str(sym)
            if score < 0.10:
                continue
            if sym_s in seen:
                continue
            nxt = sym_s; break
        if nxt is None:
            break
        if nxt == container:
            chain.append(nxt)
            return {"answer": True, "chain": chain, "depth": len(chain) - 1}
        seen.add(nxt); chain.append(nxt); cursor = nxt
    return {"answer": False, "chain": chain, "depth": len(chain) - 1}


def direction_between(kb: ShardedKnowledgeBase, a: str, b: str) -> dict:
    """Look up an explicit (a, <direction>_of, b) fact."""
    a = a.lower(); b = b.lower()
    for rel in ("north_of", "south_of", "east_of", "west_of"):
        ans, score = kb.answer({"S": a, "R": rel}, "O")
        if ans is not None and score > 0.10 and str(ans) == b:
            direction = rel.replace("_of", "")
            return {"answer": direction,
                    "verbal": f"{a} is {direction} of {b}.",
                    "confidence": float(score)}
    return {"answer": None, "verbal": f"I don't know the direction between {a} and {b}.",
            "confidence": 0.0}
