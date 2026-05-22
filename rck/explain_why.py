"""Provenance graph traversal: explain WHY a fact is in the KB.

Every stored fact carries a ProvenanceRecord. For user-provided
facts this is a leaf -- "the user said so." For derived facts
(source="induced" or "rule") the record carries a `derivation`
list: the chain of (s, r, o) tuples that produced this fact.

This module walks the derivation backwards recursively. A
fact derived from a 2-hop chain might itself involve facts that
were ALSO derived from earlier chains. The traversal terminates
at leaves (user-provided or unknown-source facts).

The output is an `ExplanationNode` tree that callers can render
as text or a graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rck.provenance import ProvenanceStore


@dataclass
class ExplanationNode:
    subject: str
    relation: str
    obj: str
    source: str = "unknown"
    confidence: float = 1.0
    children: list["ExplanationNode"] = field(default_factory=list)
    is_leaf: bool = False
    note: str = ""

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def total_facts(self) -> int:
        """Total facts in the derivation tree, including this node."""
        return 1 + sum(c.total_facts() for c in self.children)

    def verbalize(self, indent: int = 0) -> str:
        prefix = "  " * indent
        leaf_marker = " (leaf)" if self.is_leaf else ""
        lines = [f"{prefix}({self.subject}, {self.relation}, {self.obj})  "
                 f"source={self.source}{leaf_marker}"]
        for child in self.children:
            lines.append(child.verbalize(indent + 1))
        return "\n".join(lines)


def explain(provenance: ProvenanceStore,
            subject: str, relation: str, obj: str,
            *, max_depth: int = 5,
            _seen: Optional[set[tuple]] = None) -> ExplanationNode:
    """Build an ExplanationNode tree for `(subject, relation, obj)`.

    Recurses up to `max_depth` hops. Cycles are broken by tracking
    `_seen`. Leaves are reached when:
      * `source == "user"` (the user told us this).
      * No provenance record exists (`source = "unknown"`).
      * `derivation` is empty.
      * The recursion hit max_depth.
    """
    seen = _seen if _seen is not None else set()
    s, r, o = subject.lower(), relation.lower(), obj.lower()
    key = (s, r, o)
    if key in seen:
        return ExplanationNode(
            subject=s, relation=r, obj=o, source="cycle",
            is_leaf=True, note="already visited (cycle break)",
        )
    seen = seen | {key}

    rec = provenance.get(s, r, o)
    if rec is None:
        return ExplanationNode(
            subject=s, relation=r, obj=o, source="unknown",
            confidence=1.0, is_leaf=True,
            note="no provenance record",
        )
    node = ExplanationNode(
        subject=s, relation=r, obj=o,
        source=rec.source, confidence=rec.confidence,
    )
    if not rec.derivation or max_depth <= 0:
        node.is_leaf = True
        if rec.source == "user":
            node.note = "user-asserted"
        return node
    # Recurse into derivation children.
    for c_s, c_r, c_o in rec.derivation:
        child = explain(provenance, c_s, c_r, c_o,
                         max_depth=max_depth - 1, _seen=seen)
        node.children.append(child)
    return node


def explanation_summary(node: ExplanationNode) -> dict:
    """Flatten an ExplanationNode tree into a summary dict."""
    def _walk(n: ExplanationNode, depth: int) -> tuple[int, int, dict[str, int]]:
        sources: dict[str, int] = {n.source: 1}
        total = 1
        max_depth = depth
        for c in n.children:
            ct, cd, cs = _walk(c, depth + 1)
            total += ct
            max_depth = max(max_depth, cd)
            for src, k in cs.items():
                sources[src] = sources.get(src, 0) + k
        return total, max_depth, sources

    total, max_depth, sources = _walk(node, 0)
    return {
        "root": (node.subject, node.relation, node.obj),
        "root_source": node.source,
        "total_facts_in_tree": total,
        "max_depth": max_depth,
        "sources": sources,
    }
