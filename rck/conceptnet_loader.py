"""ConceptNet 5.7 schema mapper + streaming loader.

ConceptNet is a free 36M-assertion common-sense knowledge graph. Its
English subset is ~3M assertions. This module maps ConceptNet's
relation URIs (/r/IsA, /r/HasA, /r/AtLocation, ...) into RCK's
canonical schema, then streams the data into a ShardedKnowledgeBase
with provenance.

ConceptNet assertion format (TSV/JSON, ~3M lines):
  uri | rel | start | end | weight | data
  /a/[/r/IsA/,/c/en/dog/,/c/en/animal/]
    /r/IsA  /c/en/dog  /c/en/animal  1.0  {...}

We parse just the rel + start + end + weight columns, filter to
English-on-both-ends, strip the `/c/en/` URI prefix, map the
relation to RCK's canonical name, and store via bulk_ingest.

For testing without the 700MB download we also provide a tiny
synthetic ConceptNet-shaped fixture.
"""
from __future__ import annotations

import csv
import re
import time
from collections.abc import Iterable
from pathlib import Path

from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore


# ConceptNet relation -> RCK canonical relation.
# Coverage is the top relations by frequency in ConceptNet 5.7 EN.
CONCEPTNET_RELATION_MAP: dict[str, str] = {
    "/r/IsA":             "isa",
    "/r/HasA":            "has",
    "/r/AtLocation":      "locatedin",
    "/r/Causes":          "causes",
    "/r/HasProperty":     "is",
    "/r/MadeOf":          "madeof",
    "/r/PartOf":          "partof",
    "/r/UsedFor":         "usedfor",
    "/r/CapableOf":       "can",
    "/r/Desires":         "desires",
    "/r/MotivatedByGoal": "motivated_by",
    "/r/HasSubevent":     "has_subevent",
    "/r/HasPrerequisite": "requires",
    "/r/CausesDesire":    "causes_desire",
    "/r/Synonym":         "synonym",
    "/r/Antonym":         "antonym",
    "/r/DerivedFrom":     "derived_from",
    "/r/HasContext":      "has_context",
    "/r/SimilarTo":       "similar_to",
    "/r/InstanceOf":      "isa",
    "/r/MemberOf":        "memberof",
    "/r/DefinedAs":       "definedas",
    "/r/Entails":         "entails",
    "/r/SymbolOf":        "symbolof",
    "/r/RelatedTo":       "related_to",
    "/r/FormOf":          "form_of",
    "/r/EtymologicallyRelatedTo": "etymologically_related_to",
    "/r/Desire":          "desires",
    "/r/HasFirstSubevent":   "first_subevent",
    "/r/HasLastSubevent":    "last_subevent",
    "/r/NotHasProperty":  "is_not",
    "/r/NotCapableOf":    "cannot",
    "/r/NotDesires":      "does_not_desire",
    "/r/CreatedBy":       "createdby",
    "/r/ReceivesAction":  "receives_action",
}


# `/c/en/some_word/n/wn/animal` -> "some_word"
_URI_RE = re.compile(r"^/c/([a-z]+)/([^/]+)(?:/.*)?$")


def parse_uri(uri: str) -> tuple[str, str] | None:
    """Return (lang, concept) from a ConceptNet URI. None if unparseable."""
    m = _URI_RE.match(uri)
    if not m:
        return None
    lang, concept = m.group(1), m.group(2)
    return lang, concept


def map_relation(rel_uri: str) -> str | None:
    """Map ConceptNet's relation URI to RCK's canonical relation, or None."""
    return CONCEPTNET_RELATION_MAP.get(rel_uri)


def parse_conceptnet_tsv(
    path: str | Path,
    *,
    language: str = "en",
    min_weight: float = 1.0,
    max_rows: int | None = None,
) -> Iterable[tuple[str, str, str, float]]:
    """Stream (s, r, o, weight) from a ConceptNet TSV file.

    Filters to assertions where BOTH endpoints are in `language`,
    relation is in our map, and weight >= min_weight.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            if len(row) < 5:
                continue
            _uri, rel_uri, start_uri, end_uri, weight_str = row[:5]
            mapped_rel = map_relation(rel_uri)
            if mapped_rel is None:
                continue
            try:
                weight = float(weight_str)
            except ValueError:
                continue
            if weight < min_weight:
                continue
            start = parse_uri(start_uri)
            end = parse_uri(end_uri)
            if start is None or end is None:
                continue
            if start[0] != language or end[0] != language:
                continue
            yield start[1].lower(), mapped_rel, end[1].lower(), weight


def import_conceptnet(
    kb: ShardedKnowledgeBase, path: str | Path,
    *, language: str = "en",
    min_weight: float = 1.0,
    max_rows: int | None = None,
    provenance: ProvenanceStore | None = None,
    log_every: int = 50_000,
) -> dict:
    """Stream a ConceptNet TSV into the KB. Returns stats."""
    t0 = time.time()
    n = 0; skipped = 0
    for s, r, o, weight in parse_conceptnet_tsv(
        path, language=language,
        min_weight=min_weight, max_rows=max_rows,
    ):
        # Multi-word concepts use underscores in CN already; keep as-is.
        kb.store({"S": s, "R": r, "O": o})
        if provenance is not None:
            provenance.store(s, r, o,
                             source="conceptnet5.7",
                             confidence=min(weight / 4.0, 1.0),
                             tags={"conceptnet"})
        n += 1
        if log_every and n % log_every == 0:
            elapsed = time.time() - t0
            rate = n / elapsed
            print(f"  loaded {n:,} facts ({rate:.0f}/s) in {elapsed:.1f}s")
    return {
        "facts": n, "skipped": skipped,
        "elapsed_s": time.time() - t0,
        "rate_per_sec": n / max(time.time() - t0, 1e-6),
    }


# ---------------------------------------------------------------------------
#  Synthetic ConceptNet sample for tests (no 700MB download required)
# ---------------------------------------------------------------------------

SAMPLE_CONCEPTNET_TSV = """\t/r/IsA\t/c/en/dog\t/c/en/animal\t1.5
\t/r/IsA\t/c/en/cat\t/c/en/animal\t1.5
\t/r/HasA\t/c/en/dog\t/c/en/tail\t2.0
\t/r/AtLocation\t/c/en/fish\t/c/en/water\t1.0
\t/r/UsedFor\t/c/en/knife\t/c/en/cutting\t1.5
\t/r/MadeOf\t/c/en/window\t/c/en/glass\t1.2
\t/r/Causes\t/c/en/rain\t/c/en/wetness\t1.0
\t/r/CapableOf\t/c/en/bird\t/c/en/flying\t1.5
\t/r/PartOf\t/c/en/wheel\t/c/en/car\t1.5
\t/r/HasProperty\t/c/en/snow\t/c/en/white\t1.5
\t/r/IsA\t/c/fr/chien\t/c/fr/animal\t1.5
\t/r/RareRelation\t/c/en/foo\t/c/en/bar\t1.0
"""


def write_sample_tsv(path: str | Path) -> None:
    """Write the synthetic sample to disk for test fixtures."""
    Path(path).write_text(SAMPLE_CONCEPTNET_TSV, encoding="utf-8")
