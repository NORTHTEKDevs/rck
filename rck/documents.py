"""Document handling -- ingest .txt / .md / .json / .jsonl.

LLMs handle documents via context-stuffing. RCK ingests them: parse
into sentences, extract triples via Open IE, store with provenance
tagging the document as source. After ingestion, the document's
content is part of RCK's queryable knowledge.

Supported formats:
  - .txt / .md   : plain text, run Open IE on each sentence
  - .json        : structured triples {"s","r","o"}
  - .jsonl       : one JSON triple per line
  - .csv         : columns s,r,o

Public API:
  ingest_file(kb, path, source_name=None)        -> stats
  ingest_directory(kb, dir_path, glob_pattern)   -> stats
  query_document(kb, source_name, query)         -> grounded answer
"""
from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

from rck.knowledge_base import ShardedKnowledgeBase
from rck.open_ie import extract_triples_from_text
from rck.provenance import ProvenanceStore
from rck.tokenizer import sentences


def ingest_text_file(kb: ShardedKnowledgeBase, path: str | Path,
                     source_name: str | None = None,
                     provenance: ProvenanceStore | None = None) -> dict:
    """Ingest a plain-text / markdown file via Open IE."""
    path = Path(path)
    if source_name is None:
        source_name = str(path.name)
    text = path.read_text(encoding="utf-8", errors="ignore")
    triples = extract_triples_from_text(text)
    for s, r, o in triples:
        kb.store({"S": s, "R": r, "O": o})
        if provenance is not None:
            provenance.store(s, r, o, source=source_name,
                             tags={"document"})
    return {"path": str(path), "source": source_name,
            "triples": len(triples)}


def ingest_jsonl(kb: ShardedKnowledgeBase, path: str | Path,
                 source_name: str | None = None,
                 provenance: ProvenanceStore | None = None) -> dict:
    path = Path(path)
    if source_name is None:
        source_name = str(path.name)
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s, r, o = str(rec["s"]), str(rec["r"]), str(rec["o"])
            kb.store({"S": s.lower(), "R": r.lower(), "O": o.lower()})
            if provenance is not None:
                provenance.store(s, r, o, source=source_name,
                                 tags={"jsonl"})
            n += 1
    return {"path": str(path), "source": source_name, "triples": n}


def ingest_csv(kb: ShardedKnowledgeBase, path: str | Path,
               source_name: str | None = None,
               provenance: ProvenanceStore | None = None,
               delimiter: str = ",") -> dict:
    path = Path(path)
    if source_name is None:
        source_name = str(path.name)
    n = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            s = str(row.get("s", "")).strip().lower()
            r = str(row.get("r", "")).strip().lower()
            o = str(row.get("o", "")).strip().lower()
            if not (s and r and o):
                continue
            kb.store({"S": s, "R": r, "O": o})
            if provenance is not None:
                provenance.store(s, r, o, source=source_name,
                                 tags={"csv"})
            n += 1
    return {"path": str(path), "source": source_name, "triples": n}


def ingest_file(kb: ShardedKnowledgeBase, path: str | Path,
                source_name: str | None = None,
                provenance: ProvenanceStore | None = None) -> dict:
    """Dispatch based on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return ingest_jsonl(kb, path, source_name, provenance)
    if suffix == ".csv":
        return ingest_csv(kb, path, source_name, provenance)
    # .txt, .md, anything else -> treat as plain text.
    return ingest_text_file(kb, path, source_name, provenance)


def ingest_directory(kb: ShardedKnowledgeBase, dir_path: str | Path,
                     glob_pattern: str = "*.txt",
                     provenance: ProvenanceStore | None = None) -> dict:
    """Ingest every file matching glob in a directory."""
    dir_path = Path(dir_path)
    files: list[Path] = sorted(dir_path.glob(glob_pattern))
    total_triples = 0
    per_file = []
    for f in files:
        stats = ingest_file(kb, f, provenance=provenance)
        total_triples += stats["triples"]
        per_file.append(stats)
    return {"files": len(files), "total_triples": total_triples,
            "per_file": per_file}


def summarize_document(kb: ShardedKnowledgeBase,
                       source_name: str,
                       provenance: ProvenanceStore,
                       max_facts: int = 12) -> str:
    """Summarize what RCK learned from a given document."""
    matching = []
    for (s, r, o), rec in provenance._records.items():
        if rec.source == source_name or source_name in rec.source:
            matching.append((s, r, o))
    if not matching:
        return f"I have no record of ingesting {source_name}."
    matching = matching[:max_facts]
    lines = [f"# Summary of {source_name}",
             f"",
             f"I extracted {len(matching)} fact(s) from this document:",
             ""]
    for s, r, o in matching:
        lines.append(f"- {s} {r} {o}".replace("_", " "))
    return "\n".join(lines)
