"""Tests for the ConceptNet schema mapper + loader."""
import tempfile
from pathlib import Path

from rck.bulk_ingest import bulk_load_triples
from rck.conceptnet_loader import (
    CONCEPTNET_RELATION_MAP, SAMPLE_CONCEPTNET_TSV,
    import_conceptnet, map_relation, parse_conceptnet_tsv, parse_uri,
    write_sample_tsv,
)
from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore


# ---- URI parsing ----------------------------------------------------------

def test_parse_uri_basic():
    assert parse_uri("/c/en/dog") == ("en", "dog")
    assert parse_uri("/c/en/dog/n") == ("en", "dog")
    assert parse_uri("/c/en/dog/n/wn/animal") == ("en", "dog")


def test_parse_uri_other_language():
    assert parse_uri("/c/fr/chien") == ("fr", "chien")
    assert parse_uri("/c/ja/inu") == ("ja", "inu")


def test_parse_uri_malformed_returns_none():
    assert parse_uri("not_a_uri") is None
    assert parse_uri("") is None


# ---- relation mapping -----------------------------------------------------

def test_map_relation_known():
    assert map_relation("/r/IsA") == "isa"
    assert map_relation("/r/HasA") == "has"
    assert map_relation("/r/AtLocation") == "locatedin"
    assert map_relation("/r/UsedFor") == "usedfor"


def test_map_relation_unknown_returns_none():
    assert map_relation("/r/SomeUnmappedRelation") is None


def test_relation_map_covers_core_relations():
    core = {"/r/IsA", "/r/HasA", "/r/AtLocation", "/r/Causes",
            "/r/UsedFor", "/r/MadeOf", "/r/PartOf"}
    for rel in core:
        assert rel in CONCEPTNET_RELATION_MAP


# ---- streaming + filtering ------------------------------------------------

def _sample_path() -> Path:
    p = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False)
    p.close()
    write_sample_tsv(p.name)
    return Path(p.name)


def test_parse_conceptnet_tsv_filters_to_english():
    path = _sample_path()
    try:
        triples = list(parse_conceptnet_tsv(path, language="en"))
    finally:
        path.unlink()
    # 10 EN assertions in the sample (after filtering FR + the rare
    # unmapped relation).
    assert len(triples) == 10
    for s, r, o, w in triples:
        # No French concepts.
        assert "chien" not in (s, o)


def test_parse_conceptnet_tsv_min_weight():
    path = _sample_path()
    try:
        triples = list(parse_conceptnet_tsv(path, language="en",
                                             min_weight=1.5))
    finally:
        path.unlink()
    # The 1.0 and 1.2 weight assertions should be dropped.
    for s, r, o, w in triples:
        assert w >= 1.5


def test_parse_conceptnet_skips_unmapped_relations():
    path = _sample_path()
    try:
        triples = list(parse_conceptnet_tsv(path, language="en"))
    finally:
        path.unlink()
    # The "/r/RareRelation" entry should be dropped.
    for s, r, o, w in triples:
        assert r != "RareRelation"
        assert r != "rarerelation"


# ---- full import into a KB -----------------------------------------------

def test_import_into_kb():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    path = _sample_path()
    try:
        stats = import_conceptnet(kb, path, min_weight=1.0, log_every=0)
    finally:
        path.unlink()
    assert stats["facts"] >= 8
    # Sanity: 'dog isa animal' should be retrievable.
    ans, score = kb.answer({"S": "dog", "R": "isa"}, "O")
    assert ans == "animal"


def test_import_with_provenance_tags():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    prov = ProvenanceStore()
    path = _sample_path()
    try:
        import_conceptnet(kb, path, min_weight=1.0,
                           provenance=prov, log_every=0)
    finally:
        path.unlink()
    rec = prov.get("dog", "isa", "animal")
    assert rec is not None
    assert "conceptnet" in rec.tags
    assert rec.source == "conceptnet5.7"


def test_parse_real_format_json_metadata_column():
    """Real conceptnet-assertions-5.7.0.csv rows carry the weight inside
    a JSON metadata blob in column 5 (there is no bare weight column).
    The loader shipped for two releases parsing column 5 as a float,
    which silently yielded ZERO triples on the actual download."""
    import json as _json

    row = "\t".join([
        "/a/[/r/IsA/,/c/en/dog/,/c/en/animal/]",
        "/r/IsA",
        "/c/en/dog",
        "/c/en/animal/n",
        _json.dumps({"dataset": "/d/conceptnet/4/en",
                     "license": "cc:by/4.0",
                     "weight": 2.828}),
    ])
    low = "\t".join([
        "/a/[/r/IsA/,/c/en/cat/,/c/en/animal/]",
        "/r/IsA",
        "/c/en/cat",
        "/c/en/animal",
        _json.dumps({"dataset": "/d/wiktionary/en", "weight": 1.0}),
    ])
    p = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False,
                                    encoding="utf-8")
    p.write(row + "\n" + low + "\n")
    p.close()
    try:
        triples = list(parse_conceptnet_tsv(p.name, language="en",
                                            min_weight=2.0))
    finally:
        Path(p.name).unlink()
    assert triples == [("dog", "isa", "animal", 2.828)]
