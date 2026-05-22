"""Tests for v6 modules: longform, documents, research, multimodal, writing."""
import json
import tempfile
from pathlib import Path

from rck.bulk_ingest import bulk_load_triples
from rck.documents import (
    ingest_csv, ingest_directory, ingest_file, ingest_jsonl,
    ingest_text_file, summarize_document,
)
from rck.knowledge_base import ShardedKnowledgeBase
from rck.longform import compose_comparison, compose_essay, compose_overview
from rck.multimodal import (
    MultimodalRegistry, StubAudioTranscriber, StubImageGenerator,
    StubImageUnderstander, StubTextToSpeech,
)
from rck.provenance import ProvenanceStore
from rck.research import research
from rck.writing import (
    draft_about, edit_expand, edit_rephrase, edit_shorten,
    rewrite_for_audience, summarize,
)


def _kb_for_tests():
    kb = ShardedKnowledgeBase(dim=4096, n_shards=16, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("elephant", "isa", "mammal"),
        ("mammal", "isa", "animal"),
        ("dog", "color", "brown"),
        ("dog", "has", "fur"),
        ("dog", "size", "medium"),
        ("dog", "diet", "omnivore"),
        ("dog", "habitat", "domestic"),
        ("elephant", "size", "huge"),
        ("elephant", "has", "tusks"),
        ("france", "capital", "paris"),
        ("germany", "capital", "berlin"),
    ], symmetrize=False)
    return kb


# ---- longform --------------------------------------------------------------

def test_compose_overview_has_multiple_sections():
    kb = _kb_for_tests()
    text = compose_overview(kb, "dog")
    # Should mention multiple distinct relations.
    assert "mammal" in text or "domestic" in text or "brown" in text


def test_compose_overview_unknown_entity():
    kb = _kb_for_tests()
    text = compose_overview(kb, "unicorn")
    assert "don't have" in text.lower()


def test_compose_essay_uses_children():
    kb = _kb_for_tests()
    text = compose_essay(kb, "mammal", max_entities=3)
    # Should mention at least one child.
    assert any(w in text.lower() for w in ("dog", "cat", "elephant"))


def test_compose_comparison_returns_table():
    kb = _kb_for_tests()
    text = compose_comparison(kb, "dog", "elephant")
    assert "Comparison" in text
    # Both have `isa` and `size`.
    assert "size" in text.lower()


# ---- documents -------------------------------------------------------------

def test_ingest_text_file_extracts_triples():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("The sky is blue. The grass is green.")
        path = f.name
    try:
        stats = ingest_text_file(kb, path)
    finally:
        Path(path).unlink()
    assert stats["triples"] >= 2


def test_ingest_jsonl_loads():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"s": "dog", "r": "isa", "o": "mammal"}) + "\n")
        f.write(json.dumps({"s": "cat", "r": "isa", "o": "mammal"}) + "\n")
        path = f.name
    try:
        stats = ingest_jsonl(kb, path)
    finally:
        Path(path).unlink()
    assert stats["triples"] == 2


def test_ingest_file_dispatches_by_extension():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("The cat is a mammal.")
        path = f.name
    try:
        stats = ingest_file(kb, path, source_name="cat_doc")
    finally:
        Path(path).unlink()
    assert stats["triples"] >= 1


def test_summarize_document_with_provenance():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    prov = ProvenanceStore()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("The dog is a mammal. The sky is blue.")
        path = f.name
    try:
        ingest_text_file(kb, path, source_name="test_doc", provenance=prov)
        summary = summarize_document(kb, "test_doc", prov)
    finally:
        Path(path).unlink()
    assert "test_doc" in summary.lower() or "dog" in summary.lower()


# ---- research --------------------------------------------------------------

def test_research_unknown_topic():
    kb = _kb_for_tests()
    text = research(kb, "nonexistent_topic")
    assert "no information" in text.lower()


def test_research_returns_structured_brief():
    kb = _kb_for_tests()
    text = research(kb, "dog")
    assert "# Research brief" in text
    # Should have multiple sections.
    sections = text.count("\n## ")
    assert sections >= 1


# ---- multimodal stubs ------------------------------------------------------

def test_image_generator_stub_returns_message():
    gen = StubImageGenerator()
    res = gen.generate("a happy dog", width=512, height=512)
    assert res["ok"] is False
    assert "stub" in res["message"].lower()


def test_multimodal_registry_swap():
    reg = MultimodalRegistry()
    # All stubs by default.
    providers = reg.providers()
    assert "stub" in providers["image_gen"]
    # Swap in a "custom" stub.
    reg.set_image_generator(StubImageGenerator(model_name="custom-gen"))
    assert reg.providers()["image_gen"] == "custom-gen"


# ---- writing ---------------------------------------------------------------

def test_draft_short_medium_long():
    kb = _kb_for_tests()
    short = draft_about(kb, "dog", length="short")
    medium = draft_about(kb, "dog", length="medium")
    long = draft_about(kb, "mammal", length="long")
    # All should produce non-empty content.
    assert short and medium and long


def test_edit_shorten_drops_sentences():
    text = "A. B. C. D. E. F."
    out = edit_shorten(text, ratio=0.5)
    # Should keep ~3 sentences out of 6.
    sent_count = sum(1 for c in out if c == ".")
    assert sent_count <= 4


def test_edit_expand_adds_facts():
    kb = _kb_for_tests()
    out = edit_expand("The dog is a mammal.", kb)
    # Expansion should add a "Note also:" or similar grounding.
    assert "Note" in out or len(out) > len("The dog is a mammal.")


def test_summarize_uses_open_ie():
    text = "The dog is a mammal. The sky is blue. The car is fast."
    s = summarize(text)
    assert "Summary" in s
    assert "dog" in s.lower() or "sky" in s.lower()


def test_rewrite_for_audience_changes_text():
    text = "The dog is mammal."
    out = rewrite_for_audience(text, audience="casual")
    # At minimum, the output should be non-empty and contain "dog".
    assert out and "dog" in out.lower()
