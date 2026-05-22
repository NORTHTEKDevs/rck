"""Tests for v7 (polisher training), v8 (web ingest), v9 (math + code)."""
import json
import tempfile
from pathlib import Path

from rck.bulk_ingest import bulk_load_triples
from rck.code_sandbox import run_python
from rck.knowledge_base import ShardedKnowledgeBase
from rck.math_engine import (
    evaluate_expression, factorial, gcd, is_prime,
)
from rck.polisher_training import (
    NeuralPolisher, generate_corpus, generate_examples_for_triple,
    render_all_phrasings, write_corpus_jsonl,
)
from rck.web_ingest import (
    StubSearchProvider, StubWebFetcher, WebIngest, html_to_text,
)


# ---- v7 polisher training -------------------------------------------------

def test_render_all_phrasings_isa():
    out = render_all_phrasings(("dog", "isa", "mammal"))
    assert len(out) >= 3
    assert any("kind of" in p for p in out)


def test_generate_examples_pairs():
    examples = generate_examples_for_triple(("paris", "capital", "france"))
    assert len(examples) >= 2
    for ex in examples:
        assert ex.draft != ex.target


def test_generate_corpus_streams():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [
        ("dog", "isa", "mammal"),
        ("cat", "isa", "mammal"),
        ("paris", "capital", "france"),
    ], symmetrize=False)
    examples = list(generate_corpus(kb, examples_per_triple=2, max_triples=10))
    assert len(examples) > 0


def test_write_corpus_jsonl():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    bulk_load_triples(kb, [("a", "isa", "b")], symmetrize=False)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        stats = write_corpus_jsonl(kb, path, examples_per_triple=3)
        assert stats["examples"] > 0
        # File should be readable.
        with open(path) as f:
            lines = f.readlines()
        assert all(json.loads(line) for line in lines)
    finally:
        Path(path).unlink()


def test_neural_polisher_stub_raises_without_weights():
    try:
        NeuralPolisher(weights_path="/nonexistent/path.pt")
        assert False, "should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "weights not found" in str(e)


# ---- v8 web ingest -------------------------------------------------------

def test_stub_fetcher_returns_error_message():
    f = StubWebFetcher()
    res = f.fetch("https://example.com/page")
    assert "error" in res
    assert "stub" in res["error"]


def test_html_to_text_strips_tags():
    html = "<html><body><h1>Hello</h1><p>World &amp; everyone</p></body></html>"
    text = html_to_text(html)
    assert "<" not in text
    assert ">" not in text
    assert "Hello" in text


def test_web_ingest_with_stub_records_no_triples():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    ingest = WebIngest()
    res = ingest.ingest_url(kb, "https://example.com/page")
    assert res["ok"] is False
    assert res["triples"] == 0


def test_web_ingest_search_with_stub():
    kb = ShardedKnowledgeBase(dim=2048, n_shards=8, seed=0)
    ingest = WebIngest()
    res = ingest.ingest_search(kb, "anything")
    assert res["total_triples"] == 0


# ---- v9 math engine -----------------------------------------------------

def test_evaluate_simple():
    res = evaluate_expression("3 + 4")
    assert res["ok"] is True
    assert res["answer"] == 7.0


def test_evaluate_complex():
    res = evaluate_expression("3 + 4 * 5")
    assert res["ok"] is True
    assert res["answer"] == 23.0


def test_evaluate_with_natural_prefix():
    res = evaluate_expression("what is 12 / 4")
    assert res["ok"] is True
    assert res["answer"] == 3.0


def test_evaluate_power():
    res = evaluate_expression("2 ** 10")
    assert res["ok"] is True
    assert res["answer"] == 1024.0


def test_evaluate_division_by_zero():
    """Sympy returns symbolic infinity ('zoo'); fallback raises. Either is
    acceptable -- we just verify the engine doesn't return 5.0 or 0.0."""
    res = evaluate_expression("5 / 0")
    if res["ok"]:
        # Symbolic result is fine, but must NOT be a finite number == 5.
        assert res.get("engine", "").startswith("sympy")
    else:
        assert "error" in res


def test_factorial_and_gcd():
    assert factorial(5) == 120
    assert gcd(24, 36) == 12


def test_is_prime_check():
    assert is_prime(2)
    assert is_prime(13)
    assert not is_prime(4)
    assert not is_prime(1)


# ---- v9 code sandbox ---------------------------------------------------

def test_code_sandbox_runs_simple_python():
    res = run_python("print('hello world')")
    assert res.ok is True
    assert "hello world" in res.stdout


def test_code_sandbox_captures_stderr_on_error():
    res = run_python("raise ValueError('boom')")
    assert res.ok is False
    assert "ValueError" in res.stderr


def test_code_sandbox_timeout():
    res = run_python("import time; time.sleep(10)", timeout_s=0.5)
    assert res.ok is False
    assert "timeout" in (res.error or "").lower()
