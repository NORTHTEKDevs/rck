"""Regression tests for the CLUTRR-style kinship study (scripts/clutrr_style_study.py).

Guards against exactly the class of bug this study caught once already: a
generator/table mismatch that the symbolic control's <100% score exposed
(see the module docstring in clutrr_style_study.py, and the in-law gender
key fix it documents). If this test suite goes red, the composition table
or the tree generator regressed -- do not touch rck/.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import clutrr_style_study as study  # noqa: E402


def test_dataset_covers_every_k():
    dataset = study.generate_dataset(study.SEED)
    by_k = {}
    for ex in dataset:
        by_k[ex.k] = by_k.get(ex.k, 0) + 1
    for k in study.K_VALUES:
        assert by_k.get(k, 0) > 0, f"no examples generated for k={k}"


def test_every_example_has_exactly_k_edges():
    dataset = study.generate_dataset(study.SEED)
    for ex in dataset:
        assert len(ex.edges) == ex.k, ex.id


def test_symbolic_control_is_100_percent():
    """The harness sanity check. If this is not 100%, the tree generator
    or the composition table has a bug -- see the module docstring."""
    dataset = study.generate_dataset(study.SEED)
    correct = sum(1 for ex in dataset
                  if study.symbolic_infer(ex.edges) == ex.true_relation)
    assert correct == len(dataset)


def test_dataset_is_reproducible_under_fixed_seed():
    a = study.generate_dataset(study.SEED)
    b = study.generate_dataset(study.SEED)
    assert [(ex.id, ex.start, ex.end, ex.true_relation) for ex in a] == \
           [(ex.id, ex.start, ex.end, ex.true_relation) for ex in b]


def test_known_compositions_match_the_task_brief_examples():
    """The task brief's own worked examples: mother-of-mother = grandmother,
    mother-of-son = brother. Direct table checks, no tree involved."""
    assert study.term_from_up_down(2, 0, "F") == "grandmother"
    assert study.term_from_up_down(1, 1, "M") == "brother"
    assert study.term_from_up_down(1, 1, "F") == "sister"
    assert study.term_from_up_down(2, 1, "M") == "uncle"
    assert study.term_from_up_down(2, 1, "F") == "aunt"


def test_excluded_shapes_return_none_not_a_guess():
    """Ascent != descent, both >= 2 (e.g. 'cousin once removed') has no
    clean English term in this table -- must be excluded, never guessed."""
    assert study.term_from_up_down(3, 2, "M") is None
    assert study.term_from_up_down(0, 0, "M") is None


def test_inlaw_rules_keyed_on_endpoint_gender():
    """Regression for the exact bug the symbolic control caught: the
    in-law label must key off the ENDPOINT person's own gender, not a
    flipped/derived one."""
    assert study.symbolic_infer([("a", "brother", "b"), ("b", "wife", "c")]) == "sister-in-law"
    assert study.symbolic_infer([("a", "sister", "b"), ("b", "husband", "c")]) == "brother-in-law"
    assert study.symbolic_infer([("a", "husband", "b"), ("b", "brother", "c")]) == "brother-in-law"
    assert study.symbolic_infer([("a", "wife", "b"), ("b", "sister", "c")]) == "sister-in-law"


def test_rck_discover_infers_a_two_hop_grandmother(monkeypatch=None):
    """Fast end-to-end smoke test against the real ConsciousAgent public
    API (not the internal chain modules) -- one example, not the full 368,
    to keep the test suite quick."""
    dataset = study.generate_dataset(study.SEED)
    example = next(ex for ex in dataset if ex.k == 2 and ex.pattern == "grandmother")
    result = study.evaluate_rck_discover(example)
    assert result["found_path"] is True
    assert result["predicted"] == "grandmother"
    assert result["correct"] is True


def test_generator_is_reproducible_across_processes():
    """A fixed random.Random(seed) is NOT enough for reproducibility.

    resolve_updown and build_blood_edges pick the lowest common ancestor
    with min(common, key=...) over a set. When two ancestors tie on total
    distance, the winner was decided by str hash order, which
    PYTHONHASHSEED randomises PER PROCESS -- so the "seeded" generator
    produced a different dataset in every run (measured: 344/348/350
    unique edges across three processes). Both call sites now tie-break on
    the node id. This test runs the generator in fresh subprocesses,
    because an in-process check cannot see the bug at all.
    """
    import hashlib
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = (
        "import importlib.util,sys,hashlib;"
        "from pathlib import Path;"
        f"spec=importlib.util.spec_from_file_location('css',Path(r'{root}')/'scripts'/'clutrr_style_study.py');"
        "m=importlib.util.module_from_spec(spec);sys.modules['css']=m;spec.loader.exec_module(m);"
        "ex=m.generate_dataset();"
        "edges=sorted({e for x in ex for e in x.edges});"
        "print(hashlib.sha256(repr(edges).encode()).hexdigest())"
    )
    hashes = set()
    for _ in range(3):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=str(root))
        assert out.returncode == 0, out.stderr[-500:]
        hashes.add(out.stdout.strip())
    assert len(hashes) == 1, (
        f"generator is not reproducible across processes: {len(hashes)} "
        f"distinct datasets from 3 runs"
    )
