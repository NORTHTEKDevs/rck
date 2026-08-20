"""Phase 2 Task 3: the cross-backend parity suite -- the deliverable.

Parity does NOT mean identical everywhere. Three divergence classes are
correct behaviour, not bugs (see docs/plans/2026-08-19-dict-backend.md):

  1. Shard-partition-dependent functions (curiosity.detect_global_gaps,
     research._related_entities, subject_summary.summarize_subject):
     their early-exit `break` is checked at shard-loop granularity, so
     results depend on how facts are partitioned across shards. A dict
     backend has exactly one pseudo-shard, so it applies the same cap
     globally instead of per-shard-boundary. See the per-function
     ALLOWED_EXCEPTIONS comments in tests/test_backend_interface.py
     (lines ~83-114) for the exact mechanism of each.
  2. Density-dependent epistemic state: HRR crosstalk under load can
     push a true, stored answer's score below IDKPolicy.idk_threshold
     (or into a near-tie with noise), flipping ask_with_idk's verdict
     to IDK/AMBIGUOUS where the dict backend correctly says KNOWN.
     IDK-state equality is therefore only asserted on KBs kept under
     the HRR capacity cliff; over-cliff KBs carve this out explicitly.
  3. Induction Gate 1 (chain_induction.InductionPolicy.min_confidence)
     is substrate-relative: on dict every hop scores exactly 1.0, so
     Gate 1 can never reject a chain cascade_induct tries. On HRR, real
     crosstalk legitimately pushes some true chains below the floor.
     Dict therefore commits inductions HRR rejects, at realistic load.

If a parity assertion below needs a tolerance or inequality instead of
exact equality, the docstring says which class it belongs to, or names
it as a new finding -- never silently.

checkpoint()/load_session() parity: session.py's four `_memory` call
sites (rck/session.py:37,52,145,159 as of Task 3) did not support the
dict backend when this file was first written -- that was Task 4's
job. The checkpoint/load_session parity assertion therefore lives at
the bottom of this file, added once Task 4 landed, rather than in its
thematic position next to the other equality assertions above. See
the final report for this plan-vs-codebase sequencing note.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = str(_ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rck.conscious_agent import ConsciousAgent
from rck.cascading_induction import cascade_induct
from rck.chain_induction import InductionPolicy
from rck.dreaming import compress_duplicates
from rck.curiosity import detect_global_gaps
from rck.research import _related_entities
from rck.subject_summary import summarize_subject
from rck.provenance import ProvenanceStore

BACKENDS = ("hrr", "dict")


def _agent(backend: str, *, dim: int = 4096, n_shards: int = 16, seed: int = 0,
           **kw) -> ConsciousAgent:
    return ConsciousAgent(dim=dim, n_shards=n_shards, seed=seed,
                          backend=backend, install_self=False, **kw)


def _twins(*, dim: int = 4096, n_shards: int = 16, seed: int = 0, **kw):
    return _agent("hrr", dim=dim, n_shards=n_shards, seed=seed, **kw), \
           _agent("dict", dim=dim, n_shards=n_shards, seed=seed, **kw)


def _tell_all(agent: ConsciousAgent, facts) -> None:
    for s, r, o in facts:
        agent.tell(s, r, o)


STAIRCASE = [("a", "isa", "b"), ("b", "isa", "c"), ("c", "isa", "d")]


def _tree_shape(node):
    """Structural comparison for ExplanationNode trees: subject/relation/
    obj/source/is_leaf/children -- deliberately excludes `confidence`,
    which is a real HRR-vs-dict numeric difference, not a structural one."""
    return (node.subject, node.relation, node.obj, node.source, node.is_leaf,
            tuple(_tree_shape(c) for c in node.children))


# =============================================================================
# Core parity: tell / deny / ask_with_idk
# =============================================================================

def test_tell_deny_ask_with_idk_parity():
    hrr, dic = _twins()
    _tell_all(hrr, STAIRCASE)
    _tell_all(dic, STAIRCASE)
    hrr.deny("a", "isa", "fish")
    dic.deny("a", "isa", "fish")

    eh = hrr.ask_with_idk({"S": "a", "R": "isa"}, "O")
    ed = dic.ask_with_idk({"S": "a", "R": "isa"}, "O")
    assert eh.state == ed.state
    assert eh.top_symbol == ed.top_symbol

    nh = hrr.ask_with_idk({"S": "a", "R": "not_isa"}, "O")
    nd = dic.ask_with_idk({"S": "a", "R": "not_isa"}, "O")
    assert nh.state == nd.state
    assert nh.top_symbol == nd.top_symbol

    # A genuine miss: IDK on both, under the capacity cliff.
    mh = hrr.ask_with_idk({"S": "nobody", "R": "isa"}, "O")
    md = dic.ask_with_idk({"S": "nobody", "R": "isa"}, "O")
    assert mh.state == md.state


# =============================================================================
# explain_why trees
# =============================================================================

def test_explain_why_parity():
    hrr, dic = _twins()
    _tell_all(hrr, STAIRCASE)
    _tell_all(dic, STAIRCASE)
    hrr.induce("a", "c")
    dic.induce("a", "c")

    eh = hrr.explain_why("a", "isa", "c")
    ed = dic.explain_why("a", "isa", "c")
    assert _tree_shape(eh) == _tree_shape(ed)


# =============================================================================
# discover / reason / induce
# =============================================================================

def test_discover_parity():
    hrr, dic = _twins()
    _tell_all(hrr, STAIRCASE)
    _tell_all(dic, STAIRCASE)
    sh = hrr.discover("a", "d", max_depth=4)
    sd = dic.discover("a", "d", max_depth=4)
    assert sh is not None and sd is not None
    assert sh["relations"] == sd["relations"]
    assert sh["directions"] == sd["directions"]


def test_reason_parity():
    hrr, dic = _twins()
    _tell_all(hrr, STAIRCASE)
    _tell_all(dic, STAIRCASE)
    rh = hrr.reason("a", ["isa", "isa"])
    rd = dic.reason("a", ["isa", "isa"])
    assert rh["answer"] == rd["answer"] == "c"


def test_induce_parity():
    hrr, dic = _twins()
    _tell_all(hrr, STAIRCASE)
    _tell_all(dic, STAIRCASE)
    ih = hrr.induce("a", "c")
    idi = dic.induce("a", "c")
    assert ih is not None and idi is not None
    assert (ih.subject, ih.relation, ih.obj, ih.verified) == \
           (idi.subject, idi.relation, idi.obj, idi.verified)


# =============================================================================
# multi-hop chains at depth 2..6
# =============================================================================

CHAIN7 = [(f"n{i}", "isa", f"n{i + 1}") for i in range(6)]


@pytest.mark.parametrize("depth", [2, 3, 4, 5, 6])
def test_multihop_chain_parity(depth):
    hrr, dic = _twins()
    _tell_all(hrr, CHAIN7)
    _tell_all(dic, CHAIN7)
    rh = hrr.reason("n0", ["isa"] * depth)
    rd = dic.reason("n0", ["isa"] * depth)
    assert rh["answer"] == rd["answer"] == f"n{depth}"


# =============================================================================
# A chain walk through a genuinely multi-valued intermediate hop [R2]
# =============================================================================

def test_chain_walk_through_multivalued_intermediate_hop():
    """[R2 required addition] `chain_walker.walk_chain` and `answer()`
    take `results[0]` with no ambiguity handling, so a multi-valued
    intermediate hop can silently walk a DIFFERENT chain on each
    backend -- documented in Task 1 as a known non-parity point (dict's
    tie-break is insertion order; HRR's is whatever the bundle's
    crosstalk ranks highest, not necessarily insertion order).

    This test does NOT assert cross-backend equality of the branch
    taken -- that would contradict the documented non-parity point.
    It asserts: (a) dict deterministically takes the first-inserted
    branch (proving OUR implementation's contract), and (b) both
    backends still complete the walk to a valid leaf without error.
    """
    facts = [
        ("a", "isa", "b"),
        ("b", "isa", "c1"),  # b is multi-valued: two "isa" targets
        ("b", "isa", "c2"),
        ("c1", "isa", "z1"),
        ("c2", "isa", "z2"),
    ]
    hrr, dic = _twins()
    _tell_all(hrr, facts)
    _tell_all(dic, facts)

    rd = dic.reason("a", ["isa", "isa"])
    assert rd["answer"] == "c1"  # dict: deterministic, first-inserted branch

    rh = hrr.reason("a", ["isa", "isa"])
    assert rh["answer"] in ("c1", "c2")  # HRR: some valid branch, not asserted which


# =============================================================================
# detect_conflicts / resolve_conflicts
# =============================================================================

def test_detect_and_resolve_conflicts_parity():
    hrr, dic = _twins()
    for agent in (hrr, dic):
        agent.knowledge.store({"S": "fish", "R": "isa", "O": "animal"})
        agent.knowledge.store({"S": "fish", "R": "isa", "O": "vegetable"})
        # Different sources so SOURCE_PRIORITY resolves the winner
        # deterministically on both backends, instead of falling through
        # to the final "kept by HRR score" tie-break -- which is a real
        # tie (both score 1.0) on dict and would be order-dependent.
        agent.provenance.store("fish", "isa", "animal", source="user")
        agent.provenance.store("fish", "isa", "vegetable", source="induced")

    ch = hrr.detect_conflicts()
    cd = dic.detect_conflicts()

    def _shape(conflicts):
        return sorted(
            (c.subject, c.relation, tuple(sorted(o for o, _ in c.candidates)))
            for c in conflicts
        )
    assert _shape(ch) == _shape(cd)

    ph = hrr.resolve_conflicts(apply=True)
    pd = dic.resolve_conflicts(apply=True)
    assert len(ph) == len(pd) == 1
    assert ph[0].keep.obj == pd[0].keep.obj == "animal"
    assert [f.obj for f in ph[0].drop] == [f.obj for f in pd[0].drop] == ["vegetable"]


# =============================================================================
# extract_rules / instantiate_rules
# =============================================================================

def test_extract_and_instantiate_rules_parity():
    hrr, dic = _twins()
    _tell_all(hrr, STAIRCASE)
    _tell_all(dic, STAIRCASE)
    for agent in (hrr, dic):
        # Populate skills via REAL chain walks (substrate-dependent
        # retrieval), not by seeding the SkillLibrary directly.
        agent.reason("a", ["isa", "isa"])
        agent.reason("b", ["isa", "isa"])

    rh = hrr.extract_rules(min_support=2)
    rd = dic.extract_rules(min_support=2)
    assert {r.signature() for r in rh.all_rules()} == {r.signature() for r in rd.all_rules()}
    assert rh.size() == rd.size() >= 1

    fh = hrr.instantiate_rules(min_support=2)
    fd = dic.instantiate_rules(min_support=2)

    def _shape(facts):
        return sorted((f.subject, f.relation, f.obj) for f in facts)
    assert _shape(fh) == _shape(fd)


# =============================================================================
# maintain()
# =============================================================================

def test_maintain_parity_under_the_cliff():
    """[R2 finding] Even on a tiny 3-fact staircase (well under any
    HRR capacity cliff), cascade_induct's Gate 1 (divergence class 3)
    still bites: measured hrr chain_induction_verified=6 vs dict=8,
    final_kb_size=11 vs 13. Ordinary HRR cleanup noise -- not density
    -- is enough to occasionally drop a chain's confidence a hair
    below InductionPolicy.min_confidence=0.20, and dict's Gate 1 can
    never reject at all. So those two keys assert `dict >= hrr`, per
    class 3, not equality. Every OTHER maintain() metric here is
    induction-count-independent and asserts exact equality."""
    hrr, dic = _twins()
    _tell_all(hrr, STAIRCASE)
    _tell_all(dic, STAIRCASE)
    sh = hrr.maintain(consolidate_episodes=False)
    sd = dic.maintain(consolidate_episodes=False)
    for key in ("rule_cascade_verified", "negations_propagated",
                "conflicts_resolved", "skills_promoted"):
        assert sh[key] == sd[key], f"{key}: hrr={sh[key]!r} dict={sd[key]!r}"
    for key in ("chain_induction_verified", "final_kb_size"):
        assert sd[key] >= sh[key], (
            f"divergence class 3 expects dict >= hrr on {key}; "
            f"got hrr={sh[key]!r} dict={sd[key]!r}"
        )


# =============================================================================
# merge_from on dict, mixed-backend raises
# =============================================================================

def test_merge_from_dict_backend():
    a = _agent("dict")
    b = _agent("dict")
    a.tell("dog", "isa", "mammal")
    b.tell("cat", "isa", "mammal")
    pre = a.knowledge.size()
    a.merge_from(b)
    assert a.knowledge.size() > pre
    ans, score = a.knowledge.answer({"S": "cat", "R": "isa"}, "O")
    assert ans == "mammal" and score == 1.0


def test_merge_from_mixed_backend_raises_type_error():
    dict_agent = _agent("dict")
    hrr_agent = _agent("hrr")
    dict_agent.tell("dog", "isa", "mammal")
    hrr_agent.tell("bird", "isa", "animal")
    with pytest.raises(TypeError):
        dict_agent.merge_from(hrr_agent)


# =============================================================================
# checkpoint() / load_session() parity (Task 4)
# =============================================================================

def test_checkpoint_load_session_parity(tmp_path):
    """Deferred here from the original Task 3 list -- session.py did
    not support the dict backend until Task 4. Now that it does, both
    backends must round-trip to the same answer and derivation."""
    from rck.session import load_session
    hrr, dic = _twins()
    _tell_all(hrr, STAIRCASE)
    _tell_all(dic, STAIRCASE)
    hrr.induce("a", "c")
    dic.induce("a", "c")

    hrr.checkpoint(tmp_path / "hrr_snap")
    dic.checkpoint(tmp_path / "dict_snap")
    hrr2 = load_session(tmp_path / "hrr_snap")
    dic2 = load_session(tmp_path / "dict_snap")

    assert hrr2.backend == "hrr"
    assert dic2.backend == "dict"
    eh = hrr2.explain_why("a", "isa", "c")
    ed = dic2.explain_why("a", "isa", "c")
    assert _tree_shape(eh) == _tree_shape(ed)


# =============================================================================
# [R2] Anchor to ground truth, not just to each other.
# =============================================================================

import clutrr_style_study as _clutrr  # noqa: E402


def _rck_discover_relation(backend: str, example) -> str | None:
    """Mirrors clutrr_style_study.evaluate_rck_discover, parametrized by
    backend instead of hardcoding ConsciousAgent(install_self=False)."""
    agent = _agent(backend, dim=4096, n_shards=16)
    for s, r, o in example.edges:
        agent.tell(s, r, o)
    spec = agent.discover(example.start, example.end, max_depth=example.k,
                          allow_reverse=False)
    if spec is None:
        return None
    rels = spec["relations"]
    if len(rels) == 2 and (rels[0] in _clutrr.SPOUSE_PRIMITIVES
                            or rels[1] in _clutrr.SPOUSE_PRIMITIVES):
        return _clutrr.symbolic_infer([("_", r, "_") for r in rels])
    if rels and all(r in _clutrr.BLOOD_PRIMITIVES for r in rels):
        return _clutrr.symbolic_infer([("_", r, "_") for r in rels])
    return None


def _clutrr_sample():
    """One example per k value (2..6), deterministic given SEED."""
    dataset = _clutrr.generate_dataset(_clutrr.SEED)
    sample = []
    seen_k = set()
    for ex in dataset:
        if ex.k not in seen_k:
            sample.append(ex)
            seen_k.add(ex.k)
        if len(seen_k) >= len(_clutrr.K_VALUES):
            break
    return sample


_CLUTRR_SAMPLE = _clutrr_sample()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("example", _CLUTRR_SAMPLE, ids=lambda ex: ex.id)
def test_discover_matches_independent_oracle(backend, example):
    """Each backend's discover()+compose is checked against
    `symbolic_infer` -- the CLUTRR-style study's independent oracle,
    computed from the tree's ground truth by a wholly separate code
    path (LCA search, not KB retrieval). Cross-backend equality could
    pass while both are wrong; this cannot."""
    oracle = _clutrr.symbolic_infer(example.edges)
    assert oracle == example.true_relation  # harness sanity, per the script's own docstring
    predicted = _rck_discover_relation(backend, example)
    assert predicted == oracle, (
        f"{backend} backend disagreed with the independent oracle on "
        f"{example.id}: predicted {predicted!r}, oracle {oracle!r}"
    )


# =============================================================================
# [R2] Divergence class 3: induction gate parity at realistic load.
# =============================================================================

def test_induction_gate_class3_dict_commits_more_at_realistic_load():
    """cascade_induct's walk_chain always uses the default geometric_mean
    rule (no config=). On dict every hop scores exactly 1.0, so 2-hop
    chain confidence = 1.0 * chain_decay**1 = 0.95, always above
    InductionPolicy.min_confidence=0.20 -- Gate 1 can never reject on
    dict. On HRR, a single shard pinned past its capacity cliff (120
    facts, dim=4096 target_fill=80, auto_reshard disabled) produces real
    crosstalk that legitimately drags some true chains' propagated
    confidence below the floor.

    Dict is therefore expected to commit AT LEAST as many verified
    inductions as HRR here -- this is the documented divergence, not a
    bug. Do NOT weaken this into an equality assertion.
    """
    n = 120
    facts = [(f"n{i}", "isa", f"n{i + 1}") for i in range(n)]

    hrr = ConsciousAgent(dim=4096, n_shards=1, backend="hrr", install_self=False)
    hrr.knowledge.auto_reshard = False  # pin to one shard past the cliff
    dic = ConsciousAgent(dim=4096, n_shards=1, backend="dict", install_self=False)
    _tell_all(hrr, facts)
    _tell_all(dic, facts)

    policy = InductionPolicy(min_confidence=0.20)
    res_hrr = cascade_induct(hrr.knowledge, max_rounds=3, probes_per_round=60,
                             policy=policy)
    res_dict = cascade_induct(dic.knowledge, max_rounds=3, probes_per_round=60,
                              policy=policy)

    assert res_dict.total_verified > 0
    assert res_dict.total_verified >= res_hrr.total_verified, (
        f"expected dict (Gate 1 never rejects) to commit >= HRR's verified "
        f"inductions; got dict={res_dict.total_verified} hrr={res_hrr.total_verified}"
    )


# =============================================================================
# [R2] Divergence class 2: density-dependent epistemic state -- carved out.
# =============================================================================

def test_idk_state_over_the_cliff_dict_stays_known_hrr_not_asserted():
    """[R2] Over the HRR capacity cliff, crosstalk can push a true, stored
    answer's score below IDKPolicy.idk_threshold, flipping ask_with_idk
    to IDK/AMBIGUOUS on HRR for a query with one unambiguous answer. The
    dict backend, being an exact index, always correctly reports KNOWN
    for a fact that is actually stored. This test deliberately does NOT
    assert HRR's state -- asserting it would either be flaky (depends on
    exactly which query happens to cross the cliff) or would silently
    re-encode the very divergence this plan documents. Every OTHER
    ask_with_idk parity assertion in this suite stays under the cliff,
    per the plan's explicit instruction, and asserts full equality."""
    n = 120
    hrr = ConsciousAgent(dim=4096, n_shards=1, backend="hrr", install_self=False)
    hrr.knowledge.auto_reshard = False
    dic = ConsciousAgent(dim=4096, n_shards=1, backend="dict", install_self=False)
    facts = [(f"n{i}", "isa", f"n{i + 1}") for i in range(n)]
    _tell_all(hrr, facts)
    _tell_all(dic, facts)

    ed = dic.ask_with_idk({"S": "n0", "R": "isa"}, "O")
    assert ed.state.value == "known"
    assert ed.top_symbol == "n1"
    # HRR's state is intentionally not asserted here -- see docstring.
    _ = hrr.ask_with_idk({"S": "n0", "R": "isa"}, "O")


# =============================================================================
# [R2] Index invariant: re-query a deduped fact, not merely count facts.
# =============================================================================

@pytest.mark.parametrize("backend", BACKENDS)
def test_requery_deduped_fact_after_compress_duplicates(backend):
    """compress_duplicates() reassigns `shard._facts = keep` directly,
    bypassing store()/forget() -- both backends' query() must reflect
    the dedup on the very next query, not merely on kb.size() (which
    compress_duplicates does not update on EITHER backend -- it was
    already stale after a direct dedup before this plan; not this
    plan's concern to fix). Task 3 must re-query, so this asserts the
    query result, not the count.

    Note: compress_duplicates() is not wired into ConsciousAgent.maintain()
    anywhere in this codebase (grepped: its only caller is
    dreaming.consolidate(), a lower-level pass ConsciousAgent never
    calls) -- so this test calls it directly on the KB rather than via
    maintain(), unlike the plan text's literal phrasing. See final report.
    """
    agent = _agent(backend)
    agent.knowledge.store({"S": "a", "R": "isa", "O": "b"})
    agent.knowledge.store({"S": "a", "R": "isa", "O": "b"})  # exact duplicate
    removed = compress_duplicates(agent.knowledge)
    assert removed == [("a", "isa", "b")]
    results = agent.knowledge.query({"S": "a", "R": "isa"}, "O", top_k=10)
    assert len(results) == 1
    assert results[0][0] == "b"


# =============================================================================
# [R2] Divergence class 1: shard-partition-dependent functions.
# Uses the accurate per-function language from
# tests/test_backend_interface.py's ALLOWED_EXCEPTIONS comments, not one
# blanket sentence.
# =============================================================================

def test_divergence_subject_summary_per_shard_early_exit_cap():
    """subject_summary.summarize_subject has the identical per-shard
    early-exit quirk documented for research.py: the `max_facts` break
    is checked at shard-loop granularity, so the exact set (and count)
    of a subject's facts collected before the cap depends on shard
    boundaries. On dict (one pseudo-shard) the cap applies exactly; on
    HRR (multiple shards) the running total can overshoot it -- measured
    here: HRR collects MORE than max_facts, dict collects EXACTLY
    max_facts."""
    hrr = ConsciousAgent(dim=1024, n_shards=32, backend="hrr", install_self=False).knowledge
    dic = ConsciousAgent(dim=1024, n_shards=32, backend="dict", install_self=False).knowledge
    for kb in (hrr, dic):
        for i in range(400):
            kb.store({"S": "star", "R": f"rel{i}", "O": f"o{i}"})

    sh = summarize_subject(hrr, ProvenanceStore(), "star", max_facts=50)
    sd = summarize_subject(dic, ProvenanceStore(), "star", max_facts=50)
    assert sd.n_facts == 50          # dict: exact cap, one pseudo-shard
    assert sh.n_facts > 50           # hrr: per-shard checkpoints overshoot it
    assert sh.n_facts != sd.n_facts  # the documented divergence, pinned


def test_divergence_research_related_entities_shard_order_dependent_content():
    """research._related_entities's incoming-facts loop breaks once
    max_each*10 is reached, but the break only exits the CURRENT shard's
    inner loop (no matching break on the outer shard loop) -- so on HRR
    the collected list is built in shard-then-insertion order, while on
    dict (one pseudo-shard) it is pure insertion order. The final
    `incoming[:max_each*4]` slice can therefore contain the SAME COUNT
    but DIFFERENT CONTENT on the two backends -- measured here: both
    return exactly 20 items, but as different sets/orders."""
    hrr = ConsciousAgent(dim=1024, n_shards=32, backend="hrr", install_self=False).knowledge
    dic = ConsciousAgent(dim=1024, n_shards=32, backend="dict", install_self=False).knowledge
    for kb in (hrr, dic):
        for i in range(400):
            kb.store({"S": f"s{i:04d}", "R": "points_to", "O": "hub"})

    rh = _related_entities(hrr, "hub", max_each=5)
    rd = _related_entities(dic, "hub", max_each=5)
    assert len(rh["incoming"]) == len(rd["incoming"]) == 20
    # dict: pure insertion order -> the first 20 subjects inserted.
    assert [s for s, _ in rd["incoming"]] == [f"s{i:04d}" for i in range(20)]
    # hrr: shard-then-insertion order -> NOT the same 20, in general.
    assert rh["incoming"] != rd["incoming"]


def test_divergence_curiosity_detect_global_gaps_shard_boundary_sample():
    """curiosity.detect_global_gaps samples entities with an early-exit
    break checked once per shard (after each shard's fact loop, not
    after each fact) -- so which entities get collected before the
    sample_size*5 cap depends on shard boundaries. On dict (one
    pseudo-shard) the cap applies globally, over the true full entity
    set; on HRR (multiple shards) it can stop after scanning only the
    first few shards, missing entities that would otherwise rank in the
    global sample -- measured here: HRR finds 0 gaps, dict finds 2, on
    the identical KB."""
    def build(kb):
        for i in range(300):
            kb.store({"S": f"e{i:04d}", "R": "isa", "O": "animal"})
            if i % 2 == 0:
                kb.store({"S": f"e{i:04d}", "R": "haslegs", "O": "four"})
        return kb

    hrr = build(ConsciousAgent(dim=1024, n_shards=64, backend="hrr",
                                install_self=False).knowledge)
    dic = build(ConsciousAgent(dim=1024, n_shards=64, backend="dict",
                                install_self=False).knowledge)

    gh = detect_global_gaps(hrr, sample_size=5, min_siblings=2)
    gd = detect_global_gaps(dic, sample_size=5, min_siblings=2)
    assert len(gh) == 0
    assert len(gd) == 2
    assert len(gh) != len(gd)  # the documented divergence, pinned
