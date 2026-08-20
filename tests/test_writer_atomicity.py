"""Task 2: every persistence writer routes through rck.atomic, so that a
broken os.replace leaves the destination untouched rather than partially
(or fully, via a bypassing raw write) written.

Parametrized per writer -- NOT one aggregate call. A single combined
save() (e.g. ConsciousAgent.save_state's dict literal, evaluated
left-to-right by Python) aborts on the FIRST writer's exception and never
exercises the rest, which would let 4 of 5 target files go unconverted
with the test still green. See docs/plans/2026-08-19-durability.md Task 2.
"""
from __future__ import annotations

import os

import pytest


def _writer_case(writer, tmp_path):
    if writer == "skills":
        from rck.skills import SkillLibrary
        lib = SkillLibrary()
        out = tmp_path / "skills.jsonl"

        def setup():
            lib.record_success([("O", "isa")])

        def call():
            lib.save(out)

        return setup, call, out

    if writer == "provenance":
        from rck.provenance import ProvenanceStore
        store = ProvenanceStore()
        out = tmp_path / "provenance.jsonl"

        def setup():
            store.store("dog", "isa", "mammal")

        def call():
            store.save(out)

        return setup, call, out

    if writer == "query_memory":
        from rck.query_memory import QueryMemory
        qm = QueryMemory()
        out = tmp_path / "query_memory.jsonl"

        def setup():
            qm.record({"S": "dog", "R": "isa"}, "O", state="known",
                       top_symbol="mammal", top_score=0.9)

        def call():
            qm.save(out)

        return setup, call, out

    if writer == "persist":
        from rck.agent import RCKAgent
        from rck import persist
        agent = RCKAgent(hv_dim=128, n_columns=2, reservoir_dim=16,
                          n_clauses=4, vocab_size=16, fep_rank=8,
                          bigram_order=1, seed=0)
        base = tmp_path / "agent"
        out = base.with_suffix(".json")

        def setup():
            agent.observe("abc", learn=True)

        def call():
            persist.save(agent, base)

        return setup, call, out

    if writer == "session":
        from rck.conscious_agent import ConsciousAgent
        from rck import session as session_mod
        agent = ConsciousAgent(dim=256, n_shards=4, install_self=False)
        out_dir = tmp_path / "sess"
        out = out_dir / "knowledge.npz"

        def setup():
            agent.tell("dog", "isa", "mammal")

        def call():
            session_mod.save_session(agent, out_dir)

        return setup, call, out

    if writer == "cli":
        from rck.agent import RCKAgent
        from rck import cli as cli_mod
        agent = RCKAgent(hv_dim=128, n_columns=2, reservoir_dim=16,
                          n_clauses=4, vocab_size=16, fep_rank=8,
                          bigram_order=1, seed=0)
        base = tmp_path / "model"
        out = base.with_suffix(".json")

        def setup():
            agent.observe("abc", learn=True)

        def call():
            cli_mod._save(agent, base)

        return setup, call, out

    if writer == "identity":
        from rck.identity import IdentityStore
        store = IdentityStore(storage_dir=tmp_path / "identities")
        out = tmp_path / "identities" / "alice.json"

        def setup():
            store.get_or_create("alice").touch()

        def call():
            store.save("alice")

        return setup, call, out

    raise ValueError(writer)


@pytest.mark.parametrize("writer", [
    "skills", "provenance", "query_memory", "persist", "session",
    "cli", "identity",
])
def test_each_writer_is_atomic(writer, tmp_path, monkeypatch):
    setup, call, out = _writer_case(writer, tmp_path)
    setup()
    monkeypatch.setattr(
        os, "replace",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("boom")),
    )
    with pytest.raises(OSError):
        call()
    assert not out.exists(), f"{writer} bypassed rck.atomic and wrote {out}"
