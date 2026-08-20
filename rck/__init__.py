"""Resonant Cognitive Kernel.

A symbolic, non-generative reasoning system that is fundamentally
different from LLMs. Facts live as discrete (subject, relation,
object) triples in a sharded HRR/VSA hyperdimensional vector memory;
reasoning is an explicit, inspectable pipeline: multi-hop chain
walking with calibrated confidence, gated fact induction, symbolic
rule extraction and instantiation, contradiction detection with
belief revision, negative facts, counterfactuals, and federated
merge. Every stored or derived fact carries provenance, and
`explain_why` traces any answer back to user-asserted facts.

Start with `rck.conscious_agent.ConsciousAgent`. No GPU required;
learning is O(1) fact storage, not retraining.

(The v1.x generative components some module names still reflect --
PCN, LSM, Tsetlin, workspace -- are an archived earlier prototype,
kept importable for continuity; they are not part of the v15 thesis.
See the paper in papers/rck-architecture/.)
"""

__version__ = "15.3.1"

import warnings

from rck.conscious_agent import ConsciousAgent
from rck.knowledge_base import ShardedKnowledgeBase
from rck.replay import DecisionRecord, record_decision, replay
from rck.snapshot_hash import state_hash
from rck.session import save_session, load_session
from rck.bulk_ingest import bulk_load_jsonl, bulk_load_csv, bulk_load_triples

__all__ = [
    "ConsciousAgent",
    "ShardedKnowledgeBase",
    "DecisionRecord",
    "record_decision",
    "replay",
    "state_hash",
    "save_session",
    "load_session",
    "bulk_load_jsonl",
    "bulk_load_csv",
    "bulk_load_triples",
]

# Everything below this line used to be re-exported from the package
# root. It is not deleted -- it stays fully reachable by importing the
# module directly (`from rck.vsa import bind`), which is the
# supported path and emits no warning. Accessing it through
# `rck.<name>` still works, for backward compatibility, but now warns:
# the top-level namespace advertises the product surface (above), not
# the research substrate.
#
# Built programmatically from the pre-subtraction `rck/__init__.py`
# (29 names in the old __all__, minus the 11 that stayed frozen above)
# via `inspect.getmodule` -- see docs/plans/2026-08-19-api-subtraction.md.
_DEMOTED = {
    "ActionRegistry": "rck.actions",
    "ActiveInference": "rck.fep",
    "BigramMemory": "rck.bigram",
    "Codebook": "rck.codebook",
    "ColumnEnsemble": "rck.columns",
    "CompositionalReasoner": "rck.compose",
    "GenerativeRCK": "rck.generative",
    "GlobalWorkspace": "rck.workspace",
    "LiquidStateMachine": "rck.lsm",
    "PCNEncoder": "rck.pcn",
    "Personality": "rck.personality",
    "RCKAgent": "rck.agent",
    "RelationalMemory": "rck.relational",
    "TsetlinLayer": "rck.tsetlin",
    "auto_symmetrize": "rck.bulk_ingest",
    "bind": "rck.vsa",
    "binarize": "rck.vsa",
    "bundle": "rck.vsa",
    "cosine": "rck.vsa",
    "detect_correction": "rck.corrections",
    "extract_triples_from_text": "rck.open_ie",
    "make_default_registry": "rck.actions",
    "permute": "rck.vsa",
    "unbind": "rck.vsa",
}


def __getattr__(name):
    if name in _DEMOTED:
        warnings.warn(
            f"rck.{name} is not part of the public API and will stop being "
            f"re-exported from the package root. Import it directly: "
            f"from {_DEMOTED[name]} import {name}",
            DeprecationWarning, stacklevel=2,
        )
        import importlib
        return getattr(importlib.import_module(_DEMOTED[name]), name)
    raise AttributeError(f"module 'rck' has no attribute {name!r}")
