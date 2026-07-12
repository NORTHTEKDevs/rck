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

from rck.codebook import Codebook
from rck.vsa import bind, bundle, permute, unbind, cosine, binarize
from rck.pcn import PCNEncoder
from rck.lsm import LiquidStateMachine
from rck.tsetlin import TsetlinLayer
from rck.workspace import GlobalWorkspace
from rck.fep import ActiveInference
from rck.columns import ColumnEnsemble
from rck.bigram import BigramMemory
from rck.relational import RelationalMemory
from rck.compose import CompositionalReasoner
from rck.agent import RCKAgent
from rck.generative import GenerativeRCK
from rck.knowledge_base import ShardedKnowledgeBase
from rck.conscious_agent import ConsciousAgent
from rck.bulk_ingest import (
    bulk_load_jsonl, bulk_load_csv, bulk_load_triples, auto_symmetrize,
)
from rck.personality import Personality
from rck.actions import ActionRegistry, make_default_registry
from rck.corrections import detect_correction
from rck.open_ie import extract_triples_from_text

__all__ = [
    "Codebook",
    "bind", "bundle", "permute", "unbind", "cosine", "binarize",
    "PCNEncoder",
    "LiquidStateMachine",
    "TsetlinLayer",
    "GlobalWorkspace",
    "ActiveInference",
    "ColumnEnsemble",
    "BigramMemory",
    "RelationalMemory",
    "CompositionalReasoner",
    "RCKAgent",
    "GenerativeRCK",
    "ShardedKnowledgeBase",
    "ConsciousAgent",
    "bulk_load_jsonl",
    "bulk_load_csv",
    "bulk_load_triples",
    "auto_symmetrize",
    "Personality",
    "ActionRegistry",
    "make_default_registry",
    "detect_correction",
    "extract_triples_from_text",
]
