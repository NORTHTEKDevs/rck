"""Resonant Cognitive Kernel.

A compact AI architecture that is fundamentally different from LLMs:
- Vector Symbolic Architecture as the representation substrate
- Predictive Coding for local-update perceptual encoding
- Liquid State Machine for temporal integration
- Tsetlin Machine for interpretable causal logic clauses
- Global Workspace Theory for module-broadcast attention
- Free Energy Principle / Active Inference for goal-directed action
- Thousand-Brains reference-frame column voting for uncertainty

No backprop. No GPU required. Continual learning by construction.
"""

__version__ = "15.0.0"

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
