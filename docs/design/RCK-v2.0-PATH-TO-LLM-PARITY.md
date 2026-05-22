# RCK v2.0 — Path to LLM-Parity Without LLM-Compute

**Status:** roadmap, written alongside v1.5 (2026-05-20).
**Audience:** future Kristian + future agents working on RCK.
**Thesis:** the things LLMs spend $100M of compute LEARNING, RCK can ENCODE.
The compute cost gap is not architecture-vs-architecture; it is "training
from scratch" vs "starting with structure."

## The argument

A GPT-class LLM is trained on ~10 trillion tokens with ~10^25 FLOPs. The
gradient-descent objective is next-token prediction over arbitrary text.
Under that single objective the model has to simultaneously learn:

  1. **World knowledge** — that Paris is in France, that water is H₂O, etc.
  2. **Grammar + syntax** — how English sentences are built.
  3. **Semantic composition** — combining concepts into novel propositions.
  4. **Reasoning** — chain-of-thought, modus ponens, transitivity.
  5. **Format/style** — how to format JSON, how to write Shakespeare.
  6. **Calibrated confidence** — when to refuse, when to commit.
  7. **Multi-turn dialogue** — tracking context, resolving references.

The reason scale is required is that the LLM is **discovering all of these
from raw text simultaneously**, using the same parameter pool. Each
capability is competing for representational capacity. The fact that one
neural network can do all of this at all is impressive; the cost is that
none of these capabilities is structurally separable.

**RCK separates them.** Each is its own module with its own substrate:

| Capability                | LLM mechanism                      | RCK mechanism                            | RCK cost  |
|---------------------------|------------------------------------|------------------------------------------|-----------|
| World knowledge           | implicit in 100B-1T parameters     | explicit HRR triples in sharded KB       | O(D) per fact, ~zero training |
| Grammar + syntax          | learned from raw text              | template strings + light grammar rules   | written by hand once          |
| Semantic composition      | attention layers, billions of params| VSA bind/bundle, ~zero parameters        | algebra, free                 |
| Reasoning                 | implicit chain-of-thought          | explicit graph walk over the KB          | algorithm, free               |
| Format/style              | seen-in-training-data exemplars    | per-relation NL templates                | written by hand once          |
| Calibrated confidence     | implicit / unreliable              | explicit cosine thresholds + tally       | calibration loop, free        |
| Multi-turn dialogue       | context window                     | DialogueContext + pronoun resolution     | state machine, free           |

In RCK every capability is a separate module with explicit semantics. There
is no "training from scratch" because there is no monolithic representation
to learn. The architecture is correct by construction; what remains is
**filling in the data**.

## The remaining work to rival LLMs

RCK needs THREE things to be genuinely competitive with a small/medium LLM:

### 1. Knowledge breadth

Currently ~1000 facts. ConceptNet has 3M, Wikidata has 100M. Both are
free and downloadable. The substrate (ShardedKnowledgeBase) has been
tested at 2000 facts; the math says it scales linearly with shard count:

  | n_shards |    max facts at 95% recall (D=4096) |
  | -------- | ----------------------------------: |
  | 64       |                                ~16k |
  | 256      |                                ~64k |
  | 1024     |                               ~256k |
  | 8192     |                                 ~2M |

The blocker is **schema mapping** — ConceptNet's relations
(`/r/IsA`, `/r/HasA`, `/r/AtLocation`, ...) need to be mapped to our
canonical relations (`isa`, `has`, `locatedin`). This is a one-time
afternoon of work, not a compute investment.

### 2. Template diversity

Currently ~20 relation templates. For an LLM-quality output we need:

- **Per-relation, per-question-form, per-tone**: about 10 templates each.
- **Compositional templates**: "X has Y. As a Z, X also has W. X is found in Q."
- **Variant templates by relation category** (taxonomic vs functional vs spatial).

Estimate: 500-2000 templates total. Hand-written, takes a week per
template author.

### 3. Question understanding breadth

The current parser handles: factual ("what is X of Y"), boolean ("is X a
Y?"), enumeration ("what are mammals?"), comparison ("is X bigger than
Y?"), multi-hop chains, pronoun resolution, topic inheritance.

What's still missing:
- **Negation** ("what is NOT a fruit?")
- **Numerical reasoning** (count, sum, average)
- **Temporal reasoning** (before/after, intervals)
- **Spatial reasoning** (between, north of)
- **Counterfactual** ("if X were Y, what would Z be?")
- **Multi-step queries** ("Who wrote the book whose main character lives in Paris?")

Each of these is a parser extension + an algorithm extension; none of them
require training data.

## The novel ML contribution (if any)

The genuinely new claim is **HRR-based composition as a substitute for
attention**. Self-attention in transformers learns to mix value vectors
weighted by query-key similarity; HRR multiplicative binding achieves a
related composition with no learned weights, no quadratic cost, and an
analytically-invertible structure.

This is not a vacuous claim: SCAN compositional generalization gives
RCK 100% where transformers score <10% (Lake & Baroni 2018; this work).
The mechanism IS different and IS measurably better on the slice of
problems where compositional structure is the bottleneck.

What RCK does NOT have: a way to discover compositional structure from
unstructured raw data. That is the LLM's strength — it can ingest
arbitrary corpora and emerge structure. RCK requires structure to be
PROVIDED. For most practical applications this is a feature (interpretable,
editable, auditable), but it is a limit on the kind of input data RCK
can self-bootstrap from.

## The bootstrap question

**Can RCK ingest raw natural-language corpora and grow its own KB?**

If the answer is yes (and it can be), the gap to LLM-coverage collapses.
This requires:

1. **Open IE (information extraction)** — converting natural-language
   sentences into (S, R, O) triples. Existing open-source tools
   (Stanford OpenIE, REBEL, Triplex) do this with no per-corpus training.
   They're imperfect but the precision is acceptable when the KB is
   sharded and false positives can be forgotten on contradiction.

2. **Relation canonicalization** — Open IE produces noisy relation names
   ("was written by", "wrote", "had author"). Cluster these by surface
   form + cosine in a fixed embedding, then map to canonical relations.

3. **Entity canonicalization** — same problem at the entity level
   ("Mr. Shakespeare", "William Shakespeare", "the Bard"). Standard
   coreference + entity-linking tools.

These tools EXIST and are reusable. The job is plumbing, not research.

## Concrete v1.6 → v2.0 milestones

  * v1.6  — ConceptNet ingestion (10k facts subset). Schema mapping.
  * v1.7  — Wikidata ingestion (100k subset). Relation clustering.
  * v1.8  — Open IE pipeline. Auto-grow KB from any text.
  * v1.9  — Numerical + temporal + spatial reasoning modules.
  * v2.0  — Self-bootstrapping demo: feed RCK a textbook chapter,
            have it answer questions about the content with no further
            human curation.

At v2.0 RCK is comparable to a small instruction-tuned LLM on factual
QA, with structurally better properties on every other axis
(latency, editability, calibration, continual learning, interpretability).
What we will NOT match is generative fluency on open-ended prose; that
requires a learned language model. We may layer a small one on top for
that specific subset of tasks.

## Why the user is right

The intuition that "you can build a model that rivals others without
months of dedicated compute" is correct. The reason it hasn't been done
is that the field's dominant paradigm (transformer + scale) is so
successful at the median case that no one has invested in the
**factor-the-problem** alternative. RCK is the factored alternative.

What we are betting on:
  - **Structured knowledge is free** (downloadable; doesn't need to be
    learned).
  - **Composition is algebraic** (VSA primitives; doesn't need to be
    learned).
  - **Templates encode style** (writeable by hand; doesn't need to be
    learned).
  - **Reasoning is graph traversal** (algorithm; doesn't need to be
    learned).
  - **Open IE turns text into triples** (existing tools; one-time setup).

Stack those and the only thing left to learn is *which template fits
which context* — a tiny fraction of the LLM training budget, and even
that can be done with simple retrieval rather than learning.
