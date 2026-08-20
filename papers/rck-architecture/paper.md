# RCK: A Hallucination-Free Reasoning System Built on Hyperdimensional Computing

**Kristian Baer** · Independent Researcher · Anchorage, Alaska · `kristianb43r@gmail.com`

*Markdown narrative version of the LaTeX paper. For the formal manuscript see `paper.tex`. This file is the human-readable rendering; it's also the source for blog posts / HN posts / newsletter pitches. The two are kept in sync; where they ever disagree, `paper.tex` is canonical.*

---

## Abstract

We describe RCK (Resonant Cognitive Kernel), a working AI system that isn't a language model. RCK stores facts as discrete (subject, relation, object) triples in a sharded hyperdimensional vector substrate, and reasons over them through an explicit, inspectable pipeline: multi-hop chain walking, fact induction with empirically-derived filters, symbolic rule extraction and instantiation, contradiction detection with belief revision, and counterfactual exploration. Every stored or derived fact carries provenance, and any answer the system gives can be traced back through a derivation graph to the user-asserted facts that grounded it.

The system runs on a single CPU thread, scales to thousands of facts in tens to hundreds of milliseconds, and **structurally cannot hallucinate in the generative sense** - there's no generative model fabricating outputs. That claim is scoped precisely in §1.2: it does not mean the knowledge base can never contain a false belief (our own v15.1 retraction, §5.1, is the counterexample); it means errors surface as retrievable, auditable, low-confidence facts or as an explicit "I don't know" - never as fluent fabrication.

We report empirical findings from the implementation:

- A six-gate filter stack on chain induction (inverse-pair, non-transitive same-relation, intermediate-cycle, no-meaningful-relation, type-signature mismatch, plus a confidence floor) yields **31/31 manually-validated semantically-correct inductions** on a 400-probe study against a 7,080-fact KB, with zero HRR self-verify failures. An earlier v15.0 framing reported "~87% precision" on the same probes; that figure measured HRR-roundtrip stability, not semantic correctness, and is retracted in this paper (§5.1).
- Switching the default confidence-propagation rule from multiplicative product to geometric mean extends the reported-confidence horizon from ~10 hops to **30+ hops at "moderate"-or-better** on synthetic linear chains whose retrieval accuracy is 100% out to 50 hops. The geometric mean is a display heuristic, not a probability; §4.2 and §5.3 report the principled successor.
- **Calibrated confidence restores real probabilistic semantics.** A raw HRR cosine is a similarity feature, not a probability: on a held-out benchmark, treating cosines as probabilities gives a Brier score of 0.568, while an isotonic (PAV) calibration fitted on 4,862 labeled retrievals achieves 0.0038 - 150× lower. Multiplying *calibrated* per-hop probabilities yields chain confidence that approximates P(every hop correct): a clean 50-hop chain reports 0.985 - high, but honestly decaying with length (§5.3).
- A per-shard **live-relation index** cuts 31-66% of HRR queries during BFS chain discovery, for 1.4-2.8× wall-clock speedups, with identical discovered chains - confirming the paper's earlier diagnosis that relation-count, not fact-count, dominates discovery cost (§5.5).
- **The substrate holds 100,000 real-world facts on a laptop.** On the first 100k English ConceptNet assertions (81,086 entities): ingest in 6.6 seconds, 100.0% recall@1 on sampled stored facts, 0.33 ms median query, 535 MB resident, 97% two-hop discovery at 30 ms median - single CPU thread. The enabler is shard-local cleanup (§4.5): a shard's bundle can only contain what that shard stored, so per-query cost is independent of vocabulary size (§5.6).
- A confidence-weighted analogy solver improves accuracy from 88.7% to 93.9% on a 115-probe commonsense benchmark, while surfacing calibrated probabilities for each candidate.
- Sparse-binary HRR, while attractive on per-atom memory grounds, has 8-16× lower per-shard capacity than dense bipolar HRR and is **not** a drop-in substrate replacement.

We release the full implementation (~16,100 lines of Python, 757 tests) under the Apache 2.0 license at <https://github.com/NORTHTEKDevs/rck>. Our aim is not to argue that RCK replaces LLMs but to show that a coherent alternative architecture - one that is auditable, editable, and structurally honest about its own uncertainty - is achievable and useful today, on commodity hardware, with no GPU.

---

## 1. Introduction

Modern AI is dominated by large language models. They're extraordinarily capable at surface fluency, surprisingly capable at many reasoning tasks, and structurally incapable of distinguishing between something they actually know and something that sounds about right. They can't tell you why they believe what they believe, because their "beliefs" live as continuous parameters distributed across billions of weights with no symbolic referent. When asked to explain their answers, they generate plausible-sounding explanations that have no causal connection to whatever produced the original output.

This is fine for a lot of tasks. It's not fine for medicine, law, science, finance, or anything where the difference between "I know this" and "this sounds right" can ruin somebody's day. The industry's response has been to build bigger models and hope the hallucination rate falls faster than the capability ceiling rises. We think a different approach is worth trying: build a system where **generative** hallucination is structurally impossible because there's no generative layer to fabricate, and retrieval errors are routed through an explicit "I don't know" path rather than dressed up as confident sentences.

This paper describes that system. RCK is a working, tested, publicly-available implementation. It's not a research prototype; it ships as a Python package and runs on a laptop. It's also not finished - we treat the release described here (v15.3) as a stable foundation rather than a final form.

### 1.1 What is new

The individual primitives RCK uses are not new - and the closest prior art is closer than a casual reading suggests, so we say precisely what we do and do not claim. Hyperdimensional computing and holographic reduced representations have carried integrated cognitive systems before (Eliasmith's Semantic Pointer Architecture and Spaun; Crawford et al.'s WordNet-scale HRR encoding; Holographic Declarative Memory inside ACT-R). Justification tracking, explanation, and belief revision are the classical territory of truth maintenance systems (Doyle; de Kleer), AGM theory, and expert-systems explanation (MYCIN). Promoting multi-hop paths to new facts behind confidence and type gates is the home turf of KG rule mining and path ranking (AMIE; PRA; NELL). Explicit negative knowledge has its own literature (Arnaout et al.; NegatER), rooted in the classical-negation vs. negation-as-failure distinction. §6 engages each of these directly.

What we claim is the combination, its empirical characterization, and the engineering that makes it run on commodity hardware:

- **A complete, integrated, open implementation** - to our knowledge the first single package that combines an HRR/VSA relational substrate with derivation-graph provenance, explain-why, contradiction detection with belief revision, asserted negative facts, gated chain induction, symbolic rule extraction, counterfactuals, and alignment-free federated merge, behind ~50 operations on one `ConsciousAgent` object, with 757 passing tests.
- **An empirically-derived six-gate induction filter stack** whose gates come from inspecting real HRR-substrate failure modes (hub round-trips, cleanup crosstalk, type confusion) - including a published retraction of our own earlier headline number when a failure-mode audit showed it measured the wrong thing (§5.1).
- **A calibration result**: the first (to our knowledge) published cosine→P(correct) calibration for chain reasoning on an HRR substrate, which both quantifies how miscalibrated raw cosines are (held-out Brier 0.568 → 0.0038) and restores honest multiplicative semantics to multi-hop confidence (§5.3).
- **A negative result** on sparse HRR substrates that may save other implementers time (§5.4).
- **A scaling result**: shard-local cleanup makes per-query cost independent of vocabulary size, taking the same substrate from a 7,080-fact ceiling to 100,000 real ConceptNet facts at 100.0% recall@1 and sub-millisecond queries on one CPU thread (§4.5, §5.6).
- **A practical artifact**: a system someone can install today, run on a laptop, and integrate into a real product via the bundled MCP server.

We make no claim that RCK is competitive with frontier LLMs on open-domain text generation, creative writing, or any task where the required output is a long free-form passage. It is not. The point of this paper is that there's a useful zone of tasks - structured-knowledge question answering with provenance, multi-hop reasoning with citations, fact ingestion with editability, contradiction-resilient knowledge management - where RCK's auditability, editability, and explicit treatment of uncertainty are worth more than the surface fluency of an LLM.

### 1.2 What "hallucination-free" does and does not mean

The title's claim invites a specific, well-documented failure mode: systems marketed as "hallucination-free" that turn out to be merely *less* wrong under adversarial audit (Magesh et al. 2024 documented exactly this for legal research tools; the survey literature on graph-grounded LLMs consistently concludes "reduce," not "eliminate" - Agrawal et al. 2024). We therefore scope the claim precisely.

**What it means.** There is no generative model anywhere in the answer path. An answer is a stored or derived (S, R, O) triple with a score, or an explicit IDK/AMBIGUOUS state. The system cannot produce fluent prose about things never asserted or derived, because it cannot produce fluent prose at all (the optional polisher renders surface form only, and is off by default). Every answer is traceable to user-asserted facts through the derivation graph.

**What it does not mean.** It does not mean the knowledge base cannot contain false beliefs. Our own v15.1 release is the demonstration: under the v15.0 induction policy, the system confidently stored type-confused derived facts like `(shark, implies, blue)` - wrong facts, held with unwarranted confidence, functionally similar to a hallucination from the user's perspective. We found this by auditing our own headline metric, retracted the number, and added the two gates that eliminate the failure class (§5.1). Nor does it mean retrieval is infallible: under bundle saturation, cleanup can return a wrong symbol - bounded, measurable (§5.3's calibration quantifies it), and surfaced as low confidence, but an error nonetheless.

**The precise claim** is: *structurally free of generative confabulation; empirically zero false stored inductions under the six-gate policy on our test corpus; retrieval error bounded, measured, and reported as calibrated confidence.* Where we say "hallucination-free" anywhere in this paper or the repository, this is the sense we mean.

---

## 2. Background

### 2.1 Hyperdimensional computing in one paragraph

Pick a large dimensionality D (we default to 4096). Map each symbol in your vocabulary to a random vector in {-1, +1}<sup>D</sup>.

Two facts about this setup matter. First, almost all such vectors are approximately orthogonal: the cosine similarity between two random vectors converges to zero as D grows. Second, you can combine vectors with elementwise multiplication (**binding**) and elementwise addition followed by sign-thresholding (**bundling**); the resulting vectors still behave like discrete symbols you can look up by similarity. Binding is its own inverse for bipolar vectors, so you can store a labeled fact and later recover any of its slots by binding with the others and looking up the result in the symbol table.

### 2.2 Holographic reduced representations for relational facts

Following Plate, we encode a fact (S, R, O) as a multiplicative bind of role-vector ⊗ symbol-vector for each slot. A relational memory is a bundle (additive sum, sign-thresholded) of many such fact vectors. To answer "what is the R of S?" we multiply the memory by the role bindings of the known slots and clean up the result against the symbol table.

Recall is approximately correct: noise from non-matching facts averages out in the bundle and the cleanup step rejects it, up to a capacity limit that depends on D. We measured this limit empirically (§5.4).

### 2.3 Sharding

A single bundle saturates at ~80 facts for D=4096 before recall starts to degrade. To scale, we shard the relational memory into N independent bundles routed by a stable hash of (S, R). Two implications: each shard has its own bundle and capacity, so the system scales linearly with N; and the role-vector embedding is shared across all shards (name-hashed) so federated merge across agents is a per-shard bundle sum.

---

## 3. Architecture

The bottom layer is `ShardedKnowledgeBase`: a list of N `RelationalMemory` shards, each of dimension D, sharing a single `Codebook` of symbol vectors. Facts are routed to a shard via blake2b-based stable hashing of (subject, relation). Storage is O(1). Retrieval is one bind plus one cleanup against the codebook. A snapshot `RelationIndex` (v15.2) records which relations have facts in which shards, so search-time callers can skip queries that could only return crosstalk noise (§5.5).

`ConsciousAgent` composes the substrate with the higher-level machinery. The agent owns:

- a knowledge base (the sharded HRR memory)
- a provenance store (per-fact metadata)
- a skill library (a record of which chain shapes have succeeded)
- a query memory (an episodic log of every ask)
- a chain cache
- a belief KB for theory-of-mind facts about other agents
- a small language-model component for surface polish (optional)

### Reasoning primitives

| Primitive | What it does |
|---|---|
| Chain walking | Walk a known sequence of relations from a start entity, propagating confidence |
| Chain discovery | BFS over the HRR graph to find a chain from start to target, pruned by the live-relation index |
| Chain induction | Commit a confident chain as a new direct edge (six-gate filter; see §4.1, §5.1) |
| Rule extraction | Lift repeated chain shapes into symbolic universal rules |
| Rule instantiation | Walk the KB and apply stored rules forward to derive more facts |
| Analogy | "A:B::C:?" via two relational queries plus a softmax over candidate relations |
| Causal | Specialized walker over the `causes` relation, forward and backward |

### Knowledge management

| Module | What it does |
|---|---|
| Provenance | Every fact carries source, timestamp, confidence, count, tags, derivation |
| Explain-why | Recursive expansion of the derivation graph back to user-asserted facts |
| Contradiction detection | Surfaces (S, R, A) + (S, R, B) on functional R; surfaces direct negation collisions |
| Belief revision | Source-priority resolution with recency/count/confidence tiebreaks |
| Negative facts | `NOT_R` substrate; respected by induction and rule instantiation |
| Negation propagation | Lift `NOT_R` through `isa`/`partof` to siblings |

### Memory and meta

| Module | What it does |
|---|---|
| Query memory | Bounded FIFO log of episodes; drift detection per-call and aggregate |
| Calibration (per relation) | Predicted-vs-actual tally per relation; `record_truth(...)` feeds ground truth |
| Score calibration (v15.2) | Isotonic cosine→P(correct) map; powers the `calibrated_product` chain rule (§4.2, §5.3) |
| Chain cache | LRU, versioned; auto-invalidated on bulk writes / induction / rule emission |

### Multi-agent

| Module | What it does |
|---|---|
| Federated merge | Two agents' state folded: skill counters sum, provenance combines, HRR bundle sums |
| Consensus | `majority(agents, query)` aggregates by vote, confidence, or both |
| Diff | What does one agent know that another doesn't |

---

## 4. Key design decisions

This is the section that took the most experimental work. We describe the design decisions that came out of empirical findings and changed the system materially.

### 4.1 The six-gate filter stack on chain induction

Naïve chain induction - "if the chain walks confidently, store the shortcut" - fails approximately half the time on real knowledge bases, in characteristic ways. Some examples we observed on the commonsense KB:

- **Phantom shortcuts through hubs.** The chain `greatexpectations → author → dickens → wrote → olivertwist` walks with high confidence (both hops are direct facts after auto-symmetrization), but the induced shortcut `(greatexpectations, wrote, olivertwist)` is wrong. Dickens is the hub; the chain goes through him and emerges pointing at an unrelated work he wrote.
- **Same-relation HRR cleanup artifacts.** The chain `X → wrote → Y → wrote → Z` can fire on cleanup noise even when no true 2-hop wrote-wrote relationship exists.
- **Hub round-trips.** `caesar → country → rome → capitalof → italy` produces the technically-consistent but semantically-wrong shortcut `(caesar, capitalof, italy)`.
- **Degenerate cycles.** When HRR cleanup noise causes the final answer to be one of the intermediate nodes, the resulting fact is meaningless.
- **Silent type confusion (found in v15.1's audit).** A chain whose hops are individually valid can compose into a claim of the wrong *type*: `(shark, implies, blue)` via `[habitat, color]`, `(elephant, implies, tree)` via `[has, partof]`. The v15.0 policy stored these under a generic `implies` label; the v15.1 audit showed ~88% of "verified" inductions were such fallbacks, some encoding false claims.

We accumulated these failure modes by running induction and inspecting every wrong result. The recipe that emerged is six gates applied before commit:

1. **Confidence floor.** The chain must walk above a minimum propagated confidence at all.
2. **Inverse-pair filter.** Reject chains where consecutive relations form an inverse pair (`author/wrote`, `partof/haspart`, `capital/capitalof`, etc). These are round-trips through a hub.
3. **Non-transitive same-relation filter.** Reject same-relation chains unless the relation is in an explicitly-curated transitive set (`isa`, `partof`, `locatedin`, `ancestorof`, `hassubtype`, `haspart`, `contains`, `descendantof`).
4. **Intermediate-cycle filter.** Reject chains whose final answer equals an earlier node in the chain.
5. **No-meaningful-relation gate (v15.1).** If the chain's first hop is not a "lifting" relation (`isa`, `partof`, `locatedin`, `memberof`, `instanceof`, `kind`, `class`, `subclass`, `subtype`, `category`) and the chain is not same-relation transitive, no specific relation can be ascribed and the induction is rejected. Storing the chain under a generic `implies` fallback is opt-in (`InductionPolicy(allow_generic_implies=True)`) and off by default.
6. **Type-signature gate (v15.1).** Each lifting-relation family permits only a specific set of last-hop relations: `partof` transfers `{locatedin, madeof, usedfor}` (the part inherits the whole's location, material, use) but not `has` (a wing doesn't have what the bird has); `locatedin` transfers only aggregate-location relations `{continent, country, region, city}`; `isa`/`class`/`category` transfer `{isa, has, kind, class, category}`.

Quantified empirical results in §5.1.

**Filter generality.** The transitive, inverse-pair, lifting, and type-transfer sets are hand-curated. We expect all of them to be domain-specific: a medical KB will want different transitive relations (`subclass_of`, `caused_by`, `treats`) than a geographic one. Our claim is that the *structure* of the filter stack - reject hub round-trips, reject non-transitive same-relation chains, reject degenerate cycles, refuse to ascribe meaning when no type-coherent relation exists - generalizes. The specific relation lists are configuration, exposed on `InductionPolicy` so a domain user can extend them without forking the library. This is the same species of gating that KG rule mining uses (support/confidence thresholds in AMIE, type constraints in embedding models); what's specific to RCK is that several gates target HRR-substrate failure modes (cleanup crosstalk, hub round-trips) that symbolic stores don't have.

### 4.2 Confidence: from a bad default, to a display heuristic, to calibrated probabilities

**The bad default.** In our first version, chain confidence was the product of per-hop cosines. A clean retrieval at 0.7 cosine over 10 hops yields 0.7¹⁰ ≈ 0.028 - under the "uncertain" threshold. We treated this as a substrate capacity limit and stopped pursuing deeper reasoning. It isn't a substrate limit: the substrate walks 50-hop linear chains at 100% retrieval accuracy (§5.2). The product rule is just a bad model of what cosine similarity means in HRR - a 0.7 cosine is not a 0.7 probability.

**The display heuristic.** We changed the default propagation rule to the geometric mean of hop scores (with a mild per-hop decay). The argmax answer is unchanged; reported confidence stays readable for long, uniformly-strong chains - "moderate"-or-better to 30+ hops on the synthetic benchmark (§5.2). We keep it as the default because it's zero-configuration and monotone in the evidence. But we are explicit about what it is: a length-normalized score in the long tradition of non-collapsing truth-value combinators (fuzzy t-norms, PSL's Łukasiewicz logic, MLN's log-linear pooling, subjective-logic discounting, logarithmic opinion pools - see §6), chosen by curve-fitting, not derived from semantics. Its known weakness is length-*in*sensitivity: a 50-hop chain of 0.7s scores exactly what a 1-hop 0.7 scores. A reviewer who calls that a rescaling trick has a point.

**The calibrated rule (v15.2).** The principled fix is to stop treating cosines as probabilities and measure what they're actually worth. We fit an isotonic regression (pool-adjacent-violators, with Beta-smoothed block probabilities so the calibrator never claims certainty) on 4,862 labeled retrievals spanning clean to saturated bundle loads: cosine in, empirical P(top-1 correct) out. With calibrated per-hop probabilities, plain multiplication becomes the right rule again:

```
chain_confidence = product_i  P_hat(correct | cosine_i)  ~  P(every hop correct)
```

under an independence assumption. This is length-sensitive - a 50-hop chain *is* less certain than a 2-hop chain - without collapsing on clean hops, because a clean hop calibrates to ~0.9997, not 0.7. Available as `PropagationConfig(rule="calibrated_product", calibrator=...)`; fitting is one script (`scripts/confidence_calibration_study.py`). Empirical results, including held-out Brier scores and the full depth table, in §5.3.

### 4.3 Provenance as a graph, not a tag

Early versions of RCK stored provenance as flat metadata: a source and a timestamp. This was insufficient for derived facts. When we extended the system to chain induction and rule instantiation, each derived fact began carrying a `derivation` field containing the actual chain of source facts that produced it. The `explain_why` routine walks this graph recursively.

The result is that any fact in the KB has a traceable path back to the user-asserted facts that grounded it. This is the structural property that makes RCK auditable in a way that LLM outputs aren't. Mechanically it is a justification structure in the truth-maintenance-system tradition (Doyle 1979; de Kleer 1986) rebuilt over a vector substrate - the vector memory itself has no derivation semantics, so the provenance graph is the symbolic shadow that restores them (§6).

### 4.4 Bayesian softmax for analogy

The original analogy solver picked the relation with the highest score, applied it to C, and returned the result. We observed that the top-1 relation often wasn't the relation that produced the best answer on C. Switching to a joint score over (relation, answer) pairs, normalized via softmax with a tunable temperature, substantially improves both the chosen answer's accuracy and the calibration of the reported confidence.

### 4.5 Shard-local cleanup (v15.3)

Through v15.2, every retrieval cleaned up against the entire codebook - one matmul over all symbols the agent had ever seen. That is O(vocabulary) work per query and a vocabulary-sized matrix in memory, which is what actually capped the benchmarks at a few thousand facts.

The fix follows from an invariant the substrate already guarantees: a shard's bundle is the sum of that shard's own fact vectors, so the only signal that can be unbound from it is a symbol stored in that shard, in that role. Cleanup therefore only needs to consider the shard's own unknown-role symbols - a few dozen candidates instead of the whole vocabulary. Candidates are collected fresh from the shard's fact log on every query, so there is no cache to go stale under store, forget, federated merge, or session load.

Three consequences, all measured. Per-query cost becomes independent of vocabulary size (§5.6: 0.33 ms median at 100,000 facts). Cross-shard false positives strictly decrease - one of our own regression tests had encoded a phantom forward edge that global cleanup surfaced from bundle crosstalk; under local cleanup the search finds the real path instead, at higher confidence. And there is a small constant cost: at small vocabularies the per-query candidate collection is slightly slower than the old cached global matmul (§5.5's table vs. its v15.2 predecessor). `cleanup="global"` remains available on every query path for exact pre-v15.3 behavior.

---

## 5. Empirical findings

All numbers come from scripts in `scripts/`, run on a commodity laptop (single CPU thread, Python 3.11+ - measured on 3.14, D=4096, default shard sizing). Data files in `data/` are committed to the repository, so every table below has both a regeneration path and a canonical artifact. Absolute latencies are machine-dependent; machine-independent counters (query counts, accuracy, yields) are reported alongside.

### 5.0 Baselines: what the substrate costs

Through v15.3 every study in this section measured RCK against RCK. That left the first question any reader asks unanswered: *why not just use a hash map, or a graph library?* This section answers it, on identical data with identical protocols. Reproduction: `python scripts/baseline_study.py` -> `data/baseline_study.json`.

**Storage and retrieval** (the axis of section 5.6), on the same committed ConceptNet subset, 1,000 sampled recall probes with valid-object-set labeling, 300 never-stored IDK probes:

| Facts | System | Ingest | RSS | recall@1 | Query median |
|---:|---|---:|---:|---:|---:|
| 10,000 | dict | **0.002 s** | **1.0 MB** | 1.0000 | **0.0003 ms** |
| 10,000 | RCK | 2.204 s | 164.8 MB | 1.0000 | 0.2929 ms |
| 30,000 | dict | **0.012 s** | **3.4 MB** | 1.0000 | **0.0003 ms** |
| 30,000 | RCK | 5.055 s | 365.7 MB | 1.0000 | 0.2774 ms |
| 100,000 | dict | **0.036 s** | **10.2 MB** | 1.0000 | **0.0004 ms** |
| 100,000 | RCK | 10.804 s | 769.0 MB | 1.0000 | 0.4849 ms |

At 100,000 facts a six-line `dict[(S, R)] -> [O]` index ingests ~300x faster, uses ~75x less memory, answers ~1,200x faster, and matches recall exactly. **On this axis the HRR substrate is strictly dominated.** Section 5.6's headline - 100,000 facts at 100.0% recall and sub-millisecond queries - should therefore be read as evidence that *sharding keeps the substrate viable at scale*, not as a competitive result. It is a substrate-validity finding. We had presented it as more than that.

**Two-hop chain discovery** (the axis of section 5.5), 30 probes whose two-hop path provably exists, on 10,000 facts:

| System | Build | RSS | Discovery rate | Median |
|---|---:|---:|---:|---:|
| networkx | **0.015 s** | **5.1 MB** | **1.0000** | **0.0050 ms** |
| RCK | 2.176 s | 170.8 MB | 0.7333 | 14.0021 ms |

Here RCK loses on completeness as well as cost: it finds 73% of chains that provably exist, against 100%.

**Reading these tables fairly.** Three caveats, all of which cut against over-reading the comparison:

1. *The chain task is not identical.* `networkx.shortest_path` finds **any** path through the graph; `agent.discover` finds a **typed relation chain** that the induction gates can then act on. networkx is solving a strictly easier problem, so its 100% is not directly RCK's 73% on the same task. The latency and memory gaps are real regardless.
2. *The dict's recall@1 is 100% by construction.* An exact index cannot lose information, so this is a floor, not an achievement. It is exactly the point: the substrate pays memory and latency for an approximation that, at these scales, buys no retrieval accuracy.
3. *Neither comparator does the work the rest of this paper is about.* No provenance graph, no calibrated confidence, no IDK state, no chain induction, no negative facts, no contradiction resolution, no federated merge. A dict is not a competing system; it is a measurement of what the substrate costs.

**What this does and does not imply.** It does not imply RCK is pointless - the reasoning and auditability layer above the substrate is the contribution, and no baseline here replicates it. It does imply that the substrate is currently a **cost centre rather than a differentiator**, and that two properties which could justify it - federated merge without entity alignment (section 8.4), and analogy as native vector algebra (section 5.7) - have never been benchmarked against a non-VSA alternative. Until they are, the honest claim is that RCK's reasoning layer would likely run on a plain index, and we have not shown otherwise.

*Configuration note:* RCK is provisioned here as `expected_facts = 2 x |facts|` because `tell()` symmetrizes inverse relations, so the stored fact count is roughly double the input. That yields a higher shard count, and therefore higher RSS, than section 5.6's `scale_study.py` path (769 MB vs 535 MB at 100,000). Both are real measurements of different configurations; neither supersedes the other.

### 5.1 Chain-induction precision

We ran the chain-induction diagnostic against a combined knowledge base assembled from all four bundled KBs (commonsense, extended, ultra, massive) - 5,991 raw triples that expand to 7,080 stored facts after inverse-relation symmetrization, across 256 auto-sized shards. We generated 400 two-hop transitive probes. Reproduction: `python scripts/chain_induction_failure_analysis.py`.

**An earlier framing was misleading - and is retracted.** v15.0 reported "~87% precision" (328 committed of 376 attempts, Wilson 95% CI [83.5%, 90.2%]). A failure-mode audit showed ~88% of those "committed" facts were not specific claims at all - they were `implies`-relation fallbacks emitted whenever the chain didn't qualify for typed relation forwarding. Some were correct-but-mislabeled (`(owl, implies, beak)` via `[class, has]` should be `(owl, has, beak)`). Others were silent type violations: `(shark, implies, blue)` via `[habitat, color]`; `(elephant, implies, tree)` via `[has, partof]`. The "87%" figure measured *HRR-roundtrip stability*, not semantic correctness. We retracted it and added gates 5 and 6 (§4.1).

**Numbers under the corrected six-gate pipeline** (fresh run, same 400-probe protocol):

| Stage | Count | % of attempts |
|---|---:|---:|
| Probes generated | 400 | - |
| Chains discovered + walked | 381 | - |
| Induction attempts | 370 | 100.0% |
| Rejected: confidence floor | 26 | 7.0% |
| Rejected: inverse-pair gate | 30 | 8.1% |
| Rejected: same-relation non-transitive | 0 | 0.0% |
| Rejected: degenerate cycle | 1 | 0.3% |
| Rejected: type-signature mismatch | 12 | 3.2% |
| Rejected: no meaningful relation | 270 | 73.0% |
| **Verified and stored** | **31** | **8.4%** |
| Self-verify (HRR roundtrip) failures | 0 | 0.0% |

The headline result, reframed honestly: **31/31 of inductions that survive all six gates pass HRR self-verification, and manual inspection of all 31 finds each to be a semantically valid new direct fact** (e.g. `(leaf, locatedin, forest)`, `(engine, usedfor, driving)`, `(crow, has, beak)`, `(paris, continent, europe)`). The 12 type-signature rejections are exactly the chains that produced false claims under the v15.0 fallback policy. Full per-record output: `data/chain_induction_failures.json`.

**What the 8.4% yield means.** Of every 100 two-hop chains the substrate walks, roughly 8 produce a meaningful new direct fact; the other 92 are chains whose relation composition is not type-coherent and should not be stored under any label. That's a property of commonsense knowledge graphs, not a substrate weakness: the substrate is right to walk them, and the symbolic layer is right to refuse to ascribe meaning to most of them. We report "31/31 manually validated under the v15.1 pipeline" rather than "100% precision" - the latter is not what a 31-sample empirical claim supports. A larger external benchmark (e.g. a 10,000-probe Wikidata sample) is future work.

### 5.2 Chain depth at varying propagation rules

Reported confidence as a function of chain depth on a synthetic 50-node linear chain (relation `next`). **Retrieval accuracy is 100% at every depth for every rule** - the substrate walks the chain correctly; only reported confidence differs. `calibrated_product` uses the calibrator fitted in §5.3.

| Depth | `product` (legacy) | `min` | `geometric_mean` (default) | `calibrated_product` (v15.2) |
|---:|---:|---:|---:|---:|
| 1 | 0.711 (strong) | 0.711 (strong) | 0.711 (strong) | 1.000 (strong) |
| 5 | 0.204 (moderate) | 0.571 (strong) | 0.618 (strong) | 0.998 (strong) |
| 10 | 0.056 (weak) | 0.442 (strong) | 0.495 (strong) | 0.997 (strong) |
| 15 | 0.011 (uncertain) | 0.236 (moderate) | 0.378 (strong) | 0.995 (strong) |
| 20 | 0.010 (uncertain) | 0.183 (moderate) | 0.296 (moderate) | 0.994 (strong) |
| 30 | 0.010 (uncertain) | 0.109 (moderate) | 0.172 (moderate) | 0.991 (strong) |
| 50 | 0.010 (uncertain) | 0.039 (weak) | 0.061 (weak) | 0.985 (strong) |

Thresholds: strong ≥ 0.30, moderate ≥ 0.10, weak ≥ 0.03, uncertain < 0.03.

Read the table with the right metric for each claim. *Substrate capability:* argmax answers are correct out to 50 hops under every rule. *Geometric mean:* moderate-or-better to 30+ hops (not 50 - at depth 50 it reports 0.061, "weak"). *Calibrated product:* because these clean hops each calibrate to ~0.9997, the honest probability that all 50 are correct is ~0.985 - and unlike the geometric mean, the number genuinely decreases with every hop added. The 1.000 at depth 1 is display rounding of 0.9997.

### 5.3 Calibrated confidence: what a cosine is worth

Protocol (`scripts/confidence_calibration_study.py`): we sample labeled retrievals from synthetic KBs at five bundle loads spanning clean to heavily saturated (500 → 12,000 facts at 64 shards; per-query label = top-1 answer in the valid-object set, plus never-stored probes labeled false), fit a PAV isotonic calibrator with Beta(1,1)-smoothed block probabilities on the pooled 4,862 samples, and evaluate on a held-out set the calibrator never saw: the real commonsense KB (705 samples).

Held-out results:

| Metric | Raw cosine as probability | Calibrated |
|---|---:|---:|
| Brier score | 0.5681 | **0.0038** |

The raw-cosine column quantifies §4.2's claim: cosines are drastically miscalibrated as probabilities (correct answers cluster near 0.10-0.24 cosine on this KB and are essentially always right; treating 0.15 as "15% probable" is off by ~85 points). The calibrated map recovers this almost perfectly on held-out data: everything above the noise floor is ~certain, everything at the noise floor is ~never right, and the transition band (empirical accuracy 88.7% in the reliability table's third bin) is where the calibrated probabilities do real work.

Two design details matter. *Smoothing:* each isotonic block reports (successes + 1)/(n + 2) - the Beta(1,1) posterior mean - so no block ever claims probability exactly 1.0, which is what keeps 50-hop calibrated products below certainty (~0.985 at depth 50, §5.2) instead of saturating at 1.0. *Independence caveat:* the product reads as P(all hops correct) only if hop errors are independent; correlated failures (e.g. one saturated shard serving several hops) would make it optimistic. We state this rather than hide it - it's the same assumption PRA-style path probabilities make.

The fitted calibrator ships in `data/confidence_calibration_study.json` and loads via `ScoreCalibrator.from_dict(...)`.

### 5.4 Sparse vs. dense substrate capacity

We expected sparse-binary HRR to be a substrate win: dense bipolar HRR uses one byte per dimension while sparse representations use a few bytes per atom regardless of D. Sparse atoms are 6-13× smaller than dense atoms at typical parameters.

The substrate-level win does not survive bundling. Per-shard recall cliff as a function of stored fact count for dense (D=4096, 8192, 16384) and sparse (D=4096...16384, k=80...320) substrates: at equal D, sparse substrates have **8-16× lower per-shard capacity**. To compensate by sharding we would need 8-16× more shards, which exceeds the per-atom memory savings.

**The honest conclusion**: sparse HRR is not a drop-in replacement for dense HRR in this use case. It remains useful for similarity-only caches and large-vocabulary cleanup where bundling isn't required.

### 5.5 Chain discovery latency, and the live-relation index

The v15.0 revision of this section reported a three-KB latency table and diagnosed that BFS cost is dominated by relation count (one HRR query per relation per frontier), proposing a per-shard live-relation set as future work. Two things have changed. First, the mid-size KBs from that table no longer exist in the repository, so the old numbers (14.5/7.7/55.5 ms) were not reproducible from the shipped artifacts - the current study sweeps the four KB tiers that *are* shipped, and its data file is committed. Second, we implemented the proposed index (v15.2): a `RelationIndex` snapshot built from the shards' fact logs in the same O(total facts) scan discovery already needed, letting the search skip any (node, relation) whose routed shard holds no fact with that relation, and restrict reverse fan-outs to shards where the relation lives. A skipped query could only ever have returned bundle crosstalk, so real chains are unaffected - measured discovery results are identical with and without the index at every tier.

30 two-hop transitive probes per tier; per-probe median of 3 repeats; warm-started; single thread:

| KB tier | Facts | Relations | Shards | Discovery rate | Avg ms (indexed) | HRR queries cut | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| commonsense | 716 | 21 | 16 | 97% | 20.9 | 31% | 1.4× |
| +extended | 1,495 | 33 | 32 | 90% | 20.7 | 51% | 2.0× |
| +ultra | 3,666 | 49 | 64 | 90% | 268.1 | 60% | 2.3× |
| +massive | 7,080 | 76 | 128 | 73% | 561.4 | 66% | 2.8× |

Four honest readings. *The diagnosis was right:* the query cut and the speedup both grow with relation count (31% → 66%, 1.4× → 2.8×), confirming relation-count scaling dominates - and the 100,000-fact ConceptNet KB of §5.6, with only 19 relations, discovers at a median of 30 ms, faster than the 76-relation 7,080-fact tier here. *Absolute latency is machine- and load-dependent:* wall-clock on this benchmark varies up to ~4× run-to-run under OS scheduling noise, which is why the study reports per-probe medians and the machine-independent query counts alongside. *Shard-local cleanup trades a constant for scalability:* per-query cost gained a small constant overhead (candidate collection from the fact log) that global cached cleanup doesn't pay at small vocabularies - visible in this table versus the v15.2 numbers - in exchange for per-query cost that no longer grows with vocabulary at all (§5.6). *Discovery rate degrades on relation-heavy KBs:* 97% on the smallest tier down to 73% at 7,080 facts / 76 relations with beam width 3 and depth ≤ 4 - deeper beams recover some of this at proportional cost. We report the degradation rather than tuning the benchmark around it.

### 5.6 Scale: 100,000 real-world facts on a laptop

Everything above runs on KBs of a few thousand facts, and "tested to ~7,000 facts" was this paper's honest ceiling through v15.2. This study measures the substrate at 10×-100× that, on real-world knowledge: the first 100,000 unique English ConceptNet 5.7 assertions at min-weight 2.0 (19 relations, 81,086 entities). The exact subset ships in the repository (`data/conceptnet_scale_100k.jsonl`), so the study reproduces offline; regenerating the subset from the public ConceptNet download is two scripted steps (`scripts/import_conceptnet.py`, then `scripts/scale_study.py --source ...`).

The enabler is shard-local cleanup (§4.5): per-query cost depends on shard fill, not vocabulary, so queries cost the same at 100,000 facts as at 700. (Global cleanup at this scale would also mean materializing a ~1.3 GB codebook matrix; the local path never builds it.)

Per tier - 1,000 sampled stored facts for recall (valid-object-set labeling; ConceptNet is heavily multi-valued), 300 never-stored probes for IDK safety, 30 two-hop probes for discovery, single process, D=4096, auto-sharded:

| Facts | Shards | Ingest | RSS | recall@1 | recall@3 | Query median | Query p95 | Discovery |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 256 | 0.7 s | 135 MB | 99.9% | 100.0% | 0.22 ms | 0.81 ms | 83% @ 5 ms |
| 30,000 | 512 | 2.4 s | 253 MB | 100.0% | 100.0% | 0.35 ms | 0.94 ms | 90% @ 18 ms |
| 100,000 | 2,048 | 6.6 s | 535 MB | 100.0% | 100.0% | 0.33 ms | 1.03 ms | 97% @ 30 ms |

Five readings. *Recall holds at 100%* because auto-sharding keeps per-shard fill (~49 at the top tier) under the measured ~80-fact cliff (§5.4's capacity study) - the sharding thesis verified at 100,000 real facts rather than extrapolated. *Query latency is flat* (0.2-0.35 ms median) across a 10× size range: vocabulary-independence in practice, on a single CPU thread with no index structures beyond the shards themselves. *"I don't know" survives scale:* never-stored probes score at most ~0.05 at p95 while stored facts sit above ~0.10 at p5, at every tier - the IDK margin doesn't erode as the KB grows. *Discovery gets better, not worse:* 83% → 97% as the graph densifies (more alternative two-hop paths), at 5-30 ms medians - consistent with §5.5's finding that relation count (19 here), not fact count, dominates BFS cost. *Ingest is O(1) per fact in practice:* ~13,000-15,000 facts/second, so the full 100k KB builds in under seven seconds.

The honest bound: 100,000 facts is the largest configuration we have benchmarked. Millions should follow the same shard arithmetic, but that remains unverified - the claim stops where the measurements stop.

### 5.7 Analogical reasoning accuracy

On a 115-probe analogy benchmark drawn from the commonsense KB, the confidence-weighted Bayesian solver achieves:

- **92.2%** (106/115) relation-inference accuracy
- **93.9%** (108/115) final-answer accuracy

The 88.7% baseline quoted in the abstract is the earlier argmax-only solver (102/115 on the same probes, as reported in the v15.0 study); the shipped script benchmarks the current solver only, so the baseline is a historical reference point, not a number this revision re-measures. Most remaining failures come from multi-valued relations (e.g., a dog has fur, legs, tail, whiskers - a 5-way ambiguous analogy is essentially a guess) and aren't true errors.

**These numbers are worse than a twenty-line symbolic solver.** Section 5.10 measures a non-VSA baseline on the same construction: exact indices plus a loop score 100.0% relation inference and 97.0% exact answers against RCK's 89.0% and 95.0%, at roughly 2,800x lower latency. The right reading of this section is therefore *"vector-algebra analogy works"*, not *"vector-algebra analogy is a reason to use a vector substrate"*.

### 5.8 Cross-shard distribution

Chains across the commonsense KB visit approximately 2 distinct shards on average, and 95-98% of two-hop chains have endpoints on different shards at realistic shard counts (n_shards ≥ 64). This confirms that the sharded design genuinely distributes reasoning load.

### 5.9 Cascading induction

Iterating chain induction to a fixed point on the commonsense KB produces **6 new verified facts in 4 rounds** (2 + 2 + 2 + 0, saturating at round 4; dominant pattern `locatedin → continent`), growing the KB 716 → 722. An earlier draft of this section reported "~11 facts, with round 2 out-producing round 1"; the committed study data supports neither detail, so we report the reproducible numbers. Rule-based cascading is dramatically more productive because rules are reusable: in the full-stack demo, extracting rules from the skill library and applying them forward grows the same KB 718 → 843 (+125) (`examples/v14_full_stack_demo.py`). A dedicated extraction study against the post-cascade 722-fact KB yields 21 rules at min-support 2, min-confidence 0.5 (`scripts/rule_extraction_study.py` → `data/rule_extraction_study.json`; two noise-supported rules from the v15.2 run no longer form under shard-local cleanup).

### 5.10 Does the substrate earn its place?

Section 5.0 showed the substrate is dominated on storage, retrieval and chain discovery. It named two properties as the remaining candidates that could still justify it: **federated merge without entity alignment** (section 8.4) and **analogy as native vector algebra** (section 5.7). Neither had ever been compared against a non-VSA alternative. This section does that. Reproduction: `python scripts/substrate_justification_study.py` -> `data/substrate_justification_study.json`.

The baselines are deliberately plain - a few dict indices and about twenty lines of set logic - because the question is not whether something can beat RCK, but whether the vector substrate buys anything a trivial symbolic implementation does not already have.

**Analogy** (`a:b::c:?`, 100 probes, construction identical to `scripts/analogy_study.py`, 486-triple commonsense KB):

| Solver | Relation inference | Exact answer | Valid-set answer | Per probe |
|---|---:|---:|---:|---:|
| symbolic (dict + set logic) | **1.0000** | **0.9700** | **1.0000** | **0.001 ms** |
| RCK (HRR vector algebra) | 0.8900 | 0.9500 | 0.9600 | 2.788 ms |

The symbolic solver wins every accuracy column and is roughly 2,800x faster.

**Federated merge** (486 triples split between two parties, 200 probes):

| Scenario | dict merge | RCK bundle-sum |
|---|---:|---:|
| Shared identifiers - post-merge recall | **1.0000** (0.11 ms) | 0.9700 (5.10 ms) |
| Divergent identifiers - post-merge recall | **1.0000** | 0.9750 |
| Divergent identifiers - cross-name resolution | 0.0000 | 0.0100 |

The third row is the one that matters. Merging is trivially easy when both parties already use the same identifiers, and that is the only case the first two rows exercise - a dict does it exactly and faster. When identifiers diverge, neither system resolves them, and RCK's 1.0% is bundle crosstalk rather than alignment: a false positive, which is strictly worse than an honest zero.

**The caveat that cuts the other way - and the study that resolved it.** The analogy protocol above, inherited from our own earlier study, constructs every probe so that `(a, R, b)` is *already stored*. Exact indexing is therefore sufficient **by design**, and the benchmark structurally favours it. A protocol where the relation must be *generalised* rather than looked up is exactly where vector algebra should have the advantage, so we built one (`scripts/analogy_generalization_study.py` -> `data/analogy_generalization_study.json`): identical probes, but each probe's `(a, R, b)` edge is **held out of the KB**, so the relation must be inferred from the rest of the graph. Both systems get the same held-out KB.

| Solver | Relation inference | Answer | Per probe |
|---|---:|---:|---:|
| majority-relation (floor) | 0.0600 | 0.0400 | - |
| type-based symbolic | **0.6500** | **0.6300** | 0.175 ms |
| RCK (HRR vector algebra) | **0.0000** | 0.1400 | 59.811 ms |

The regime designed to favour the substrate is where it performs worst. RCK infers the correct relation in **zero** of 100 held-out probes - below the majority-guess floor - while a type-fit symbolic baseline reaches 65%, roughly 340x faster. RCK's answer accuracy (14%) does clear the 4% floor, so it is not pure noise, but the relation column is unambiguous: the vector algebra does not generalise a held-out edge, it retrieves a stored one.

A fair reading of the implementation: `solve_analogy` queries the KB for `(a, ?, b)` and has no fallback when that edge is absent, so it is arguably not *designed* for held-out generalisation. We accept that - and it is precisely the finding. "Analogy as native vector algebra" implies a generalisation capability the implementation does not have; what it actually provides is lossy retrieval of a stored edge.

**Conclusion.** Across five axes now measured against non-VSA baselines - ingest, memory, query latency, chain discovery, analogy, and federated merge - the HRR substrate does not win on any of them, and on two it is beaten by roughly twenty lines of dictionary code. The contribution of this work is the layer *above* the substrate: the provenance graph, calibrated confidence, IDK as a first-class state, the six-gate induction filter, negative facts, contradiction resolution, and replayable decision records. None of those require holographic reduced representations. A faithful reimplementation on an exact index would, on present evidence, be smaller, faster, and equally auditable. We think that is worth stating plainly in the paper that proposed the substrate.

### 5.11 Reproducibility

All numbers in §5 are reproducible from the public repository, and the canonical output of every study is committed under `data/`:

| Result | Script | Data file |
|---|---|---|
| §5.0 baselines (dict, networkx) | `scripts/baseline_study.py` | `data/baseline_study.json` |
| §5.10 substrate justification | `scripts/substrate_justification_study.py` | `data/substrate_justification_study.json` |
| §5.1 induction precision | `scripts/chain_induction_failure_analysis.py` | `data/chain_induction_failures.json` |
| §5.2 chain depth | `scripts/chain_depth_study.py` (+ calibrated column from the §5.3 script) | `data/chain_depth_study.json` |
| §5.3 confidence calibration | `scripts/confidence_calibration_study.py` | `data/confidence_calibration_study.json` |
| §5.4 sparse vs dense capacity | `scripts/sparse_capacity_study.py`, `scripts/run_capacity_study.py` | `data/sparse_capacity_study.json`, `data/capacity_study.json` |
| §5.5 discovery latency + index | `scripts/chain_discovery_study.py` | `data/chain_discovery_study.json` |
| §5.6 scale (ConceptNet 100k) | `scripts/scale_study.py` (subset: `scripts/import_conceptnet.py`) | `data/scale_study.json`, `data/conceptnet_scale_100k.jsonl` |
| §5.7 analogy accuracy | `scripts/analogy_study.py` | `data/analogy_study.json` |
| §5.8 cross-shard distribution | `scripts/cross_shard_chain_study.py` | `data/cross_shard_chain_study.json` |
| §5.9 cascade induction | `scripts/cascade_induction_study.py`; rule extraction via `scripts/rule_extraction_study.py`; rule cascade via `examples/v14_full_stack_demo.py` | `data/cascade_induction_study.json`, `data/rule_extraction_study.json` |

All scripts run from the repository root with no external services and no GPU. Environment: Python 3.11+ (numbers in this revision measured on CPython 3.14), single CPU thread, default agent settings (D=4096, auto-sharded). Random seeds are fixed (`seed=0` throughout), so accuracy-type numbers reproduce exactly; latency-type numbers vary by machine. The test suite (`pytest -q`) is **757/757** passing on the same environment. The v15.3.1 tag on GitHub marks the exact source state of this revision.

---

## 6. Comparison with related work

**LLMs.** Modern LLMs operate by predicting the next token given context. They lack a mechanism for distinguishing knowledge from confabulation: every output is generated, including "explanations" for previous outputs. Retrieval-augmented generation mitigates but does not eliminate the problem, because the generator can still hallucinate around retrieved facts; the graph-grounding survey literature consistently reaches "reduce, not eliminate" (Agrawal et al. 2024). RCK is structurally different: it retrieves discrete facts and reasons over them; the optional polisher only renders surface form.

**HRR-based cognitive architectures - the closest prior art.** Eliasmith's Semantic Pointer Architecture and the Spaun model (Eliasmith et al. 2012; Eliasmith 2013) built integrated cognition on exactly the substrate RCK uses - Plate's HRRs - a decade earlier, and Crawford, Gingerich & Eliasmith (2016) encoded the entire WordNet lexical KB (100k+ terms) as HRR structures, so RCK is neither the first integrated HRR reasoning system nor the largest HRR knowledge store. Holographic Declarative Memory (Kelly et al. 2020) put an HRR memory inside ACT-R. RCK differs in goal and shape: those systems model cognition (often in spiking neurons, with psychological plausibility as the criterion); RCK is a knowledge-engineering artifact - explicit editable triples, derivation-graph provenance, contradiction handling, negative facts, and a test suite - optimized for auditability rather than biological plausibility. We know of no prior system in this lineage that ships that combination.

**Truth maintenance, belief revision, and explanation.** RCK's provenance graph, `explain_why`, and conflict resolution are, mechanically, a justification-based TMS (Doyle 1979) with contradiction handling in the spirit of de Kleer's ATMS (1986), belief revision in the AGM tradition (Alchourrón, Gärdenfors & Makinson 1985), and WHY-style explanation going back to MYCIN (Buchanan & Shortliffe 1984). Cyc has carried per-assertion provenance and natural-language justification for decades (Lenat & Marcus 2023 make the Cyc-vs-LLM trustworthiness argument directly). RCK's contribution here is not the mechanism but the substrate integration: the HRR memory itself has no derivation semantics, so the provenance graph is the symbolic side-channel that restores auditability over a lossy vector store - and it survives federated bundle merges.

**KG completion, rule mining, and path ranking.** Promoting multi-hop paths to new edges behind confidence and structural gates is the core move of AMIE's rule mining (Galárraga et al. 2013), the Path Ranking Algorithm (Lao & Cohen 2010; Lao, Mitchell & Cohen 2011), NELL's coupled learning with a confidence-assigning knowledge integrator (Carlson et al. 2010), and type-constrained KG completion (Krompaß et al. 2015). RCK's six-gate stack belongs to this family. What's different: the gates were derived from HRR-specific failure modes (cleanup crosstalk, hub round-trips through auto-symmetrized inverses) that symbolic and embedding stores don't exhibit, and the validation methodology - 400 probes, manual inspection of every committed fact, and a published retraction of the earlier metric - is, as far as we know, unusual in its explicitness at this scale.

**Negative knowledge.** Storing explicit negatives so a system can refute rather than merely fail to find is an active research program (Arnaout, Razniewski & Weikum 2020; NegatER - Safavi et al. 2021), with formal roots in the classical-negation vs. negation-as-failure distinction (Clark 1978; Gelfond & Lifschitz 1991). RCK does not claim the idea; its contribution is carrying asserted negatives (`NOT_R`) inside a VSA substrate, propagating them through taxonomic relations, and consulting them as a hard gate in induction and rule instantiation.

**Confidence propagation.** Non-collapsing combination rules for chained uncertain evidence have deep prior art: NARS's truth-value algebra (Wang 2006), Łukasiewicz t-norms in probabilistic soft logic (Bach et al. 2017), Markov logic's log-linear pooling (Richardson & Domingos 2006), subjective logic's trust discounting (Jøsang 2016), and the logarithmic opinion pool - a weighted geometric mean - studied since the 1980s (Genest & Zidek 1986). RCK's geometric mean is a member of that family chosen empirically, and we make no theoretical claim for it. The calibration result (§5.3) is the part we haven't found precedent for on this substrate: measuring the cosine→accuracy curve of HRR cleanup under varying bundle load and using it to restore multiplicative path semantics, in the spirit of PRA's path probabilities but with the probabilities measured rather than assumed.

**Vector-symbolic architectures.** VSAs (Kanerva 2009; Plate 1995) provide the primitives; torchhd (Heddes et al. 2023) is the reference library; IBM's neuro-vector-symbolic architecture solves Raven's matrices with VSA algebra over neural perception (Hersche et al. 2023); HolE brought Plate's circular correlation to KG link prediction as a *learned* embedding model (Nickel et al. 2016). RCK differs from the embedding line in that nothing is learned - facts are stored and retrieved algebraically, so a fact can be added, edited, or deleted in O(1) without retraining - and from the perception line in that its scope is knowledge management, not visual abstraction. Among open implementations we know of, RCK is the most complete *symbolic KB-reasoning stack* (provenance, revision, negation, induction, federation) on a VSA substrate; we'd welcome corrections to that claim.

**Federated and multi-party HD computing.** Aggregating hyperdimensional representations across parties by vector addition is established in federated HDC (e.g. FedHDC - Zeulin et al. 2023; secure HD computation offload in SecureHD - Imani et al. 2019), and KG federation via embedding alignment has its own literature. RCK's merge is the same additive-bundling pattern; what it adds is that agents share a deterministically name-hashed codebook, so merging needs no entity-alignment or embedding-reconciliation step, and per-fact provenance (source tags, derivations) survives the sum - the merged agent can still cite which contributor asserted what.

**NARS.** NARS (Wang 2006) shares with RCK the commitment to non-axiomatic, evidence-based reasoning with explicit confidence, with a more principled truth-value algebra and a far more developed inference rule set. RCK trades that theoretical maturity for HRR-substrate cheapness and empirically-derived gates. Cross-pollination is an open direction we'd welcome - NARS-style truth values on RCK's substrate, or RCK-style filter empirics in NARS's engine.

**Knowledge graphs and graph databases.** Graph databases (Neo4j, RDF stores) provide discrete, queryable, editable triples but no reasoning beyond what the query language exposes. RCK adds the derivation pipeline (chain induction, rule extraction, cascade) and the provenance graph.

An earlier revision continued: *"RCK adds the HRR substrate (cheap fuzzy retrieval) ... a graph database is roughly RCK without the substrate or the reasoning layer."* We withdraw the first half. Section 5.0 measures it: against a plain index the substrate is ~300x slower to build, ~75x larger, and ~1,200x slower per query at 100,000 facts, at identical recall - and against `networkx` it discovers fewer of the chains that provably exist. "Cheap fuzzy retrieval" was an assertion we had never tested, and on these measurements it is not cheap and its fuzziness buys no accuracy. The honest statement is narrower: **RCK's contribution is the reasoning and auditability layer, and that layer does not currently require the HRR substrate.** Whether the substrate earns its place rests on properties we have not yet benchmarked against a non-VSA alternative - federated merge without entity alignment, and analogy as native vector algebra.

**Neuro-symbolic systems.** DeepProbLog (Manhaeve et al. 2018) embeds neural perception into logical programs; the logic is expressive but the system is large and Prolog-centric. RCK takes the opposite direction: embed symbolic knowledge management into a neural-shaped substrate that's small and fast.

---

## 7. Limitations

**Surface fluency.** The optional polisher is trained on a synthetic paraphrase corpus and is small. Its outputs are grammatical but mildly stilted. RCK isn't designed for creative writing or open-ended dialogue and doesn't compete with LLMs on those tasks.

**Ingestion bottleneck.** To populate the KB from raw text we use a rule-based Open IE extractor. It works for clean text and fails on conversational or ambiguous text. Scaling RCK to Wikipedia-grade knowledge bases requires a better triple extractor, possibly itself an LLM run as a one-time ingestion pass - which would reintroduce a generative component at ingestion time, with exactly the caveats §1.2 describes.

**Capacity at scale.** The largest configuration benchmarked in this paper is 100,000 real ConceptNet facts at 100.0% recall@1 and sub-millisecond queries (§5.6). The per-shard recall cliff (~80 facts/shard at D=4096) is measured (§5.4, `data/capacity_study.json`), and auto-sharding keeps fill under it. Millions of facts should follow the same shard arithmetic - more shards, same per-shard physics - but have not been benchmarked; the claim stops where the measurements stop.

**Discovery completeness on relation-heavy KBs.** BFS discovery hit-rate degrades from 97% (716 facts, 21 relations) to 73% (7,080 facts, 76 relations) at the default beam width and depth (§5.5); on the 19-relation ConceptNet KB it holds 97% at 100,000 facts (§5.6). Relation count, not fact count, is the pressure. Wider beams recover coverage at linear cost; we haven't yet characterized the frontier.

**Confidence semantics.** The calibrated product (§5.3) assumes hop independence; correlated hop errors make it optimistic. The default geometric mean is a heuristic and length-insensitive. Calibration is fitted per substrate configuration (D, shard sizing) - changing configuration means refitting.

**Open-vocabulary relations.** RCK uses whatever relation names you give it. Two different relation names that mean the same thing (`author` vs `wrote_by`) are treated as unrelated unless declared as synonyms.

**No global plan or meta-reasoning.** RCK does not have a planner that decides which sub-question to answer next given a complex goal. The agent answers what you ask. Composing complex behaviors requires the caller to compose the individual queries.

---

## 8. What this opens up

The properties RCK has - auditability, editability, treatment of uncertainty as a first-class state, cheap operation - suggest several directions. We expand the first because it's the most immediately actionable and the one where current LLM tooling has the most obvious gap.

### 8.1 A memory layer for LLM agents

Current LLM agents (whether built on OpenAI, Anthropic, or open-source models) face a structural problem: the model doesn't reliably remember anything you've told it, and when it does, it can't distinguish between something it learned during training and something the user told it in conversation. The standard mitigations - RAG over a vector store, fine-tuning, ever-larger context windows - all fall short in characteristic ways: vector stores can't reason about what they retrieve, fine-tuning is expensive and slow, and big contexts don't help with cross-session memory.

RCK can sit underneath an LLM as its memory layer. The LLM handles intent parsing and surface generation; RCK handles storage, retrieval, reasoning, and explanation. The integration is a thin RPC layer (we provide an MCP server in the repository) and the conceptual API has four primitives:

```python
agent.remember(subject, relation, object)
   # source-tagged, provenance-recorded, O(1) write

result = agent.recall(subject=None, relation=None, object=None)
   # leaves any role None; returns EpistemicAnswer with
   #   .state ∈ {KNOWN, AMBIGUOUS, IDK}
   #   .top_symbol, .top_score
   #   .alternatives
   #   .drift_from_prior

explanation = agent.explain(fact_id)
   # walks the provenance graph; returns a tree of source facts

resolution = agent.correct(fact_id, replacement)
   # records the user correction; updates calibration tally;
   # invalidates affected cache entries; runs belief revision
```

In an LLM agent loop, this becomes:

```
user message → LLM extracts triples → agent.remember(...)
user question → LLM parses intent → agent.recall(...)
                                  → LLM renders the EpistemicAnswer
                                    as natural-language reply,
                                    refusing to invent if state == IDK
user correction → LLM identifies the disputed fact
                → agent.correct(...)
```

Two properties fall out for free. First, the LLM can no longer hallucinate facts the user told it about themselves - they're either in `agent.recall`'s output or they're not, with an IDK in the latter case. Second, the user can demand `agent.explain(...)` on any answer and get a real derivation tree, not a freshly-generated explanation.

We have not built and benchmarked this integration ourselves; it would be a natural collaboration with an LLM-tooling project. The RCK side is ready (the MCP server already exposes the four primitives above plus the full agent API).

### 8.2 Vertical agents in regulated domains

Medicine, law, finance, defense - domains where "the AI made it up" is a compliance issue, not just a quality issue. The auditability properties (provenance graph, IDK detection, contradiction surfacing) are valuable here in a way they aren't for consumer applications. A medical scribe that can be asked "where did you get the diabetes diagnosis from?" and produces the actual derivation tree pointing at the source notes is structurally different from one that generates plausible justifications.

### 8.3 Personal memory

A user owns their own KB. The KB lives on their device. Facts go in, answers come out, with citations, no data leaves the machine. The cost is approximately zero per query; the value is high for users who care about retaining ownership and auditability of what an AI knows about them.

### 8.4 Federated knowledge bases

Multiple parties each have their own KB; merging is a per-shard bundle sum with provenance preserved. Source tags survive the merge (`source="multi"` on collision), so a merged agent can still cite which contributor said what. This is appealing for medical or legal networks where parties want shared reasoning without ceding control over their own data.

**We withdraw the claim that "no entity-alignment step is needed."** It was true but vacuous, and section 5.10 shows why: name-hashed vectors need no alignment only because both parties already use identical identifiers - which is equally true of merging two dictionaries. When identifiers actually diverge, which is the only situation entity alignment exists to address, `hash("nyc") != hash("new_york_city")` and the bundle sum resolves nothing. Measured: after merging a renamed party's KB, cross-name resolution is 0.0% for a dict and 1.0% for RCK, and the 1.0% is bundle crosstalk rather than alignment - a false positive, which is worse than the honest zero. Federated merge is a genuine capability; it is not a property the substrate provides that a plain index lacks.

### 8.5 Research substrate

The implementation is small enough to fork (~16,100 lines of plain numpy Python, 757 tests, no GPU). Researchers interested in VSA-based reasoning, neuro-symbolic integration, calibration of vector-memory retrieval, or empirical study of chain-based induction can build directly on it. The filter stack, the propagation rules, and the calibrator are configuration, not hard-coded behavior, so alternative policies are easy to swap in.

---

## 9. Conclusion

We've described a working alternative to LLMs for structured reasoning tasks. RCK is not a competitor to GPT on open-domain text generation; it's something else. It demonstrates that a small, testable, CPU-only neuro-symbolic system can answer factual questions with calibrated confidence, explain why it knows what it knows, resolve contradictions between sources, learn new facts in O(1) from a single example, and reason across long chains of inference with honestly-reported uncertainty - all on commodity hardware, with no GPU and no generative layer to manage. Where we found our own numbers wanting - the retracted precision headline, the over-claimed 50-hop confidence horizon, a latency table whose source KBs had drifted - we corrected them in public, because a system whose pitch is auditability has to start with its own paper.

The code is available at <https://github.com/NORTHTEKDevs/rck> under the Apache 2.0 license. We welcome experimentation, criticism, and collaboration.

---

## Acknowledgements

This work was performed independently. The author thanks the foundational contributions of Pentti Kanerva, Tony Plate, Pei Wang, Chris Eliasmith, and the broader vector-symbolic, truth-maintenance, and neuro-symbolic communities, whose decades of work made this system possible to assemble in a year.

## References

- Agrawal, Garima and Kumarage, Tharindu and Alghamdi, Zeyad and Liu, Huan (2024). Can Knowledge Graphs Reduce Hallucinations in LLMs?: A Survey. *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics*.
- Alchourrón, Carlos E. and Gärdenfors, Peter and Makinson, David (1985). On the Logic of Theory Change: Partial Meet Contraction and Revision Functions. *Journal of Symbolic Logic 50*.
- Arnaout, Hiba and Razniewski, Simon and Weikum, Gerhard (2020). Enriching Knowledge Bases with Interesting Negative Statements. *Automated Knowledge Base Construction (AKBC)*.
- Bach, Stephen H. and Broecheler, Matthias and Huang, Bert and Getoor, Lise (2017). Hinge-Loss Markov Random Fields and Probabilistic Soft Logic. *Journal of Machine Learning Research 18*.
- Buchanan, Bruce G. and Shortliffe, Edward H. (1984). Rule-Based Expert Systems: The MYCIN Experiments of the Stanford Heuristic Programming Project. *Addison-Wesley*.
- Carlson, Andrew and Betteridge, Justin and Kisiel, Bryan and Settles, Burr and Hruschka Jr., Estevam R. and Mitchell, Tom M. (2010). Toward an Architecture for Never-Ending Language Learning. *Proceedings of the Twenty-Fourth AAAI Conference on Artificial Intelligence*.
- Clark, Keith L. (1978). Negation as Failure. In *Logic and Data Bases*, Plenum Press.
- Crawford, Eric and Gingerich, Matthew and Eliasmith, Chris (2016). Biologically Plausible, Human-Scale Knowledge Representation. *Cognitive Science 40*.
- de Kleer, Johan (1986). An Assumption-Based TMS. *Artificial Intelligence 28*.
- Doyle, Jon (1979). A Truth Maintenance System. *Artificial Intelligence 12*.
- Eliasmith, Chris (2013). How to Build a Brain: A Neural Architecture for Biological Cognition. *Oxford University Press*.
- Eliasmith, Chris; Stewart, Terrence C.; Choo, Xuan; Bekolay, Trevor; DeWolf, Travis; Tang, Yichuan; Rasmussen, Daniel (2012). A Large-Scale Model of the Functioning Brain. *Science 338*.
- Galárraga, Luis; Teflioudi, Christina; Hose, Katja; Suchanek, Fabian M. (2013). AMIE: Association Rule Mining under Incomplete Evidence in Ontological Knowledge Bases. *Proceedings of the 22nd International Conference on World Wide Web*.
- Garcez, Artur d'Avila and Lamb, Luís C. (2020). Neurosymbolic AI: The 3rd Wave. *arXiv:2012.05876*.
- Gelfond, Michael and Lifschitz, Vladimir (1991). Classical Negation in Logic Programs and Disjunctive Databases. *New Generation Computing 9*.
- Genest, Christian and Zidek, James V. (1986). Combining Probability Distributions: A Critique and an Annotated Bibliography. *Statistical Science 1*.
- Heddes, Mike; Nunes, Igor; Vergés, Pere; Kleyko, Denis; Abraham, Danny; Givargis, Tony; Nicolau, Alexandru; Veidenbaum, Alexander (2023). Torchhd: An Open Source Python Library to Support Research on Hyperdimensional Computing and Vector Symbolic Architectures. *Journal of Machine Learning Research 24*.
- Hersche, Michael; Zeqiri, Mustafa; Benini, Luca; Sebastian, Abu; Rahimi, Abbas (2023). A Neuro-Vector-Symbolic Architecture for Solving Raven's Progressive Matrices. *Nature Machine Intelligence 5*.
- Huang, Lei et al. (2024). A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. *ACM Transactions on Information Systems*.
- Imani, Mohsen; Kim, Yeseong; Riazi, M. Sadegh; Messerly, John; Liu, Patric; Koushanfar, Farinaz; Rosing, Tajana (2019). A Framework for Collaborative Learning in Secure High-Dimensional Space. *IEEE 12th International Conference on Cloud Computing (CLOUD)*.
- Jøsang, Audun (2016). Subjective Logic: A Formalism for Reasoning Under Uncertainty. *Springer*.
- Kanerva, Pentti (2009). Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors. *Cognitive Computation 1*.
- Kelly, Matthew A.; Arora, Nipun; West, Robert L.; Reitter, David (2020). Holographic Declarative Memory: Distributional Semantics as the Architecture of Memory. *Cognitive Science 44*.
- Krompaß, Denis; Baier, Stephan; Tresp, Volker (2015). Type-Constrained Representation Learning in Knowledge Graphs. *Proceedings of the 14th International Semantic Web Conference*.
- Lao, Ni and Cohen, William W. (2010). Relational Retrieval Using a Combination of Path-Constrained Random Walks. *Machine Learning 81*.
- Lao, Ni; Mitchell, Tom; Cohen, William W. (2011). Random Walk Inference and Learning in a Large Scale Knowledge Base. *Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing*.
- Lenat, Doug and Marcus, Gary (2023). Getting from Generative AI to Trustworthy AI: What LLMs Might Learn from Cyc. *arXiv:2308.04445*.
- Lewis, Patrick et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems*.
- Magesh, Varun; Surani, Faiz; Dahl, Matthew; Suzgun, Mirac; Manning, Christopher D.; Ho, Daniel E. (2024). Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools. *arXiv:2405.20362*.
- Manhaeve, Robin; Dumančić, Sebastijan; Kimmig, Angelika; Demeester, Thomas; De Raedt, Luc (2018). DeepProbLog: Neural Probabilistic Logic Programming. *Advances in Neural Information Processing Systems*.
- Nickel, Maximilian; Rosasco, Lorenzo; Poggio, Tomaso (2016). Holographic Embeddings of Knowledge Graphs. *Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence*.
- Plate, Tony A. (1995). Holographic Reduced Representations. *IEEE Transactions on Neural Networks 6*.
- Richardson, Matthew and Domingos, Pedro (2006). Markov Logic Networks. *Machine Learning 62*.
- Safavi, Tara; Zhu, Jing; Koutra, Danai (2021). NegatER: Unsupervised Discovery of Negatives in Commonsense Knowledge Bases. *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*.
- Wang, Pei (2006). Rigid Flexibility: The Logic of Intelligence. *Springer*.
- Zeulin, Nikita; Galinina, Olga; Himayat, Nageen; Andreev, Sergey (2023). Federated Hyperdimensional Computing. *arXiv:2312.15966*.

(The BibTeX source for these entries is `references.bib` in this directory; the LaTeX manuscript cites them inline.)

## Citing this work

```bibtex
@misc{rck2026,
  author       = {Baer, Kristian},
  title        = {{RCK}: A Hallucination-Free Reasoning System Built on
                  Hyperdimensional Computing},
  year         = {2026},
  howpublished = {Preprint, available at \url{https://github.com/NORTHTEKDevs/rck}},
  version      = {15.3.1},
  url          = {https://github.com/NORTHTEKDevs/rck},
}
```
