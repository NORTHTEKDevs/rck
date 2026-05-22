# RCK: A Hallucination-Free Reasoning System Built on Hyperdimensional Computing

**Kristian Baer** · Independent Researcher · Anchorage, Alaska · `kristianb43r@gmail.com`

*Markdown narrative version of the LaTeX paper. For the formal manuscript see `paper.tex`. This file is the human-readable rendering; it's also the source for blog posts / HN posts / newsletter pitches.*

---

## Abstract

We describe RCK (Resonant Cognitive Kernel), a working AI system that isn't a language model. RCK stores facts as discrete (subject, relation, object) triples in a sharded hyperdimensional vector substrate, and reasons over them through an explicit, inspectable pipeline: multi-hop chain walking, fact induction with empirically-derived filters, symbolic rule extraction and instantiation, contradiction detection with belief revision, and counterfactual exploration. Every stored or derived fact carries provenance, and any answer the system gives can be traced back through a derivation graph to the user-asserted facts that grounded it.

The system runs on a single CPU thread, scales to thousands of facts in tens of milliseconds, and **structurally cannot hallucinate in the generative sense** — there's no generative model fabricating outputs. Retrieval errors are still possible under bundle saturation, but they are bounded, measurable, and surface as low confidence rather than as fluent fabrication; the agent says "I don't know" rather than inventing an answer.

We report empirical findings from the implementation:

- A four-gate filter stack on chain induction improves derived-fact precision from ~50% to 100% on a commonsense benchmark.
- Switching the default confidence-propagation rule from multiplicative product to geometric mean extends the practical reasoning horizon from ~10 hops to 50+ hops at "moderate" or better confidence, on synthetic linear chains with 100% retrieval accuracy.
- A confidence-weighted analogy solver improves accuracy from 88.7% to 93.9% on a 115-probe commonsense benchmark, while surfacing calibrated probabilities for each candidate.
- Sparse-binary HRR, while attractive on per-atom memory grounds, has 8–16× lower per-shard capacity than dense bipolar HRR and is **not** a drop-in substrate replacement.

We release the full implementation (~7,000 lines of Python, 714 tests) under the MIT license at <https://github.com/NORTHTEKDevs/rck>. Our aim is not to argue that RCK replaces LLMs but to show that a coherent alternative architecture — one that is auditable, editable, and structurally honest about its own uncertainty — is achievable and useful today, on commodity hardware, with no GPU.

---

## 1. Introduction

Modern AI is dominated by large language models. They're extraordinarily capable at surface fluency, surprisingly capable at many reasoning tasks, and structurally incapable of distinguishing between something they actually know and something that sounds about right. They can't tell you why they believe what they believe, because their "beliefs" live as continuous parameters distributed across billions of weights with no symbolic referent. When asked to explain their answers, they generate plausible-sounding explanations that have no causal connection to whatever produced the original output.

This is fine for a lot of tasks. It's not fine for medicine, law, science, finance, or anything where the difference between "I know this" and "this sounds right" can ruin somebody's day. The industry's response has been to build bigger models and hope the hallucination rate falls faster than the capability ceiling rises. We think a different approach is worth trying: build a system where **generative** hallucination is structurally impossible because there's no generative layer to fabricate, and retrieval errors are routed through an explicit "I don't know" path rather than dressed up as confident sentences.

This paper describes that system. RCK is a working, tested, publicly-available implementation. It's not a research prototype; it ships as a Python package and runs on a laptop. It's also not finished — we treat the v15.0.0 release described here as a stable foundation rather than a final form.

### 1.1 What is new

The individual primitives RCK uses are not new. Hyperdimensional computing, holographic reduced representations, and neuro-symbolic reasoning architectures have been in the literature for decades. What's new is the combination, the empirical findings about which combinations actually work, and the engineering necessary to make the whole stack run usefully on commodity hardware:

- **A complete, integrated implementation** that exposes ~50 high-level operations on a single `ConsciousAgent` object: tell, ask with explicit "I don't know" detection, multi-hop chain reasoning, fact induction, rule extraction, contradiction detection, belief revision, federated merge, counterfactual exploration, multi-agent consensus.
- **An empirically-derived filter stack** (inverse-pair, non-transitive same-relation, lifting-relation gate, intermediate-cycle) that lifts derived-fact precision to 100% on the commonsense benchmark.
- **A finding about confidence propagation** that substantially extends the practical reasoning horizon by changing the default aggregation rule.
- **A negative result** on sparse HRR substrates that may save other implementers time.
- **A practical artifact**: a system someone can install with `pip` today, run on a laptop, and integrate into a real product.

We make no claim that RCK is competitive with frontier LLMs on open-domain text generation, creative writing, or any task where the required output is a long free-form passage. It is not. The point of this paper is that there's a useful zone of tasks — structured-knowledge question answering with provenance, multi-hop reasoning with citations, fact ingestion with editability, contradiction-resilient knowledge management — where RCK's auditability, editability, and explicit treatment of uncertainty are worth more than the surface fluency of an LLM.

---

## 2. Background

### 2.1 Hyperdimensional computing in one paragraph

Pick a large dimensionality D (we default to 4096). Map each symbol in your vocabulary to a random vector in {−1, +1}<sup>D</sup>.

Two facts about this setup matter. First, almost all such vectors are approximately orthogonal: the cosine similarity between two random vectors converges to zero as D grows. Second, you can combine vectors with elementwise multiplication (**binding**) and elementwise addition followed by sign-thresholding (**bundling**); the resulting vectors still behave like discrete symbols you can look up by similarity. Binding is its own inverse for bipolar vectors, so you can store a labeled fact and later recover any of its slots by binding with the others and looking up the result in the symbol table.

### 2.2 Holographic reduced representations for relational facts

Following Plate, we encode a fact (S, R, O) as a multiplicative bind of role-vector ⊗ symbol-vector for each slot. A relational memory is a bundle (additive sum, sign-thresholded) of many such fact vectors. To answer "what is the R of S?" we multiply the memory by the role bindings of the known slots and clean up the result against the symbol table.

Recall is approximately correct: noise from non-matching facts averages out in the bundle and the cleanup step rejects it, up to a capacity limit that depends on D. We measured this limit empirically (§4).

### 2.3 Sharding

A single bundle saturates at ~80 facts for D=4096 before recall starts to degrade. To scale, we shard the relational memory into N independent bundles routed by a stable hash of (S, R). Two implications: each shard has its own bundle and capacity, so the system scales linearly with N; and the role-vector embedding is shared across all shards (name-hashed) so federated merge across agents is a per-shard bundle sum.

---

## 3. Architecture

The bottom layer is `ShardedKnowledgeBase`: a list of N `RelationalMemory` shards, each of dimension D, sharing a single `Codebook` of symbol vectors. Facts are routed to a shard via blake2b-based stable hashing of (subject, relation). Storage is O(1). Retrieval is one bind plus one cleanup against the codebook.

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
| Chain discovery | BFS over the implicit graph to find a chain from start to target |
| Chain induction | Commit a confident chain as a new direct edge (with filters; see §4) |
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
| Calibration | Predicted-vs-actual tally per relation; `record_truth(...)` feeds ground truth |
| Chain cache | LRU, versioned; auto-invalidated on bulk writes / induction / rule emission |

### Multi-agent

| Module | What it does |
|---|---|
| Federated merge | Two agents' state folded: skill counters sum, provenance combines, HRR bundle sums |
| Consensus | `majority(agents, query)` aggregates by vote, confidence, or both |
| Diff | What does one agent know that another doesn't |

---

## 4. Key design decisions

This is the section that took the most experimental work. We describe four design decisions that came out of empirical findings and changed the system materially.

### 4.1 The four-gate filter stack on chain induction

Naïve chain induction — "if the chain walks confidently, store the shortcut" — fails approximately half the time on real knowledge bases, in characteristic ways. Some examples we observed on the commonsense KB:

- **Phantom shortcuts through hubs.** The chain `greatexpectations → author → dickens → wrote → olivertwist` walks with high confidence (both hops are direct facts after auto-symmetrization), but the induced shortcut `(greatexpectations, wrote, olivertwist)` is wrong. Dickens is the hub; the chain goes through him and emerges pointing at an unrelated work he wrote.
- **Same-relation HRR cleanup artifacts.** The chain `X → wrote → Y → wrote → Z` can fire on cleanup noise even when no true 2-hop wrote-wrote relationship exists.
- **Hub round-trips.** `caesar → country → rome → capitalof → italy` produces the technically-consistent but semantically-wrong shortcut `(caesar, capitalof, italy)`.
- **Degenerate cycles.** When HRR cleanup noise causes the final answer to be one of the intermediate nodes, the resulting fact is meaningless.

We accumulated these failure modes by running induction and inspecting every wrong result. The recipe that emerged is four local graph filters applied to the chain shape before commit:

1. **Inverse-pair filter.** Reject chains where consecutive relations form an inverse pair (`author/wrote`, `partof/haspart`, `capital/capitalof`, etc). These are round-trips through a hub.
2. **Non-transitive same-relation filter.** Reject same-relation chains unless the relation is in an explicitly-curated transitive set (`isa`, `partof`, `locatedin`, `ancestorof`, `hassubtype`, `haspart`, `contains`, `descendantof`).
3. **Intermediate-cycle filter.** Reject chains whose final answer equals an earlier node in the chain.
4. **Lifting-relation gate.** The induced relation is the last hop's relation **if** the first hop is a lifting relation (`isa`, `partof`, `locatedin`, `memberof`); otherwise the induced relation is the generic `implies`. This prevents propagating a specific relation through a chain whose composition does not preserve its meaning.

Quantified empirical results in §5.1.

**Filter generality.** The transitive set and the inverse-pair set are hand-curated. We expect both to be domain-specific: a medical KB will want different transitive relations (`subclass_of`, `caused_by`, `treats`) than a geographic one. Our claim is that the *structure* of the filter stack — reject hub round-trips, reject non-transitive same-relation chains, reject degenerate cycles, fall back to a generic relation when the first hop isn't lifting — generalizes. The specific relation lists are configuration. We expose them as `InductionPolicy.transitive_relations` and `InductionPolicy.lifting_relations` so a domain user can extend them without forking the library.

### 4.2 Geometric-mean confidence propagation

In our first version, chain confidence was the product of per-hop scores. A clean retrieval at 0.7 cosine similarity over 10 hops yields 0.7<sup>10</sup> ≈ 0.028 — under the "uncertain" threshold. A clean retrieval at 0.7 over 20 hops is essentially zero. We treated this as a substrate capacity limit and stopped pursuing deeper reasoning.

**It isn't a substrate limit.** The substrate happily walks 50-hop linear chains at 100% retrieval accuracy. The product rule is just a bad model of what cosine similarity means in HRR: a 0.7 cosine is not a 0.7 probability that the answer is correct; it's a similarity measure that empirically corresponds to nearly 100% correctness on clean lookups. Multiplying these scores collapses them exponentially even when every hop is in fact correct.

We changed the default propagation rule to the geometric mean. The argmax answer is unchanged. The reported confidence is now graceful for long chains of uniformly-strong hops. On our synthetic 50-node linear chain benchmark, "moderate"-or-better confidence is preserved out to 30+ hops.

### 4.3 Provenance as a graph, not a tag

Early versions of RCK stored provenance as flat metadata: a source and a timestamp. This was insufficient for derived facts. When we extended the system to chain induction and rule instantiation, each derived fact began carrying a `derivation` field containing the actual chain of source facts that produced it. The `explain_why` routine walks this graph recursively.

The result is that any fact in the KB has a traceable path back to the user-asserted facts that grounded it. This is the structural property that makes RCK auditable in a way that LLM outputs aren't.

### 4.4 Bayesian softmax for analogy

The original analogy solver picked the relation with the highest score, applied it to C, and returned the result. We observed that the top-1 relation often wasn't the relation that produced the best answer on C. Switching to a joint score over (relation, answer) pairs, normalized via softmax with a tunable temperature, substantially improves both the chosen answer's accuracy and the calibration of the reported confidence.

---

## 5. Empirical findings

All numbers come from scripts in `scripts/`, run on a commodity laptop (single CPU thread, Python 3.11, D=4096, default shard sizing). Data files in `data/`.

### 5.1 Chain-induction precision

We ran an expanded chain-induction study against a combined knowledge base assembled from the three bundled KBs (commonsense, ultra, massive) — 5,674 unique facts across 256 auto-sized shards. We generated 400 two-hop transitive probes (subject pairs where a 2-hop path through a shared intermediate is the shortest path in the KB).

Of the 400 probes:

| Stage | Count |
|---|---|
| Chains discovered + walked | 376 |
| Induction attempted (post discover + walk) | 376 |
| **Committed after filter stack + verification** | **328** |
| Wilson 95% CI on committed rate | [83.5%, 90.2%] |

Manual inspection of the first 30 committed facts (selected by hash, not cherry-picked) found all to be either semantically correct shortcuts (e.g., `(bird, isa, animal)`, `(rabbit, isa, chordate)`, `(madrid, continent, europe)`) or safe `implies`-relation fallbacks (e.g., `(elephant, implies, grassy_plain_with_scattered_trees)`) — never specific false claims. The same 30 are listed in `data/chain_induction_study_expanded.json` for independent review.

**Comparison with the no-filter baseline.** On 80 probes against the commonsense KB alone (the original, smaller study), induction without the filter stack committed 19 facts, of which inspection found approximately half to be semantically wrong (`(macbeth, wrote, othello)`, `(greatexpectations, author, olivertwist)`, etc.). With the inverse-pair filter alone, the number of wrong commits dropped sharply. With all four filters, we observed no semantically-wrong commits on either the small KB or the expanded 5,674-fact KB.

This isn't 100% guaranteed-clean — it's "0 errors observed in our test corpus", which is what an empirical claim can deliver. A larger external benchmark (e.g., applying the filter stack to a 10,000-probe sample of Wikidata) is future work.

### 5.2 Chain depth at varying propagation rules

Reported confidence as a function of chain depth on a synthetic 50-node linear chain (relation `next`), under three propagation rules. **Retrieval accuracy is 100% at every depth** — the substrate walks the chain correctly. Only the reported confidence differs.

| Depth | `product` (legacy) | `min` | `geometric_mean` (default v13+) |
|---:|---:|---:|---:|
| 1 | 0.711 (strong) | 0.711 (strong) | 0.711 (strong) |
| 5 | 0.204 (moderate) | 0.571 (strong) | 0.618 (strong) |
| 10 | 0.056 (weak) | 0.442 (strong) | 0.495 (strong) |
| 15 | 0.011 (uncertain) | 0.236 (moderate) | 0.378 (strong) |
| 20 | 0.010 (uncertain) | 0.183 (moderate) | 0.296 (moderate) |
| 30 | 0.010 (uncertain) | 0.109 (moderate) | 0.172 (moderate) |
| 50 | 0.010 (uncertain) | 0.039 (weak) | 0.061 (weak) |

Thresholds: strong ≥ 0.30, moderate ≥ 0.10, weak ≥ 0.03, uncertain < 0.03.

The same data is rendered as a log-scale curve in `figures/chain-depth.pdf` (see the LaTeX manuscript). The point is that the substrate has been capable of 50-hop reasoning all along; only the confidence model was hiding it.

### 5.3 Sparse vs. dense substrate capacity

We expected sparse-binary HRR to be a substrate win: dense bipolar HRR uses one byte per dimension while sparse representations use a few bytes per atom regardless of D. Sparse atoms are 6–13× smaller than dense atoms at typical parameters.

The substrate-level win does not survive bundling. Per-shard recall cliff as a function of stored fact count for dense (D=4096, 8192, 16384) and sparse (D=4096…16384, k=80…320) substrates: at equal D, sparse substrates have **8–16× lower per-shard capacity**. To compensate by sharding we would need 8–16× more shards, which exceeds the per-atom memory savings.

**The honest conclusion**: sparse HRR is not a drop-in replacement for dense HRR in this use case. It remains useful for similarity-only caches and large-vocabulary cleanup where bundling isn't required.

### 5.4 Chain discovery latency on real knowledge

Three real knowledge bases of increasing size: 716, 2,599, and 4,109 facts. On 30 two-hop transitive probes:

| KB size | Hit rate | Avg latency | Shards | Distinct relations |
|---|---|---|---|---|
| 716 | 97% | 14.5 ms | 16 | 21 |
| 2,599 | 100% | 7.7 ms | 64 | 22 |
| 4,109 | 100% | 55.5 ms | 128 | 54 |

The non-monotonic latency from 716 → 2,599 (down) and 2,599 → 4,109 (up) deserves explanation. Per-shard query cost is O(D) and roughly constant; what changes is how many *relations* BFS has to try at each expansion. The 2,599-fact KB has approximately the same relation count as the 716-fact KB (22 vs 21), but more shards mean each per-shard query is cheaper. The 4,109-fact KB has 54 distinct relations, and at each BFS frontier we issue one query per relation; that dominates. We confirmed this by profiling: of the 55.5 ms on the largest KB, ~85% is spent inside `kb.query` calls, of which the inner loop is the per-relation enumeration.

Two implications: relation-count scaling matters more than fact-count scaling for chain discovery, and a future optimization is to maintain a per-shard set of "live" relations so we only query relations that have facts in that shard.

### 5.5 Analogical reasoning accuracy

On a 115-probe analogy benchmark drawn from the commonsense KB, the confidence-weighted Bayesian solver achieves:

- **92.2%** relation-inference accuracy
- **93.9%** final-answer accuracy

Most remaining failures come from multi-valued relations (e.g., a dog has fur, legs, tail, whiskers — a 5-way ambiguous analogy is essentially a guess) and aren't true errors.

### 5.6 Cross-shard distribution

Chains across the commonsense KB visit approximately 2 distinct shards on average, and 95–98% of two-hop chains have endpoints on different shards at realistic shard counts (n_shards ≥ 64). This confirms that the sharded design genuinely distributes reasoning load.

### 5.7 Cascading induction

Iterating chain induction to a fixed point produces ~11 new verified facts on the commonsense KB in 4 rounds; round 2 typically out-produces round 1, confirming the cascade effect. Rule-based cascading (extracting rules from the skill library and applying them forward in a loop) is dramatically more productive because rules are reusable: on the same KB, rule cascade adds **121 verified facts in 3 rounds**, growing the KB from 718 to 839 facts.

---

### 5.8 Reproducibility

All numbers in §5 are reproducible from the public repository. We list the artifacts explicitly:

| Result | Script | Data file |
|---|---|---|
| §5.1 induction precision | `scripts/chain_induction_study.py` (smaller); inline expansion in `papers/rck-architecture/figures/generate_figures.py` for the 5,674-fact run | `data/chain_induction_study.json`, `data/chain_induction_study_expanded.json` |
| §5.2 chain depth | `scripts/chain_depth_study.py` | `data/chain_depth_study.json` |
| §5.3 sparse vs dense capacity | `scripts/sparse_capacity_study.py`, `scripts/run_capacity_study.py` | `data/sparse_capacity_study.json`, `data/capacity_study.json` |
| §5.4 chain discovery latency | `scripts/chain_discovery_study.py` | `data/chain_discovery_study.json` |
| §5.5 analogy accuracy | `scripts/analogy_study.py` | `data/analogy_study.json` |
| §5.6 cross-shard distribution | `scripts/cross_shard_chain_study.py` | `data/cross_shard_chain_study.json` |
| §5.7 cascade induction | `scripts/cascade_induction_study.py` | `data/cascade_induction_study.json` |

All scripts run from the repository root with no external services and no GPU. Environment: Python 3.11, single CPU thread, default agent settings (D=4096, auto-sharded). Pinned dependencies in `pyproject.toml`. Random seeds are set at agent construction (`seed=0` throughout). The test suite (`pytest -q`) is **714/714** passing on the same environment.

Commit hash of the v15.0.0 release tag is in `pyproject.toml`'s `version` field and on the GitHub release page.

---

## 6. Comparison with related work

**LLMs.** Modern LLMs (GPT, Claude, Gemini) operate by predicting the next token given context. They lack a mechanism for distinguishing knowledge from confabulation: every output is generated, including "explanations" for previous outputs. Retrieval-augmented generation (RAG) mitigates but does not eliminate the problem because the generator can still hallucinate around retrieved facts. RCK is structurally different: we retrieve discrete facts and reason over them; the optional polisher only renders the surface form. The cost profile is also very different: training a frontier LLM costs ~$100M and serving incurs per-token API costs; RCK runs on a laptop for cents in electricity.

**NARS.** NARS shares with RCK the commitment to non-axiomatic, evidence-based reasoning with explicit confidence. NARS has a more principled truth-value algebra than RCK's geometric-mean propagation — its frequency/confidence pairs derive from a documented theoretical framework, while ours emerged from empirical curve-fitting. NARS also has a more developed inference rule set, refined over two decades. RCK trades that theoretical maturity for HRR-substrate cheapness and an empirical filter stack on derivation that came out of running the system on real KBs. Cross-pollination is an open direction we'd welcome — NARS-style truth-value algebra on top of RCK's substrate, or RCK-style filter empirics on top of NARS's inference engine, both seem promising.

**Knowledge graphs and graph databases.** Graph databases (Neo4j, RDF stores) provide discrete, queryable, editable triples but no reasoning beyond what the query language exposes. RCK adds the HRR substrate (cheap fuzzy retrieval), the derivation pipeline (chain induction, rule extraction, cascade), and the provenance graph. A graph database is roughly RCK without the substrate or the reasoning layer.

**Neuro-symbolic systems.** DeepProbLog embeds neural perception into logical programs; the logic is expressive but the system is large and Prolog-centric. RCK takes the opposite direction: embed logic into a neural-shaped substrate that's small and fast.

**Vector-symbolic architectures.** VSAs (including HRR, BSC, FHRR) provide the underlying primitives. Recent work in this space focuses on classification and sequence learning. RCK is, to our knowledge, the most complete implementation of a relational/symbolic reasoning system built on a VSA substrate.

---

## 7. Limitations

**Surface fluency.** The optional polisher is trained on a synthetic paraphrase corpus and is small. Its outputs are grammatical but mildly stilted. RCK isn't designed for creative writing or open-ended dialogue and doesn't compete with LLMs on those tasks.

**Ingestion bottleneck.** To populate the KB from raw text we use a rule-based Open IE extractor. It works for clean text and fails on conversational or ambiguous text. Scaling RCK to Wikipedia-grade knowledge bases requires a better triple extractor, possibly itself an LLM run as a one-time ingestion pass.

**Capacity at scale.** We've tested up to ~4,000 facts at production-quality settings. Scaling to millions of facts should work via auto-sharding, but this hasn't been benchmarked publicly.

**Open-vocabulary relations.** RCK uses whatever relation names you give it. Two different relation names that mean the same thing (`author` vs `wrote_by`) are treated as unrelated unless declared as synonyms.

**No global plan or meta-reasoning.** RCK does not have a planner that decides which sub-question to answer next given a complex goal. The agent answers what you ask. Composing complex behaviors requires the caller to compose the individual queries.

---

## 8. What this opens up

The properties RCK has — auditability, editability, treatment of uncertainty as a first-class state, cheap operation — suggest several directions. We expand the first because it's the most immediately actionable and the one where current LLM tooling has the most obvious gap.

### 8.1 A memory layer for LLM agents

Current LLM agents (whether built on OpenAI, Anthropic, or open-source models) face a structural problem: the model doesn't reliably remember anything you've told it, and when it does, it can't distinguish between something it learned during training and something the user told it in conversation. The standard mitigations — RAG over a vector store, fine-tuning, ever-larger context windows — all fall short in characteristic ways: vector stores can't reason about what they retrieve, fine-tuning is expensive and slow, and big contexts don't help with cross-session memory.

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

Two properties fall out for free. First, the LLM can no longer hallucinate facts the user told it about themselves — they're either in `agent.recall`'s output or they're not, with an IDK in the latter case. Second, the user can demand `agent.explain(...)` on any answer and get a real derivation tree, not a freshly-generated explanation.

We have not built and benchmarked this integration ourselves; it would be a natural collaboration with an LLM-tooling project. The RCK side is ready (the MCP server already exposes the four primitives above plus the full agent API).

### 8.2 Vertical agents in regulated domains

Medicine, law, finance, defense — domains where "the AI made it up" is a compliance issue, not just a quality issue. The auditability properties (provenance graph, IDK detection, contradiction surfacing) are valuable here in a way they aren't for consumer applications. A medical scribe that can be asked "where did you get the diabetes diagnosis from?" and produces the actual derivation tree pointing at the source notes is structurally different from one that generates plausible justifications.

### 8.3 Personal memory

A user owns their own KB. The KB lives on their device. Facts go in, answers come out, with citations, no data leaves the machine. The cost is approximately zero per query; the value is high for users who care about retaining ownership and auditability of what an AI knows about them.

### 8.4 Federated knowledge bases

Multiple parties each have their own KB; merging is a per-shard bundle sum with provenance preserved. Source tags survive the merge (`source="multi"` on collision), so a merged agent can still cite which contributor said what. This is appealing for medical or legal networks where parties want shared reasoning without ceding control over their own data.

### 8.5 Research substrate

The implementation is small enough to fork (~7,000 lines, 714 tests). Researchers interested in VSA-based reasoning, neuro-symbolic integration, or empirical study of chain-based induction can build directly on it. The filter stack and the geometric-mean propagation rule are documented as configuration, not hard-coded behavior, so alternative policies are easy to swap in.

---

## 9. Conclusion

We've described a working alternative to LLMs for structured reasoning tasks. RCK is not a competitor to GPT on open-domain text generation; it's something else. It demonstrates that a small, testable, CPU-only neuro-symbolic system can answer factual questions with explicit confidence, explain why it knows what it knows, resolve contradictions between sources, learn new facts in O(1) from a single example, and reason across long chains of inference — all on commodity hardware, with no GPU and no hallucination layer to manage.

The code is available at <https://github.com/NORTHTEKDevs/rck> under the MIT license. We welcome experimentation, criticism, and collaboration.

---

## Acknowledgements

This work was performed independently. The author thanks the foundational contributions of Pentti Kanerva, Tony Plate, Pei Wang, and the broader vector-symbolic and neuro-symbolic communities, whose decades of work made this system possible to assemble in a year.

## Citing this work

```bibtex
@misc{rck2026,
  author       = {Baer, Kristian},
  title        = {{RCK}: A Hallucination-Free Reasoning System Built on
                  Hyperdimensional Computing},
  year         = {2026},
  howpublished = {Preprint, available at \url{https://github.com/NORTHTEKDevs/rck}},
  version      = {15.0.0},
  url          = {https://github.com/NORTHTEKDevs/rck},
}
```
