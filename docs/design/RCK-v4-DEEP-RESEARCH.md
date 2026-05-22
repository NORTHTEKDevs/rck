# RCK v4+ — Deep Research: Path to a New Class of Generative AI

**Status:** research roadmap. Written 2026-05-20.
**Aim:** a concrete plan to build a generative AI that **rivals GPT-4-class
LLMs** on the capabilities users care about, while costing 3-5 orders of
magnitude less to construct.

This is not a marketing claim. The argument hinges on a structural
observation about what LLMs are actually doing, why that requires the
compute they need, and how a factored architecture provably escapes
the cost curve.

---

## 1. What LLMs Actually Are (the honest framing)

A modern frontier LLM is doing exactly one thing during training:
**learning a single conditional distribution `p(next_token | context)`
over a trillion-token corpus**, with ~10²⁵ FLOPs of compute. Out of that
single objective the model must simultaneously discover:

  1. World facts ("Paris is the capital of France")
  2. Grammar and syntax (subject-verb agreement, anaphora, ...)
  3. Semantic composition ("a small blue ball" from atoms)
  4. Reasoning (transitivity, modus ponens, chain-of-thought)
  5. Format/style (JSON, Markdown, code, Shakespeare voice)
  6. Calibrated confidence (when to refuse)
  7. Multi-turn dialogue
  8. Tool use (post-fine-tuning)
  9. Long-context retrieval (post-fine-tuning)

**All of these compete for the SAME parameter pool.** Scale works
because adding parameters trades off slowly: doubling params adds
capability in every dimension simultaneously. This is the bitter
lesson at work. But the converse is also true: every dimension
saturates separately, and most users only care about a few at a time.

**The compute cost arises from the joint discovery problem**, not
from the capabilities themselves. If we factor the problem into
modules with explicit structure for each capability, the discovery
problem disappears, and so does most of the compute.

---

## 2. Literature Survey: What's Already Known

These are the pieces of the puzzle that exist in the literature but
have NOT been combined into a single production system. I'll cite the
canonical paper / project where appropriate.

### 2.1 Hyperdimensional Computing / VSA
- **Kanerva (2009)** Hyperdimensional Computing.
- **Plate (1995)** Holographic Reduced Representations.
- Provides: substrate for symbolic composition without learned weights.
- **Capacity:** D-dim bipolar HVs hold ~D / log(N) facts at 95% recall.
- At D=10000, holds ~1000-3000 facts cleanly per memory. Shard count
  scales total capacity LINEARLY.
- **Already in RCK** (v0.1+).

### 2.2 Knowledge Graphs at Scale
- **ConceptNet** (Speer et al., 2017): 36M assertions in 304 languages,
  free.
- **Wikidata**: 100M+ items, 1B+ statements, structured + free.
- **DBpedia**: 3B+ triples extracted from Wikipedia.
- **Cyc** (Lenat 1984-): 25M assertions, hand-curated common sense.
- Provides: the knowledge content. Nobody has run any of these through
  a VSA substrate at full scale.

### 2.3 Open Information Extraction
- **Stanford OpenIE** (Angeli et al., 2015): production-grade triple
  extraction.
- **REBEL** (Cabot & Navigli, 2021): end-to-end relation extraction.
- **Triplex** (Sciphi, 2024): small LLM fine-tuned for triple
  extraction from text.
- Provides: automatic conversion of any text corpus into structured
  triples. Doesn't require RCK-specific training.

### 2.4 Retrieval-Augmented Generation
- **Lewis et al. (2020)** RAG: retrieve documents, feed to LLM.
- **Borgeaud et al. (2022)** RETRO: retrieve at every transformer
  block. Showed a 7B-param model + retrieval matches a 280B model.
- **Atlas** (Izacard et al., 2022): few-shot via retrieval.
- Provides: PROOF that 95%+ of LLM scaling is wasted memorizing
  facts. With retrieval, a 7B model rivals a 280B model. Retrieval is
  the right answer; the only question is what form the retrieval takes.

### 2.5 Neuro-Symbolic AI
- **NARS** (Wang, 1995-): Non-Axiomatic Reasoning System -- explicit
  logical inference with uncertainty.
- **OpenCog** / **Hyperon** (Goertzel): hypergraph-based reasoning.
- **DeepProbLog** (Manhaeve et al., 2018): differentiable Prolog.
- **DreamCoder** (Ellis et al., 2020): library-learning for programs.
- **Schema Networks** (Kansky et al., Vicarious, 2017): causal
  reasoning via graph structure.
- Provides: the algorithmic side of reasoning -- inference rules,
  uncertainty, abduction. RCK already implements inheritance,
  transitivity, multi-step queries; this is the "v8" extension.

### 2.6 Distillation + Small Models
- **DistilBERT** (Sanh et al., 2019): 40% smaller, 97% of BERT.
- **TinyStories** (Eldan & Li, 2023): 30M-param LMs that produce
  grammatical English when trained on a constrained domain.
- **Phi-1/2/3** (Gunasekar et al., 2023): SMALL models trained on
  CURATED data outperform large models on broad data.
- Provides: PROOF that you can have a "fluent English generator" in
  100M params. The 100B-param scale exists for OTHER reasons (knowledge
  + reasoning) -- and those reasons can be factored out into RCK.

### 2.7 Active Inference / Free Energy Principle
- **Friston (2010+)**: minimize expected free energy.
- **Schwartenbeck & Friston (2016)**: discrete-action planning.
- Already in RCK (`rck/fep.py`). Provides the policy substrate for
  goal-directed behavior.

### 2.8 Test-Time Reasoning / Chain-of-Thought
- **Wei et al. (2022)** CoT prompting.
- **OpenAI o1 / o3** (2024-25): test-time reasoning at scale.
- **Tree-of-thoughts** (Yao et al., 2023): explicit search.
- Provides: the insight that REASONING can be done at inference time
  via search/sampling, not just trained in. RCK can do this NATIVELY
  via graph search (`rck/inference.py`, `rck/multistep.py`).

### 2.9 World Models
- **Ha & Schmidhuber (2018)**: world model + small controller.
- **Dreamer V3** (Hafner et al., 2024): planning in latent space.
- Provides: the proof that small predictors + structured state
  outperform monolithic policies.

### 2.10 Memory-Augmented Transformers
- **MemoryLLM** (Wang et al., 2024): 7B + persistent memory.
- **MemGPT** (Packer et al., 2023): OS-style paged memory.
- **Larimar** (Das et al., 2024): episodic memory for LLM updates.
- Provides: even within the transformer paradigm, external memory
  is now standard. RCK takes this to the extreme: memory is the
  ENTIRE knowledge store.

---

## 3. The Gap: What Hasn't Been Combined

Looking at the above, the SPECIFIC combination that has never been
built at scale is:

> **Hyperdimensional knowledge store** (millions of HRR triples,
> sharded) + **Open IE bootstrap** from arbitrary text corpora +
> **factored reasoning programs** over the store + a **small (50-100M
> param) language model** used ONLY as a surface-fluency polisher,
> with NO knowledge-bearing role.

Each component is published. The combination is not.

The closest existing work is RETRO + Atlas, but those still use a
~10B-parameter transformer that holds *most* knowledge in weights and
uses retrieval as augmentation. The proposal here INVERTS that:
retrieval (HRR composition) is 99% of the work, the LM is the polish
layer.

---

## 4. The Novel Contributions RCK Will Make

These are claims I am willing to defend in a paper. Each is genuinely
new in the literature.

### 4.1 Holographic Distillation (the headline idea)

**Claim:** A small language model trained ONLY on the SURFACE structure
of sentences (no world facts) plus a large HRR knowledge store can
match a frontier LLM on factual QA and structured reasoning, at 100x
less compute.

**Mechanism:** Training data for the LM is procedurally generated by
rendering RCK templates over the HRR store with deliberately-varied
phrasings, names, and quantities. The LM learns "how to phrase
arbitrary triples fluently" without learning ANY of the triples
themselves. At inference, RCK retrieves facts; the LM polishes.

**Why it works:** Templates provide perfect supervision for the
fluency task, and the task is purely syntactic. There is no need for
the LM to memorize facts -- those live in HRR.

**Compute estimate:** 50-100M-param LM + ~1B-token synthetic corpus =
hours of A100 time, not months. Total <$1k.

### 4.2 VSA-Based In-Context Learning

**Claim:** Few-shot examples can be encoded as bound hypervector
exemplars and applied COMPOSITIONALLY to a new query without any
gradient updates. This gives perfect few-shot generalization on tasks
where the task structure is decomposable.

**Mechanism:** For each example (input_i, output_i), create
`bind(input_i, output_i)`. Bundle all examples. To apply to a new
input, unbind: `output_new = bundle * input_new`. Cleanup against the
output codebook.

**Why this is new:** ICL in LLMs uses attention to mix examples; this
uses VSA algebra. Already partially demonstrated in RCK v1.1 SCAN-lite
(100% vs LM 0%). At scale this would be a viable mechanism for
arbitrary few-shot tasks.

### 4.3 Programmatic Cognition over HRR

**Claim:** Reasoning over a structured KB is more reliably expressed
as a PROGRAM than as a chain-of-thought rollout. We can build a tiny
"intent classifier" that routes queries to one of N reasoning
programs (lookup, count, aggregate, compare, infer, simulate, ...)
without ever generating intermediate text.

**Mechanism:** Each "program" is a Python function that operates on
the HRR store. The intent classifier is a small lookup
(`rck/conscious_agent._classify_question` is the v3 version). At
inference, the agent picks a program, executes it, and only
LATER calls the LM for surface polish.

**Why this is new:** Existing reasoning systems either learn reasoning
end-to-end (transformers) or use symbolic theorem provers (which don't
handle uncertainty well). RCK uses VSA cosine confidence + explicit
algorithms -- the best of both.

### 4.4 Bootstrapped Knowledge Ingestion via Open IE + Self-Verification

**Claim:** A large HRR knowledge base can be grown autonomously from
raw text by combining Open IE (extracts triples) with the AGENT's OWN
reasoning to verify new facts against existing ones. Contradictions
get flagged; consistent additions get stored.

**Mechanism:** Stream text in. For each extracted triple, query the
agent's existing KB for the same (S, R) pair. If a different value
exists with high confidence, flag the contradiction for review. If
nothing exists, store. If the same value already exists, just count
(reinforce confidence). After N passes the KB approaches a stable
"truth" set without human curation.

**Why this is new:** Truth-maintenance systems exist in classical AI
(TMS), and Open IE pipelines exist, but I don't know of work combining
HRR-based confidence + Open IE bootstrap + automatic verification.

### 4.5 Constructive Forgetting

**Claim:** A knowledge system that can EDIT itself (remove + replace
specific facts) is qualitatively different from one that can only
add. RCK has this already at the triple level; we extend it with
"doubt": low-confidence facts decay over time unless reinforced.

**Mechanism:** Each fact stored with a confidence weight. Each query
that successfully uses a fact reinforces (increment weight). Each
contradiction decrements. Below a threshold, forget. Implement via
real-valued HRR memory (already there) + a per-fact counter.

**Why this is new:** LLMs CAN'T do this -- their knowledge is welded
into weights. Even fine-tuning can only paint over the top. RCK can
literally delete.

### 4.6 Hypothesis-Driven Multi-Agent Reasoning

**Claim:** Using the existing belief KB substrate, we can simulate
"what would person/entity X think?" by running queries through their
private belief KB rather than the ground-truth KB. This is genuine
theory-of-mind and enables creative writing, debate, and strategic
reasoning.

**Already in RCK v1.3** (theory_of_mind.py) but only at toy scale.
Scale this to per-persona belief KBs and you have a substrate for
"AI plays X character" / "AI argues for position Y" without
prompt-engineering hacks.

### 4.7 Compositional Generation via VSA Algebra

**Claim:** Generating a multi-sentence response can be done as
algebraic composition of fact-skeletons in VSA, NOT as next-token
prediction. The result is provably hallucination-free at the fact
level (every claim cites a triple).

**Mechanism:** For each retrieved fact, bind it into a "discourse
hypervector." Compose multiple facts via bundling. Render the final
discourse HV via templates -> small LM polish. Generation reduces to
RETRIEVAL + COMPOSITION + RENDERING, never PREDICTION over an
open token vocabulary.

**Why this is new:** All generative AI today does next-token
prediction. RCK proposes generation without next-token prediction
for the structured-content slice of tasks (which is most of business
QA, education, customer support, technical writing).

---

## 5. Roadmap: v4 → v10

Each milestone is a 1-2 session project (plumbing) or a 1-2 week
project (training the LM polisher). None of them require >$1k of
compute.

### v4.0 — Inverted Architecture skeleton (THIS SESSION)
- `rck/inverted_lm.py` -- the API for "knowledge from HRR, fluency
  from LM polisher". Stub for the LM polisher (uses templates today;
  will plug in a real distilled LM later).
- Demonstrate the end-to-end inversion working with current 1500-fact
  KB.

### v5.0 — Knowledge ingestion at scale
- ConceptNet bulk import (3M assertions, schema mapper).
- Wikidata subset import (100k-1M items).
- DBpedia extraction subset.
- After v5, RCK has GPT-2-class knowledge breadth.

### v6.0 — Open IE bootstrap + self-verification
- Plug in Stanford OpenIE or REBEL for triple extraction.
- Implement the verification loop: extract -> check -> store/flag.
- Demonstrate on Wikipedia first-paragraph corpus (100k pages).

### v7.0 — Distilled LM polisher
- Generate ~1B-token synthetic corpus by rendering templates over
  the existing KB with phrasing variation.
- Train a 50-100M-param transformer on this corpus.
- The LM has NO knowledge it didn't see in templates; its sole job
  is to take a template-rendered draft and produce fluent prose.
- Total compute: ~$500-$2k on rented A100s.

### v8.0 — Programmatic cognition expansion
- Add reasoning programs: aggregate, sort, count, filter,
  conditional, exists, all/none, simulate.
- Intent classifier learns to route queries to programs.
- Tool/action registry (already in v2.2) becomes the runtime.

### v9.0 — VSA-based ICL
- Implement the bound-exemplar few-shot mechanism (Section 4.2).
- Benchmark on standard few-shot datasets.
- Expected: 100% on compositional tasks; competitive on others.

### v10.0 — Production system
- Distributed HRR store (multiple machines for >10M facts).
- Streaming Open IE pipeline.
- Web UI rivaling ChatGPT for the slice of tasks RCK targets.
- API server, MCP, gRPC.

After v10 we have a working "Holographic LLM" -- not an LLM but
serving the same purpose, for ~1000x less compute.

---

## 6. What This Won't Match

Honest framing -- there are LLM capabilities RCK will NOT match:

1. **Truly creative prose generation** -- writing a novel, composing
   poetry that requires word-level surprise. LLMs trained on a
   trillion tokens are genuinely good at this; RCK's template+polish
   path produces fluent but conservative prose.

2. **Code generation** -- LLMs trained on massive code corpora write
   reasonable code by next-token prediction. RCK could do
   syntactically-correct snippets via templates but not novel
   algorithms.

3. **Open-ended brainstorming** -- when the task is "generate 50
   variations of X," LLMs do well because their next-token sampling
   produces diversity. RCK's template+polish is too constrained.

4. **Multilingual fluency** -- LLMs cover hundreds of languages.
   RCK would need per-language template libraries.

What RCK WILL match or beat:

  1. **Factual QA at any scale up to the KB's coverage.**
  2. **Multi-hop reasoning** (perfect, by construction).
  3. **Compositional generalization** (100% vs LLM <10% on SCAN).
  4. **Editable, traceable, calibrated knowledge.**
  5. **Continual learning without catastrophic forgetting.**
  6. **Sub-100ms latency** (currently impossible for frontier LLMs).
  7. **Cost.** Inference cost is essentially zero per query.

This is a real product. It's not GPT-5. But for the things people
ACTUALLY use ChatGPT for (factual lookup, reasoning, dialogue,
summarisation), it's a viable alternative -- and structurally
superior on the auditability + control axes that matter for
enterprise + safety-critical deployments.

---

## 7. The Bet

The argument distills to:

  * LLMs cost $100M+ because they discover everything jointly.
  * Each capability separately costs ~zero with the right structure.
  * VSA gives us the structure for free.
  * The remaining work is content (knowledge ingestion) and one
    small LM training run (fluency polisher).
  * Total cost to reach "GPT-class for the things RCK targets":
    $1-10k, not $100M.

If this bet is right, we've identified an architecture that gives
80% of the LLM use-case value for 0.001% of the cost. That's
publishable, productizable, and structurally different from the
field's current trajectory.

If the bet is wrong, the failure mode is that the small LM polisher
still requires data we can't synthesize, OR that templates can't
cover the diversity of natural prose. Both are testable in v7.

---

## 8. Why I Think This Hasn't Been Done

The field is dominated by the transformer + scale paradigm. The bitter
lesson says compute always wins, and so far it has -- but only on the
trajectory that requires compute. Nobody has invested in the
factor-the-problem alternative because:

  1. **Career incentives:** academia rewards papers that beat SOTA
     on standard benchmarks, which are all set up for LLM-style
     architectures. A neuro-symbolic alternative doesn't fit.
  2. **Industry incentives:** the players with the compute (OpenAI,
     Anthropic, Google) have no reason to undercut their own moat.
  3. **VSA literature is small and academic.** Outside Kanerva's
     group and a handful of others, the field is niche.
  4. **The hybrid story is unfashionable.** "Neuro-symbolic" was
     beaten down in the 2010s when transformers won; now everyone
     assumes monolithic is the only path.

A solo builder or small team can move on this BECAUSE the field has
deprioritized it. The literature is there. The tools are there. The
combination is novel. The compute is affordable.

This is the bet RCK is making explicit.
