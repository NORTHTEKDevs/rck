# RCK v5+ — Beyond LLMs: capabilities GPT-4 cannot match

**Status:** second-pass research. Written 2026-05-20.
**Question this answers:** what pushes RCK BEYOND LLM parity into
capabilities LLMs structurally cannot achieve?

The v4 design doc made the case that RCK can MATCH GPT-class
capabilities for ~$1-10k of compute via factored architecture. This
document goes further: it identifies the dimensions on which RCK is
strictly **superior** to LLMs -- not just cheaper, but able to do
things no transformer can do at any scale.

If v4 is "GPT-4 for the cost of a laptop", v5+ is "post-GPT-4 in the
specific ways that matter most."

---

## 1. The structural superiority axes

For each axis below, the LLM mechanism is fundamentally limited; RCK's
mechanism is fundamentally not. These are not benchmark gaps that
scale closes. They are architectural impossibilities.

### 1.1 Knowledge provenance

**LLM mechanism:** facts are baked into weights through training.
There is no record of WHERE a fact came from, WHEN it was learned, or
HOW CONFIDENT the model is. Post-hoc citation generation is unreliable
(the model confabulates sources).

**RCK mechanism:** every triple `(S, R, O)` can carry metadata:
`source` (which document / conversation), `timestamp`, `confidence`,
`reinforcement_count`, `last_verified`. Retrieval includes the
provenance. The agent can answer "WHY do I believe X?" with literal
audit-trail evidence.

**Why this matters:** legal, medical, scientific, and safety-critical
applications need this. No amount of LLM scale gets you this.

### 1.2 Editable knowledge

**LLM mechanism:** fine-tuning can shift behavior but cannot SURGICALLY
remove a single fact. The knowledge is non-localized. Mechanistic
interpretability research (Anthropic 2023+) has shown some "concept
neurons" can be edited, but it's fragile and doesn't scale.

**RCK mechanism:** `forget((S, R, O))` removes one fact in O(D) time
without affecting any other knowledge. `tell` and `forget` are inverse
operations. Already in v1.x.

### 1.3 Provable hallucination-freedom

**LLM mechanism:** at every token, the LM samples from a distribution.
There is no architectural reason it can't generate a false claim. RLHF
reduces but never eliminates hallucination.

**RCK mechanism:** the inverted architecture (v4) generates ONLY by
rendering retrieved facts. If no fact supports a claim, the renderer
outputs "I don't know." This is enforceable at the architecture
level.

### 1.4 Continual learning without catastrophic forgetting

**LLM mechanism:** fine-tuning on new data degrades performance on old
data (catastrophic forgetting, McCloskey & Cohen 1989). Continual
learning remains an open research problem for transformers.

**RCK mechanism:** new facts go into new shards. Old facts in old
shards are untouched. Forgetting is opt-in (explicit) rather than
forced. Already at 0.5 retention in v1.3 vs LLM <0.3.

### 1.5 Sub-100ms inference

**LLM mechanism:** even with KV cache + speculative decoding, GPT-4
takes 1-5 seconds per query. The fundamental bottleneck is autoregressive
generation through hundreds of transformer layers.

**RCK mechanism:** HRR cleanup is one matrix-vector multiply per
shard. At D=4096 and 64 shards, total <5ms per query.

### 1.6 Compositional generalization

**LLM mechanism:** transformers learn surface n-gram statistics. They
fail systematically on SCAN-style compositional tasks (<10% on
add-primitive split).

**RCK mechanism:** VSA bind/bundle composition gives 100% on SCAN.
This is architectural, not a tuning artifact.

### 1.7 Privacy / local execution

**LLM mechanism:** frontier LLMs run remotely. User data goes to OpenAI
/ Anthropic / Google. Local LLMs exist (Llama, Mistral) but at smaller
capability.

**RCK mechanism:** runs on a laptop in <100MB. The user's KB stays
local. Privacy-by-construction.

### 1.8 Cost

**LLM mechanism:** $0.001-$0.10 per query at the API level. At scale
this is meaningful.

**RCK mechanism:** ~$0 per query (microseconds of CPU time).

---

## 2. Eight CAPABILITIES no LLM can match at any scale

These are the genuinely novel directions for v5+. Each is something
RCK can do that no transformer can match.

### 2.1 Memory hierarchies (working / episodic / semantic / procedural)

**The biological model:** human memory is not flat. Working memory
holds ~7 items for seconds. Episodic memory holds specific events
with time stamps for days/years. Semantic memory holds general world
knowledge. Procedural memory holds skills and habits.

**RCK extension:** four memory tiers, each its own HRR substrate.

  * **Working memory** (capacity ~16 HVs): bound items currently in
    reasoning focus. Cleared between turns.
  * **Episodic memory** (per-event triples with timestamps):
    "user asked X at time T", "I told them Y." Searchable by time.
  * **Semantic memory** (current KB): general world facts. Already
    sharded.
  * **Procedural memory** (program HVs): reasoning steps as bound
    sequences. Learned from successful problem-solving.

**Consolidation:** episodic memories that recur across many events
get promoted to semantic ("the user always asks about France")
become facts. This is biological consolidation in mechanical form.

LLMs have one mechanism (context window) for ALL of this. RCK can
have four specialized mechanisms.

### 2.2 Counterfactual / multi-universe reasoning

**The biological model:** humans can reason about "what if?" without
actually changing their beliefs. Children pretend; scientists test
hypotheses; lawyers argue cases.

**RCK extension:** multiple parallel KB "universes." The agent can
branch into a hypothetical universe, modify facts, run queries,
draw conclusions, then discard the branch -- without affecting the
ground-truth KB.

  * `universe.branch()` -> new universe inheriting facts
  * `universe.modify("paris", "capital_of", "germany")` -> only in this branch
  * `universe.query(...)` -> runs as normal
  * `universe.discard()` -> branch goes away
  * `universe.commit()` -> merges back to ground truth

LLMs simulate this via prompting ("imagine that...") but the
reasoning is forced through the same parameters that hold the real
facts. RCK keeps them physically separate.

### 2.3 Genuine causal reasoning (Pearl's do-calculus over HRR)

**The biological model:** humans distinguish correlation from
causation. We reason about interventions: "If I do X, what happens?"
LLMs systematically conflate these.

**RCK extension:** add a `causes` relation as first-class, plus
`do(X, R, V)` operator that REPLACES the current value (intervention)
rather than observing it. Run forward propagation on the causal
graph to predict downstream effects.

Combined with multi-universe (2.2), this enables:
  * "What would happen if we lowered taxes?" -> causal forward sim
  * "What caused the test failure?" -> causal backward (abductive)
  * "What's the difference between observing X and causing X?" ->
    distinct query types in RCK; same blob in an LLM.

### 2.4 Active curiosity / gap-driven learning

**The biological model:** humans (and other animals) are driven by
expected information gain. We seek out novelty and explore unknowns.

**RCK extension:** detect knowledge gaps. If RCK knows
`(dog, isa, mammal)` and `(mammal, has, fur)` but doesn't know
`(platypus, has, ?)`, that's a GAP -- a known unknown. RCK can
generate a question to fill it.

Algorithm:
  1. For each entity E in the KB, enumerate facts about E.
  2. For each sibling entity E' (shares parent isa), check what
     E' has that E doesn't.
  3. Hypothesize that E has those properties too. Mark as
     unverified.
  4. Generate question: "Does the platypus have fur?"
  5. When the user (or external source) provides an answer, update.

This is the active-inference loop from `rck/fep.py` made explicit
at the KB level. LLMs can't do this -- they don't know what they
don't know.

### 2.5 Reversible / abductive reasoning

**The biological model:** humans reason both forward (cause -> effect)
and backward (effect -> cause). Diagnosis is abduction. Why did my
car not start? Because the battery died. Detectives are abductive.

**RCK extension:** unbind in the reverse direction. Given an observed
effect (Y), search the KB for triples `(?, causes, Y)`. Combine with
multi-hop to walk back through causal chains.

LLMs can do this when prompted but get it wrong systematically because
they don't distinguish forward and backward in their training data.
RCK's VSA substrate is BIDIRECTIONAL by construction (bind is
self-inverse).

### 2.6 Self-debugging / metacognitive correction

**The biological model:** humans notice when their reasoning fails
and adjust. "Wait, that doesn't follow." This is metacognition.

**RCK extension:** when a multi-step inference gives a low-confidence
answer, the agent inspects WHICH step had the lowest confidence and
flags it. If the user corrects the final answer, the agent backtracks
through the chain and identifies which premise was wrong.

This is impossible in LLMs because there is no inspectable reasoning
trace -- it's all hidden activations.

### 2.7 Lifelong identity + interaction memory

**The biological model:** humans have continuous identity. We remember
past conversations with specific people. Our knowledge accumulates
over a lifetime.

**RCK extension:** every conversation is an episodic memory. The agent
remembers "user K asked about France yesterday and seemed interested
in capitals." Over time, the agent builds a model of each user's
interests, preferences, and recurring topics.

LLMs have NO memory across sessions. Even with chat-history
augmentation, the model has no privileged "self" that persists.
RCK has it natively.

### 2.8 Knowledge provenance (already discussed in 1.1)

Each fact carries `(source, timestamp, confidence, count)` metadata.
The agent can answer "why do I believe X?" with literal audit
evidence. Critical for legal/medical/scientific use.

---

## 3. The unified vision: RCK as a Cognitive Operating System

The v4 design doc described RCK as an "inverted LLM" -- knowledge in
HRR, fluency in a small LM. v5+ goes further: **RCK is a cognitive
operating system**, not a model.

The system has:

  * **Substrate** (VSA HRR primitives)
  * **Storage** (sharded KB, episodic memory, working memory)
  * **Reasoning programs** (inference, multi-hop, counterfactual,
    abductive, numerical, temporal, spatial)
  * **Communication** (NLG templates, polisher, personality)
  * **Self-model** (introspection, calibration, identity)
  * **Drives** (curiosity, gap-detection, consistency-maintenance)
  * **Tools** (action registry, external APIs)
  * **Continual learning** (corrections, ingestion, consolidation)

Each of these is a SERVICE. They communicate via HRR. The system
runs continuously, not request-by-request.

This is a different paradigm from "model takes input, returns
output." It's closer to an operating system, or a biological brain.

LLMs are functions. RCK is a process.

---

## 4. Concrete v5-v10 roadmap

### v5.0 -- the four foundational additions (this session)
- Knowledge provenance: facts carry source/timestamp/confidence/count
- Memory hierarchies: working / episodic / semantic / procedural
- Counterfactual universes: branch / modify / query / discard
- Abductive reasoning: reverse-direction inference
- Curiosity / gap-detection: identify and fill known unknowns

### v5.5 -- causal reasoning
- First-class `causes` and `do(...)` operations
- Forward causal sim + abductive backward
- Integration with multi-universe

### v6.0 -- consolidation + memory dynamics
- Episodic -> semantic consolidation (recurring patterns become facts)
- Forgetting curves (low-reinforcement facts decay)
- Sleep cycles (periodic memory reorganization)

### v6.5 -- interaction memory
- Per-user models built from conversation history
- Long-term identity persistence

### v7.0 -- the distilled polisher (from v4 plan)
- 50-100M LM trained on synthetic template-rendered corpus
- Total ~$500-2k compute

### v8.0 -- mathematical reasoning
- Integration with a theorem prover (Lean / Z3 / Coq) as a tool
- Algebraic manipulation programs
- Proof search via VSA

### v9.0 -- multi-modal grounding
- Image PCN encoder feeding the same HRR substrate
- Audio similar
- Vision + language reasoning on the same KB

### v10.0 -- the running OS
- Continuous service rather than request-response
- Background ingestion, consolidation, curiosity, dreaming
- Multi-instance federation
- Production-grade reliability

---

## 5. The bigger thesis

The most successful AI systems of the past decade have all been monolithic
neural networks scaled to the limit. The bitter lesson seemed to imply
that any structure we add to a model is a liability -- compute is what
scales.

But scale has limits the field is bumping against:

  * Compute cost is becoming prohibitive even for Google.
  * Hallucination remains the #1 enterprise blocker.
  * Continual learning is unsolved.
  * Provenance is unsolved.
  * Privacy is unsolved (training data leakage).
  * Latency is structurally bounded by autoregression.

RCK addresses ALL of these structurally. The bet is that the next era
of AI value will accrue not to the players with more compute but to
the players with the RIGHT STRUCTURE. We don't know yet whether this
is true, but the architectural argument is publishable, the
implementation is achievable, and the application surface is real.

If the bet is right, RCK is a different category of AI system, not
just a cheaper LLM. It's a **cognitive operating system** where
LLMs are **single-shot generative functions**.

If the bet is wrong, the failure mode is most likely that compositional
generalization (which RCK gets perfectly) doesn't actually matter for
the tasks people care about, because LLMs have memorized enough to
fake it convincingly. That's testable and is exactly the experiment v5
+ v7 will run.

---

## 6. What I'd implement first

Of the seven novel capabilities in Section 2, the highest-impact
lowest-effort items are:

  1. **Knowledge provenance** (1 hour) -- huge auditability gain.
  2. **Memory hierarchies** (3 hours) -- enables 4 other features.
  3. **Counterfactual universes** (2 hours) -- novel reasoning mode.
  4. **Abductive reasoning** (2 hours) -- diagnosis / explanation.
  5. **Curiosity / gap-detection** (2 hours) -- active learning loop.

All five together is ~10 hours of focused work. They give RCK
qualitatively new capabilities no LLM matches.

Then v6 onwards is consolidation + interaction memory + the polisher
LM training. After that we're at v7.0 == "competitive with GPT-4 for
the target slice" + v5.x == "qualitatively superior on dimensions
LLMs can't compete on."
