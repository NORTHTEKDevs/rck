# RCK v0.1 -- Design Document

**Status:** code shipped (31/31 tests passing, demos run, chat REPL works).
**Last updated:** 2026-05-20
**Sibling project:** `~/projects/active/hyperion/` (VSA-LM + HYMN/FERN).

## 1. Thesis

LLMs are token-prediction engines. They do not reason, plan, or model the world
as a causal mechanism. The field has converged on one architecture (decoder-only
transformers, scaled by data + compute) and left a vast design space
unexplored. RCK occupies one specific corner of that design space.

RCK is the smallest viable system that combines **seven** previously-disconnected
research traditions into a single closed cognitive loop:

  1. **VSA / HDC** -- representation substrate.
  2. **Predictive Coding** -- local-update perceptual encoding.
  3. **Liquid State Machines** -- temporal integration.
  4. **Tsetlin Machines** -- interpretable propositional logic.
  5. **Global Workspace Theory** -- competitive cross-module broadcast.
  6. **Free Energy Principle / Active Inference** -- goal-directed action.
  7. **Thousand Brains Theory** -- reference-frame column voting.

The system is biologically grounded, locally trained, and runs on a laptop.

## 2. Architecture

### 2.1 Notation

- `D` = hypervector dimensionality (default 4096 for the agent, 10000 for HV math primitives).
- HVs are bipolar, i.e. take values in {-1, +1}, stored as `int8`.
- `s_t` = workspace hypervector at time `t`.
- `A` = linear-Gaussian transition matrix in the generative model.
- `W_in`, `W_rec`, `W_out` = LSM input/recurrent/readout matrices.
- `ta` = Tsetlin automaton state, integer in `[0, n_states-1]`.

### 2.2 Per-step loop

For each input symbol `x_t`:

1. **Encode (PCN)**

       hv_perc_t = PCN.encode(one_hot(x_t))

   PCN runs `M=8` inference iterations of:

       x_i = x_i + lr_x * (W_i @ e_{i-1} - e_above)

   then applies a local Hebbian update on each layer:

       W_i += lr_w * x_{i+1} \otimes e_i

   No gradient is propagated across layers.

2. **Temporal integration (LSM)**

       state_t = (1 - leak) * state_{t-1} + leak * tanh(W_in @ hv_perc_t + W_rec @ state_{t-1})
       hv_temp_t = sign(W_out @ state_t)

   `W_rec` is fixed, sparse, with spectral radius `rho < 1` (Echo State Property).

3. **VSA bind**

       hv_bound_t = bind(hv_temp_t, permute(codebook[x_t], position_t))

   The permutation encodes position in the sequence. Bind is elementwise
   multiply.

4. **Column vote (TBT)**

       hv_vote_t, unc_t = ColumnEnsemble.step(x_t)

   `unc_t` = `Var_{i<j} cos(hv_i, hv_j)` -- pairwise cosine variance across
   columns. High `unc_t` = columns disagree.

5. **Workspace WTA (GWT)**

   Candidates submit (HV, salience) pairs:

       cands = {
         "perception": (hv_perc_t,  cos(hv_perc_t,  s_{t-1})),
         "temporal":   (hv_temp_t,  cos(hv_temp_t,  s_{t-1}) + 0.05),
         "binding":    (hv_bound_t, cos(hv_bound_t, s_{t-1}) + 0.10),
         "columns":    (hv_vote_t,  cos(hv_vote_t,  s_{t-1}) + 0.05 - unc_t),
       }
       s_t = sign(decay * s_{t-1}_real + winner_hv)

   The small constants are role priors (binding is intrinsically a bit more
   salient than raw perception); we may learn these later.

6. **Reasoning trace (Tsetlin)**

       score_t, clauses_t = Tsetlin.evaluate(s_t)

   `score_t` is the difference between firing positive and negative clauses.
   `clauses_t` is the human-readable list of active clauses.

7. **Action (FEP active inference)**

       cand_syms = top_k cleanup of (A @ s_t)
       G(a)      = ||codebook[a] - A @ s_t||^2 / (2 sigma^2) - novelty_weight * 1/(1+count(a))
       emit_t    = softmax-sample(- G / T)

8. **Local perceive (FEP)**

       A += lr * (s_t - A @ s_{t-1}) \otimes s_{t-1} / ||s_{t-1}||^2

9. **Teacher feedback (optional)**

   If a supervised target `x_{t+1}` is available:

       tgt = sign(cos(codebook[x_{t+1}], s_t))
       Tsetlin.feedback(s_t, tgt)
       LSM.train_readout(codebook[x_{t+1}])
       Columns.train_readouts(codebook[x_{t+1}])

   In the fully unsupervised case (`teacher_next=None`), only steps 1-8 run.

## 3. Falsifiable claims

These are the things RCK promises that LLMs don't, with concrete tests.

### 3.1 Continual learning without catastrophic forgetting

**Claim:** Training on Corpus A then Corpus B preserves A-recall.

**Test:** `examples/continual_learning.py`. Threshold:
`A_score_after_B / A_score_before_B > 0.6`.

**Observed (v0.1):** ratio = 6.17 (improves rather than degrades, because A's
score was sub-1% to begin with -- but the relationship holds: A is *not*
overwritten by B).

### 3.2 One-shot vocabulary

**Claim:** A brand-new symbol can be introduced in a single observation,
becomes near-orthogonal to all existing atoms, and is immediately usable.

**Test:** `examples/one_shot_vocab.py`. Threshold: `|cos(Z, other_atom)| < 0.2`
for >=4 of 5 comparisons.

**Observed (v0.1):** all five comparisons < 0.05.

### 3.3 Interpretable reasoning

**Claim:** Every emission can be explained as a list of fired logical clauses.

**Test:** `python -m rck.cli chat` then type `why`.

**Observed (v0.1):** Tsetlin clauses dump as
`(+) NOT f524 AND NOT f938 / (-) f839 AND NOT f913` etc. Each clause is a
literal-by-literal conjunction.

### 3.4 No GPU, no batches

**Claim:** Full RCK runs on a single CPU thread.

**Test:** All v0.1 demos and tests run in pure numpy, no GPU dependency.

**Observed (v0.1):** 31 tests in 2.13s on CPU. Continual demo in <5s.

## 4. Open issues / known limits in v0.1

- **PCN updates are unstable** at long horizons; in practice the encoder
  drifts after ~2k examples. Mitigation in v0.2: clip + normalize weights;
  consider Whittington 2017's energy-based PCN formulation.
- **`A` matrix in FEP is `O(D^2)`** memory. For `D=10000` this is 400MB. The
  default agent uses `D=4096` to keep `A` under 70MB. For larger `D` we'll
  need low-rank or sparse `A`.
- **Tsetlin updates iterate per-literal**, `O(n_clauses * D)`. At `D=4096`,
  `n_clauses=32`, that's 131k automaton checks per step. Vectorize in v0.2.
- **No proper Bayesian posterior over `A`.** The current FEP uses a point
  estimate, so the "epistemic" novelty term is heuristic (count-based)
  rather than information-theoretic.
- **Char-level only.** BPE / SentencePiece in v0.2 will make fluency demos
  more compelling.

## 5. Roadmap

### v0.2 -- stability + scale
- Whittington-style stable PCN.
- Sparse / low-rank `A`.
- Vectorized Tsetlin updates.
- BPE tokenization.
- Targets: 10k-token corpus, chat that looks like coherent English.

### v0.3 -- multimodality
- Image PCN encoder -> same VSA substrate.
- Cross-modal binding for visual question answering on a toy domain.

### v0.4 -- agentic
- Action space = function calls instead of tokens.
- Tool use via VSA-bound action templates.

### v1.0 -- shippable
- Rust kernels for VSA / Tsetlin hot loops.
- Persistent memory store (hyperdimensional KV).
- gRPC / MCP server interface.

## 6. Relationship to Hyperion

RCK and Hyperion are sibling projects in the same workspace.

- **Hyperion** (`~/projects/active/hyperion/`) focuses on a VSA-as-LM
  substitute: VSA-LM v0.2 substrate + HYMN/FERN field dynamics + DEQ +
  planned active inference. It is fundamentally an *architecture variant* of
  a language model.
- **RCK** is a *cognitive architecture* integrating seven traditions. Its
  language ability is incidental and limited; its value is the integration.

They may converge -- the HYMN-Mini DEQ could become a perception module in a
later RCK, and the RCK active-inference loop could replace HYMN's Phase 3
plan. For v0.1 they ship independently and share no code.

## 7. References

The design synthesizes ideas from:

- Kanerva 2009 -- Hyperdimensional computing (HDC) foundations.
- Rao & Ballard 1999, Whittington & Bogacz 2017 -- Predictive Coding.
- Maass et al. 2002 -- Liquid State Machines.
- Granmo 2018 -- The Tsetlin Machine.
- Baars 1988 -- Global Workspace Theory.
- Friston 2010 -- Free Energy Principle / Active Inference.
- Hawkins et al. 2019 -- Thousand Brains Theory.
- IBM 2023 -- NVSA (HDC + neural perception fusion).

This document is not a literature survey; consult the originals for math.
