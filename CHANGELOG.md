# Changelog

## 15.2.0 — 2026-07-11

Calibrated-confidence + honest-benchmarks release. Driven by a
full adversarial audit of every empirical claim (each §5 number
re-executed fresh) and a prior-art audit of every novelty claim.

### Added
- **Score calibration** (`rck/score_calibration.py`):
  `ScoreCalibrator` fits an isotonic (pool-adjacent-violators)
  mapping cosine → empirical P(correct), with Beta(1,1)-smoothed
  block probabilities so it never claims certainty. Raw cosines are
  badly miscalibrated as probabilities: held-out Brier 0.5683
  (raw-cosine-as-probability) vs 0.0004 (calibrated) on the
  commonsense KB.
- **`calibrated_product` propagation rule**
  (`PropagationConfig(rule="calibrated_product", calibrator=...)`):
  multiplies calibrated per-hop probabilities, restoring real
  multiplicative semantics to chain confidence — length-sensitive
  (a clean 50-hop chain reports 0.979, decaying with every hop)
  without the raw product's collapse or the geometric mean's
  length-invariance.
- **Live-relation index** (`RelationIndex` in `rck/knowledge_base.py`,
  the optimization §5.4 of the v15.0 paper promised): a snapshot of
  which relations have facts in which shards, built from the shards'
  fact logs. BFS chain discovery skips HRR queries that could only
  return crosstalk: 31%/51%/59%/67% of queries cut across the four
  KB tiers, 1.5×/2.1×/2.2×/3.0× wall-clock speedups, identical
  discovered chains. `kb.query(...)` gains an optional
  `shard_subset` parameter; `discover_chains` gains
  `relation_index` / `use_relation_index` (default on;
  `use_relation_index=False` reproduces v15.1 search exactly).
- **`scripts/confidence_calibration_study.py`**: fits and evaluates
  the calibrator (train: 4,862 labeled retrievals across five bundle
  loads; eval: held-out commonsense KB), writes
  `data/confidence_calibration_study.json` including the fitted
  calibrator (`ScoreCalibrator.from_dict`).
- 27 new tests (13 relation-index/subset, 14 calibration). 719 → 746.

### Changed
- **`scripts/chain_discovery_study.py`** rewritten: the paper's old
  three-KB latency table (14.5/7.7/55.5 ms on 716/2,599/4,109
  facts) was not reproducible — the mid-size KBs no longer exist in
  the repo and fresh runs measured 24–139 ms with ~4× run-to-run
  variance. The study now sweeps the four KB tiers actually shipped
  (716 → 7,080 facts), reports per-probe medians of repeats plus
  machine-independent HRR-query counts, and A/Bs the live-relation
  index.
- **Papers corrected and synced.** `paper.md` had never received the
  v15.1 retraction content and still presented the retracted
  four-gate/"~87%" study as current — it is now fully synced with
  `paper.tex`. Both papers additionally fix: the abstract's "50+
  hops at moderate-or-better" claim (measured: 30+ at
  moderate-or-better, argmax-correct to 50; the old sentence
  conflated a confidence threshold with retrieval correctness);
  §5's cascade numbers (measured: +6 facts in rounds [2,2,2,0],
  not "~11" with "round 2 out-producing round 1"; rule cascade
  718 → 843 per the v14 demo, not 718 → 839); stale test/line
  counts; and the §5.8 reproducibility table (now maps every
  section to the script that actually regenerates it).
- **Related work rewritten** with the closest prior art the papers
  previously omitted: SPA/Spaun and WordNet-scale HRR (Eliasmith
  et al.; Crawford et al.), Holographic Declarative Memory, truth
  maintenance systems (Doyle; de Kleer) + AGM + MYCIN, AMIE/PRA/NELL
  and type-constrained KG completion, the negative-knowledge
  literature (Arnaout et al.; NegatER; Clark; Gelfond-Lifschitz),
  confidence-combination prior art (PSL, MLN, subjective logic,
  logarithmic pooling), federated HDC, HolE, NVSA, torchhd, and
  Lenat & Marcus. 29 references added.
- **"Hallucination-free" scoped** in a new §1.2: structurally free
  of generative confabulation ≠ incapable of holding a false belief
  (the v15.1 shark/blue retraction is cited as the in-house
  counterexample), with the legal-AI audit study and the
  "reduce-not-eliminate" survey cited as the definitional trap
  being avoided.
- **Packaging**: distribution renamed `rck` → `rck-kernel` (the
  PyPI name `rck` belongs to an unrelated bioinformatics package
  since 2017, so `pip install rck` never installed this project).
  Import name stays `rck`. README install instructions now use the
  git URL.
- **`data/*.json` study outputs are now tracked** in git. They were
  gitignored, which silently broke the paper's promise that readers
  could inspect `data/chain_induction_failures.json` etc. — the
  public repo shipped no study artifacts at all.

### Fixed (public-readiness pass)
- **Base installs work now.** `pip install -e ".[dev]"` + `pytest`
  previously failed collection: `rck.curriculum` imported
  `rck.polisher` (→ torch) via a dead, unused import — a leak of the
  optional [polisher] extra into the numpy-only substrate. Removed;
  the mcp/polisher test modules now `importorskip` their extras, so
  the base suite runs green without torch or mcp. CI gained a
  full-extras job alongside the base-install matrix (now also
  3.13/3.14).
- `rck.__version__` was stuck at "15.0.0"; now matches pyproject
  (enforced by a test), and the package docstring described the
  archived v1.x generative prototype instead of v15.
- `ScoreCalibrator.prob(NaN)` silently returned the
  HIGHEST-probability block (numpy sorts NaN above everything); now
  maps to the lowest.
- `shard_subset` with out-of-range indices raised a bare IndexError
  (or silently wrapped negatives); now a clear ValueError.
- README 60-second demo comment showed a stale score (0.42 vs the
  real 1.00); README pointed at a nonexistent
  `docs/guide/07-ingestion.md`; README misdescribed the v15.0
  retraction as a "100% precision" claim (it was "~87%").
- docs/guide/02: the symmetrisation example used `capital_of`, which
  is not in `INVERSE_PAIRS` (it's `capitalof`) and therefore never
  symmetrised; docs/guide/06: the `record_truth` example lacked the
  prior `ask_with_idk` episode it needs; docs/guide/07: "~7k lines"
  claim updated to reality.
- paper.md gained a rendered References section (36 works — inline
  citations previously pointed at a bibliography md readers couldn't
  see) and the 88.7% analogy baseline is now attributed (historical
  v15.0 argmax-solver number, not re-measured by the shipped
  script). paper.tex: stale `\date`, one unprefixed script path.

### Migration
- None required. Default propagation remains `geometric_mean`;
  `calibrated_product` is opt-in. Discovery behavior is unchanged on
  real chains; edges that existed only as cleanup noise are no
  longer followed (pass `use_relation_index=False` for the exact
  v15.1 search).

### Tests
719 → 746 passing (full extras; base installs skip the mcp/polisher
modules).

## 15.1.0 — 2026-05-24

Chain-induction honesty release. A failure-mode audit of the v15.0
study showed that the headline "~87% chain-induction precision"
measured HRR-roundtrip stability rather than semantic correctness:
~88% of "verified" inductions were `implies`-relation fallbacks, some
of which encoded false type-confused claims like
`(shark, implies, blue)` via `[habitat, color]` or
`(elephant, implies, tree)` via `[has, partof]`. v15.1 retracts that
framing and ships the corrected pipeline.

### Added
- **No-meaningful-relation gate**: chains whose first hop is not a
  lifting relation and which are not same-relation transitive are
  now rejected rather than stored under a generic `implies` label.
  `InductionPolicy.allow_generic_implies` (default False) opts back
  into legacy behaviour.
- **Type-signature gate**: each lifting-relation family permits only
  a specific set of last-hop relations. `partof` transfers
  `{locatedin, madeof, usedfor}` (the part inherits the whole's
  location, material, use), but NOT `has` (a wing doesn't have what
  the bird has). `locatedin` transfers only
  `{continent, country, region, city}`, not `size`/`color`/`isa`.
  `isa`/`class`/`category` siblings transfer
  `{isa, has, kind, class, category}`. Encoded as `isa_transfers`,
  `partof_transfers`, `locatedin_transfers` fields on
  `InductionPolicy`.
- **Lifting-relations allowlist expanded** to include `class`,
  `subclass`, `subtype`, `category` (in addition to the previous
  `isa, partof, locatedin, memberof, instanceof, kind`).
- **`scripts/chain_induction_failure_analysis.py`**: failure-mode
  diagnostic that captures every induction outcome by bucket and
  writes per-record JSON to `data/chain_induction_failures.json`.
  This is the reproduction harness for the new §5.1 numbers.
- New tests:
  `test_induce_rejects_non_lifting_chain_by_default`,
  `test_induce_allows_generic_implies_when_opted_in`,
  `test_induce_rejects_partof_has_chain`,
  `test_induce_rejects_locatedin_size_chain`,
  `test_induce_partof_locatedin_chain_passes`,
  `test_induce_lifting_via_class_forwards_relation`.

### Changed
- Paper §5.1 ("Chain-induction precision") rewritten with honest
  decomposition: explicitly retracts the 87% headline, presents the
  six-gate pipeline, and reports 31/31 manually-validated inductions
  on the 400-probe study with zero HRR self-verify failures (an 8.4%
  yield on attempts; the 92% rejection rate is the policy correctly
  refusing to ascribe meaning to non-type-coherent chains).
- Abstract bullet and §1 "What is new here" updated to match.

### Migration
- Callers that depended on `implies`-fallback behavior need to set
  `InductionPolicy(allow_generic_implies=True)` explicitly. All
  in-tree callers already use the new default and were unaffected.

### Tests
714 → 719 passing.

## 15.0.0 — 2026-05-21

The "product-shaped reasoning stack" release. Negation, persistence,
multi-agent merge, counterfactuals, analytics. ~30 new modules, 313
new tests on top of v13 (401 → 714 passing). Stable public API.

### Added
- **Negative facts** (`rck.negative_facts`): `agent.deny(s, r, o)`
  stores `NOT_R`; `filter_negatives` drops denied candidates;
  contradiction detection extended to surface positive-vs-negative
  collisions; chain induction and rule instantiation both respect
  stored denials.
- **Negation propagation** (`rck.negation_propagation`): when
  `(mammal, NOT_has, feathers)` and `(cat, isa, mammal)` are stored,
  `propagate_negations()` derives `(cat, NOT_has, feathers)`.
- **Hierarchical abstraction** (`rck.hierarchical_abstraction`):
  when N siblings of an `isa` parent share the same `(R, O)`, lift
  to `(parent, R, O)`.
- **Causal reasoning** (`rck.causal`): `downstream_effects(cause)`
  and `root_causes(effect)` BFS over the `causes` relation.
- **Counterfactual context manager** (`rck.counterfactual`): add
  temporary facts inside a `with` block; auto-rollback on exit.
- **Multi-agent**: `agent.merge_from(other)` federated merge,
  `consensus.majority([agents], ...)` for vote / confidence
  aggregation, `agent.diff_with(other)` for state comparison.
- **Persistence**: `agent.save_state(dir)` writes skills, provenance,
  query memory to JSONL; `load_state(dir)` restores them.
- **Analytics**: `concept_density`, `relation_cooccurrence`,
  `entity_similarity`, `subject_importance`, `gap_detection`,
  `skill_clustering`, `skill_promotion`, `shard_balance`.
- **Episodic memory** (`rck.query_memory`): every `ask_with_idk`
  logged; drift detection per-call (`drift_from_prior`) and
  aggregate (`agent.drift_report()`); `record_truth(...)` ties
  ground truth into CalibrationTally.
- **agent.maintain v2**: eight-phase nightly pass (cascade_induct →
  cascade_instantiate → propagate_negations → resolve_conflicts →
  promote_skills → consolidate_episodes → warm_cache_from_history
  → optional checkpoint). Single call.
- **Preview helpers**: `agent.what_changes(facts)`,
  `agent.what_if_user_says(text)`, `agent.delta_replay(facts)` --
  all rollback-safe.
- **Surface helpers**: `agent.summarize_subject(s)`,
  `agent.status_report()`, `agent.canonicalize(text)`,
  `agent.source_calibration()`, `agent.rule_effectiveness_report()`.

### Changed
- Bayesian softmax is the default analogy combiner (returns
  calibrated probabilities; argmax unchanged).
- Chain cache is versioned; bulk writes / induce / instantiate /
  conflict-resolve auto-bump the version.
- ProvenanceRecord gained a `derivation` field that
  `chain_induction` and `rule_instantiation` populate, enabling
  recursive `explain_why`.
- `instantiate_rule` now handles N-clause rule bodies (was 2 only).
- Rule extraction inherits the same filters as chain_induction
  (inverse-pair, non-transitive same-rel, lifting-rel) for
  consistency.

### Fixed
- `chain_cache` invalidation on bulk_load.
- Worker count auto-tuning for batch_discover (caps at 4 per the
  empirical study).

## 14.0.0 — 2026-05-21

The "chain reasoning + induction" release. Built on the v13 work
into a coherent stack: chains, rules, provenance graph,
explain-why. 401 → 553 tests.

### Added
- **Chain walker** (`rck.chain_walker`): generic n-hop execution
  with `Hop(forward|reverse)` and confidence propagation.
- **Chain discovery** (`rck.chain_discover`): BFS over the
  HRR-KB with goal predicates.
- **Chain induction** (`rck.chain_induction`): confident chains
  become direct edges. The filter stack (inverse-pair,
  non-transitive same-relation, lifting-relation, intermediate-cycle)
  guarantees precision.
- **Cascading induction** (`rck.cascading_induction`): iterate to
  fixed point.
- **Rule extraction / instantiation / cascade / composition**:
  symbolic universal rules derived from chain patterns; forward
  application; compose to longer rules.
- **Analogical reasoning** (`rck.analogy`): A:B::C:? via two
  relational queries. 88.7% → 93.9% accuracy after confidence-
  weighted joint scoring; chain fallback for indirect analogies.
- **Set reasoning** (`rck.set_reasoning`): intersect / union /
  difference across multiple constraints.
- **Contradiction detection** + **belief revision**: functional-
  relation conflicts surfaced; resolution by source priority.
- **Explainability** (`rck.explain_why`): recursive provenance
  graph traversal with cycle breaking.
- **Skill library** (`rck.skills`) records every successful chain;
  used as a relation-priority prior in chain_discover.
- **IDK detection** (`rck.idk_detection`): explicit
  KNOWN/AMBIGUOUS/IDK classifier.
- **Confidence calibration** (`rck.confidence_calibration`):
  per-provenance-source discount factors.
- **Chain memoization** (`rck.chain_cache`).
- **Parallel batch discovery** (`rck.parallel_discover`).

### Changed
- Default confidence propagation rule changed from `product` to
  `geometric_mean`. Extends usable chain depth from ~10 hops to
  50+ hops on synthetic linear chains.

### Findings documented
- `docs/design/v13-chain-depth-finding.md`
- `docs/design/v13-chain-induction-finding.md`
- `docs/design/v13-cascade-induction-finding.md`
- `docs/design/v13-chain-discovery-finding.md`
- `docs/design/v13-analogy-finding.md`
- `docs/design/v13-skill-prior-speedup.md`
- `docs/design/v13-cross-shard-chain-finding.md`
- `docs/design/v13-sparse-substrate-finding.md` (negative result)
- `docs/design/v14-narrative.md` (architecture rollup)

## 13.0.0 — 2026-05-21

The "scale and self-verification" release.

### Added
- Auto-shard sizing (`rck.shard_sizing.recommend_shards`).
- Self-verification loop (`rck.self_verify`): roundtrip / reverse /
  sibling-sanity.
- Skill discovery library (`rck.skills`).
- Cross-shard union retrieval with evidence pooling.
- Substantial polisher training (20k pairs, 2000 steps, loss
  7.0 → 0.62 in 81s CPU).

## 12.0.0 — 2026-05-21

Empirical capacity study, dreaming module, active learning,
evaluation framework, identity store, adversarial test set,
sparse-HRR experiment (later proven unfit for substrate use).

## 1.5.0 — 2026-05-20

The "scale + introspection" release. Bulk knowledge ingestion, inverse
relation auto-symmetrization, multi-sentence response composition,
synonym normalization, think-aloud chain-of-thought, conversation
persistence. KB grew 486 -> 1033+ facts across 12 domains.

### Added
- **`rck/bulk_ingest.py`**: `bulk_load_jsonl` / `bulk_load_csv` /
  `bulk_load_triples`. Auto-symmetrizes inverse relations (wrote/author,
  capital/capitalof, locatedin/contains, etc.). 1000 facts in <5s.
- **`rck/synonyms.py`**: relation aliases (`hue -> color`, `writer -> author`)
  and entity aliases (`UK -> england`, `USA -> usa`).
- **`rck/compose_answer.py`**: `describe(entity)` produces multi-sentence
  descriptions from multiple stored relations.
- **`rck/think_aloud.py`**: `narrate(question, infer_result)` renders the
  chain-of-thought as a natural-language explanation.
- **`rck/session.py`**: `save_session(agent, path)` and `load_session`
  for cross-run conversation persistence. Full KB + beliefs + dialogue
  state + calibration restored.
- **`scripts/make_extended_kb.py`** + **`data/extended_kb.jsonl`** (547
  facts): chemistry (38 elements + symbols + atomic numbers),
  astronomy (8 planets + properties), geography (mountains, rivers,
  oceans, seas), historical figures (32+), languages (24), occupations
  (24), instruments (18), sports (15), vehicles (20).
- **`examples/breadth_demo.py`**: live demo across all 12 domains.
- **`docs/design/RCK-v2.0-PATH-TO-LLM-PARITY.md`**: roadmap document
  arguing how RCK can rival LLMs without comparable training compute.
- 15 new tests in `test_bulk_ingest.py` and `test_v15.py`.

### Changed
- `ConsciousAgent.ask` now normalizes the entity + relation via the
  synonym table before querying the KB. "What is the hue of the sky?"
  routes to the same fact as "What is the color of the sky?".
- `ask(think_aloud=True)` returns a `think_aloud` field with the
  chain-of-thought narration.
- `describe()` covers 30+ relations (was 14) so entities like
  shakespeare, gold, earth produce useful multi-sentence summaries.

### Honest framing for "rivaling LLMs"
- 1033 hand-curated facts is small-LM territory, not GPT-class. But
  the substrate is now wired to ingest at scale; the next-session
  goal is ConceptNet + Wikidata bulk import (millions of triples).
- The v2.0 design doc lays out the explicit argument for why the
  factored-architecture approach can match LLM coverage without
  comparable compute. The bet: knowledge / grammar / reasoning /
  format are each free to ENCODE; LLMs spend compute because they
  LEARN them from raw next-token prediction.

## 1.4.0 — 2026-05-20

The "actually conversational" release. Multi-hop inference, natural-language
output, broader question types (boolean / enumeration / comparison), and
multi-turn dialogue context. RCK now handles essentially all the question
shapes a user would normally throw at an LLM, on a 450+ fact KB, at
sub-100ms latency.

### Added
- **`rck/inference.py`**: depth-limited chain inference engine.
  - `infer(kb, S, R)` returns answer + chain + source ("direct" /
    "inherited" / "transitive").
  - Inheritance: walks `isa`, `kind`, `category`, `locatedin`, `lives_in`,
    `partof` as parent-relations. Cities inherit attributes of their
    countries, parts inherit attributes of wholes.
  - `boolean(kb, S, R, V)` -- multi-valued-aware. Treats `has`,
    `usedfor` etc. as multi-valued (any candidate match is True), but
    `color`, `capital`, `isa` etc. as single-valued (top-1 contradicts).
  - `enumerate_subjects(kb, R, V)` -- fan-out across shards.
  - `compare(kb, A, B)` -- ranked comparison along an ordinal dimension.
- **`rck/nlg.py`**: per-relation NL templates. Renders (S, R, O) into
  full English sentences. Deterministic template choice via blake2b hash.
- **`rck/dialogue.py`**: `DialogueContext`. Tracks `last_entity`,
  `last_relation`. `resolve_references` replaces pronouns; `with_default_topic`
  rewrites "what about the grass?" -> "what color is the grass?".
- **Question-type classifier + dispatch in `ConsciousAgent`**:
  `_classify_question` routes boolean/enumeration/comparison/factual.
- **Expanded common-sense KB** to 486 facts. Added cities -> countries,
  sizes, math sums, history, body parts, weather/seasons, months.
- **`examples/conversational_demo.py`**: full 60-turn demo exercising
  every capability.
- New tests: `test_inference_nlg.py` (13 tests covering direct lookup,
  multi-hop inheritance, multi-valued boolean, contradiction, enumeration,
  comparison, NLG templates, dialogue resolution).

### Changed
- **Cascade thresholds tightened**: weak-bar 0.05 -> 0.10 so noise hits
  from cross-shard binding return "I don't know" instead of nonsense.
  Is-fallback bar raised to 0.15.
- Self-model adds capability facts so RCK can introspect on its own
  multi-hop / boolean / enumeration / comparison / dialogue / ToM /
  self-awareness support.

### Honest framing
- The 486-fact KB is hand-curated common sense. A real-scale RCK would
  ingest from ConceptNet/Wikidata (millions of triples). The sharded
  substrate is ready; the bulk import remains a future session.
- "Competitive with LLMs" means competitive on **exact retrieval,
  traceable reasoning, editable knowledge, sub-100ms latency,
  no-hallucination "I don't know"** -- the things LLMs are bad at. NOT
  on raw fluency or world-coverage breadth, which require compute at a
  scale this project cannot match.

## 1.3.0 — 2026-05-20

The "broad knowledge + grounded introspection" release. Scales RCK from
~50 facts to thousands; adds self-model, theory of mind, meta-cognitive
confidence, and an introspection API. Implements the operational
markers proposed by Global Workspace Theory of consciousness without
making any phenomenal-consciousness claims.

### Added
- **`rck/knowledge_base.py`**: `ShardedKnowledgeBase`. N HRR shards
  routed by `blake2b(subject || relation)`. Breaks the
  ~250-fact-per-memory capacity ceiling. Tested at **2000 facts /
  64 shards / >85% recall**.
- **`rck/self_model.py`**: RCK stores ~36 structured facts about
  itself (identity, architecture, capabilities, limits). `self_describe()`
  generates a grounded natural-language description from retrieved facts.
- **`rck/introspect.py`**: `IntrospectionBuffer` keeps a ring of recent
  workspace broadcasts. `think()` returns a natural-language report on
  current internal state -- prediction error, column uncertainty,
  top-firing modules, recent emission, codebook size.
- **`rck/metacog.py`**: `epistemic_category()` maps confidence to
  `know` / `think` / `guess` / `unknown`. `verbalize()` renders the
  natural-language hedged response. `CalibrationTally` tracks
  claims-vs-correctness over time.
- **`rck/theory_of_mind.py`**: belief tuples `(believer, S, R, O)` in a
  separate KB. Distinguishes "Bob believes Paris is the capital" from
  ground truth. Supports false-belief tasks.
- **`rck/conscious_agent.py`**: `ConsciousAgent` -- top-level v1.3 model
  combining all of the above into one API: `tell`, `tell_belief`,
  `load_jsonl`, `ingest_text`, `ask`, `what_does_x_think`,
  `who_am_i`, `think`.
- **`scripts/make_commonsense_kb.py`**: synthesises a 368-fact
  common-sense KB across colors / capitals / animals / parts /
  materials / uses / locations / causes / authors / scientists / foods.
- **`examples/conscious_demo.py`**: 58-question evaluation.
  **55/55 known + 3/3 soft-no on unknowns = 100%.**
- 17 new tests across `test_knowledge_base.py`,
  `test_self_metacog_tom.py`.

### Changed
- **Question parser** now has a relation-alias cascade:
  `kind/type -> isa`, `wrote -> wrote+author`, `category -> category+isa`,
  `value -> value`, `causes -> causes`, `madeof -> madeof`, etc. Tries
  the specific relation FIRST and only falls back to `is` if nothing
  passes the 0.10 confidence bar.
- **Self-model `_first` / `_all` thresholds raised to 0.10** so cross-shard
  cleanup noise doesn't pollute `self_describe()`'s output.

### Honest limits
- This release implements the **operational subset** of Global Workspace
  Theory (Baars/Dehaene): broadcast competition, introspectable history,
  self-model retrieval, meta-cognitive verbalisation. These are the
  capabilities consciousness theories propose. We do not claim phenomenal
  consciousness; that remains philosophically open.
- 368 facts is "GPT-2 trivia" scale, not "GPT-4". Bulk ingestion from
  full Wikidata / ConceptNet remains a future session (download infra).

## 1.2.0 — 2026-05-20

The "real generative AI" release. RCK now ingests natural-language
sentences, stores facts, answers questions, and falls back to
free-form generation -- all from a single unified API.

### Added
- **`rck/tokenizer.py`**: word-level tokenizer + detokenizer + sentence
  splitter. Replaces char-level for the generative path.
- **`rck/generative.py`**: `GenerativeRCK` class -- the user-facing
  generative AI model.
  - `tell(s, r, o)` stores a structured fact.
  - `ingest(text)` learns from natural-language sentences and auto-
    extracts triples from "X is Y" / "X has Y" / "X of Y is Z" /
    "X lives in Y" / "X wrote Y" patterns.
  - `ask(question)` parses + retrieves with a multi-relation cascade
    (specific relation -> `is` fallback -> reverse-subject lookup).
  - `generate(prompt, n)` free-form word-level continuation.
- **`rck/gen_server.py`**: HTTP server + web chat UI for
  `GenerativeRCK`. Tabs for ask / teach / tell / generate, live
  candidate breakdown, model state panel.
- **`examples/generative_qa.py`**: 28-question knowledge-base QA benchmark.
  **RCK 100% on 26 known questions + 2/2 soft-no on 2 unknowns.**
- **`examples/talk_to_rck.py`**: interactive REPL.
- **`data/world_knowledge.txt`**: 47-sentence canned knowledge corpus.
- 18 new tests across `test_tokenizer`, `test_generative`, `test_qa_corpus`,
  `test_gen_server`.

### Changed
- **PCN now has activation + weight clipping** (`activation_clip=6.0`,
  `weight_clip=4.0`). Fixes float32 overflow on word-level vocabularies
  (>1k input dim) during long ingestion runs.
- **Codebook and RelationalMemory salts use blake2b** instead of
  Python's built-in `hash()`. Built-in `hash()` is randomised between
  processes since 3.3, which made HVs non-deterministic across runs.
  Blake2b is process-stable.

### Results
- Generative QA on 47-sentence corpus, 28 questions:
  - 14/16 in v1.1 -> **16/16 known correct** (100%) after parser cascade.
  - 2/2 unknown soft-rejected.
  - Total **28/28 = 100%**.

## 1.1.0 — 2026-05-20

The "groundbreaking" release. Demonstrates capabilities that
mainstream LLMs structurally cannot replicate.

### Added
- **`rck/relational.py`**: Plate-style Holographic Reduced Representation
  memory. Multiplicative binding within facts; sum-of-facts memory.
  Recovers values from key-only queries via VSA unbinding.
- **`rck/compose.py`**: Slot-based compositional reasoner. Trains on
  single-slot primitive facts only; composes UNSEEN multi-slot outputs
  by algebra of hypervectors.
- **`rck/explain.py`**: Hallucination-free self-explanation generator
  that cites Tsetlin clauses + bigram-query scores + FEP signal sources.
- **`examples/compositional_generalization.py`**: 64 + 128 + 384 unseen
  multi-slot combinations -- 100% on all of them.
- **`examples/federated_bundling.py`**: Two agents trained on disjoint
  facts merged by hypervector bundling -- both knowledge sets survive at
  100% recall.
- **`examples/editable_knowledge.py`**: O(D) forget / rename / re-teach
  demos. LLMs cannot do these without retraining.
- **`examples/scan_lite.py`**: SCAN-style command->action compositional
  benchmark with a char-bigram-LM baseline. **RCK = 100% (8/8), bigram = 0% (0/8)**.
- 9 new tests: `test_relational.py`, `test_compose.py`, `test_scan.py`,
  `test_explain.py`.

### Changed
- **Codebook atoms are now name-hash-seeded** (`rck/codebook.py`). Two
  codebooks with the same `seed` mint the same HV for the same symbol
  regardless of insertion order. This is the property federated bundling
  requires.
- **Relational role HVs are name-hash-seeded** (`rck/relational.py`).
  Avoids accidental collision with codebook atoms when they share `seed`.

### Fixed
- Pre-1.1: codebook atoms collided with relational role HVs when both used
  `np.random.default_rng(seed=0)`. The first atom drawn by `codebook` was
  numerically identical to the first role drawn by `RelationalMemory`,
  silently destroying fact structure. v1.1 hashes by name so the two
  streams cannot collide.

## 1.0.0 — 2026-05-20

First stable release. The architecture is real and demonstrably works.

### Added
- **`rck/server.py`**: Dependency-free HTTP server with a web chat UI. Live
  reasoning-trace panel, codebook stats, one-shot teach button. No Flask --
  pure stdlib `ThreadingHTTPServer`.
- **`rck/mcp_server.py`**: MCP server exposing 8 tools (`rck_observe`,
  `rck_generate`, `rck_one_shot`, `rck_explain`, ...) for use from Claude
  Code or any MCP client.
- **`rck/persist.py`**: Versioned save/load. `.npz` + `.json` sidecar,
  schema-versioned, forward-compatible.
- **Top-p (nucleus) sampling** in `FEP._sample`. Replaces full-softmax
  sample; cuts noise from low-probability tail.
- **Codebook matrix cache** in `Codebook` -- `fast_cleanup` now ~10x faster
  by caching the stacked atom matrix + per-row norms.
- **Console scripts**: `rck`, `rck-server`, `rck-mcp`.
- **Benchmark suite** (`scripts/benchmark.py`) with per-module timing.
- New tests: `test_persist.py`, `test_server.py`, `test_mcp.py`.

### Changed
- **Vectorised `TsetlinLayer`**: per-clause Python loops replaced with full
  `(n_clauses, 2*n_features)` matrix ops. ~10x faster `feedback`.
- **`agent.step` is now 5.5 ms/step** (was 6.6 ms in v0.2, 9-10 ms in v0.1).
- **`Codebook.fast_cleanup`** uses `argpartition` for top-k instead of full
  argsort.

### Removed
- Pickle-based save/load. Use `rck.persist.save/load` instead.

## 0.2.0 — 2026-05-20

### Added
- **`rck/bigram.py`** -- VSA n-gram associative memory (Kanerva HRR).
- **Low-rank FEP** `A = U V^T` (default rank=96). O(D*r) per step.
- **Multi-signal EFE decoder** -- weighted ensemble over LSM readout,
  bigram query, and FEP transition.
- **N-gram repetition penalty** (bigram + trigram cycle detection).

### Changed
- Frequency prior dropped from 0.5 -> 0.05 (over-biased toward common chars).
- Deterministic decoding by default. Sampling is opt-in.

### Results (Tiny Shakespeare, char-level)
- 5k train, top-1 next-char: 0.204 → 0.355 (19.6x random).
- 5k train time: 47.4s → 26.4s.
- Bigram probes correct: 2/7 → 6/7.

## 0.1.0 — 2026-05-20

Initial proof-of-architecture release.

### Added
- `vsa.py` -- bind / bundle / permute / cosine / binarize.
- `codebook.py` -- HV atom table with cleanup memory.
- `pcn.py` -- Predictive Coding encoder, local Hebbian updates.
- `lsm.py` -- Liquid State Machine + RLS readout.
- `tsetlin.py` -- Tsetlin Machine layer with explanations.
- `workspace.py` -- Global Workspace Theory cosine-WTA.
- `fep.py` -- Free Energy Principle / Active Inference (dense A).
- `columns.py` -- Thousand-Brains column ensemble.
- `agent.py` -- top-level RCKAgent.
- `cli.py` -- `rck` CLI with train / chat / eval / demo.
- `examples/continual_learning.py`, `examples/one_shot_vocab.py`.
- 31 unit tests.
