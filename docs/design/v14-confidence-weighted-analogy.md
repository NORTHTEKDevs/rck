# v14 confidence-weighted analogy

Date: 2026-05-21
Status: closed (positive; +5.2pp accuracy)

## Question

The original analogy solver picks the top-1 relation R from A->B,
then applies it to C. This is argmax over R-scores; it ignores
whether the chosen R actually applies well to C. Can we improve
accuracy by considering multiple R candidates and choosing the
one whose application to C has the strongest joint score?

## Change

`solve_analogy` now:
1. Gets up to `top_k_relations` R candidates from find_relation.
2. For each, applies to C and computes a joint score
   `joint = R_score * answer_score`.
3. Picks the (R, answer) pair with the highest joint score.
4. Surfaces all (R, R_score, answer, answer_score, joint) rows as
   `AnalogyResult.alternatives` for inspection.

The previous behaviour was: top-1 R then top-1 answer. The new
behaviour exhaustively considers all R candidates and chooses the
best END-TO-END answer.

## Benchmark

Same 115 probes from the commonsense KB.

| metric | argmax (v13) | joint score (v14) | delta |
|--------|--------------|-------------------|-------|
| relation accuracy | 85.2% | **92.2%** | +7.0pp |
| answer accuracy   | 88.7% | **93.9%** | +5.2pp |

A real improvement at zero substrate-side cost. The trade is
slightly more KB queries per analogy: `top_k_relations` extra
lookups per call instead of one. At commonsense scale this adds
~2ms; on a 4k-fact KB still under 30ms total.

## Why this works

The relation enumeration is noisy: HRR cleanup may return high
scores for spurious R candidates that happen to share crosstalk
positions with B. The argmax variant commits to that noisy
top-1 R and then applies it to C, where it often fails. The
joint-score variant lets the answer step VETO a bad R: a strong
R that doesn't apply to C gets a low answer_score and falls in
the ranking.

## ConsciousAgent

No API change. `agent.analogy()` already returns AnalogyResult;
callers can now access `.alternatives` and `.joint_score()`.

## Open work

* The joint-score helper would benefit from a Bayesian-weighted
  combiner instead of simple product. Currently a noisy answer
  hit dominates the joint score; a more principled prior over
  what "applies to C" means would help.
* The benchmark probes pick the EXPECTED answer arbitrarily from
  the KB. Many remaining failures are multi-valued relations
  (e.g. dog has legs/fur/tail) where the gold answer is one of
  several valid analogs. Real accuracy is higher than 93.9%.
