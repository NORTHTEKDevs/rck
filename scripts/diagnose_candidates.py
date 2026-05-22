"""Diagnose where the v0.2 accuracy regression lives.

Per held-out step, log:
  - is the true target in bigram.query top-5?
  - is the true target in LSM-readout cleanup top-5?
  - is the true target in FEP transition cleanup top-5?
  - is the true target in the COMBINED candidate set?
  - is the true target the EFE-argmin pick?

This tells us whether candidate generation or EFE ranking is the bottleneck.
"""
from __future__ import annotations

from pathlib import Path

from rck.agent import RCKAgent
from rck.vsa import cosine


def main() -> None:
    text = Path("data/tiny_shakespeare.txt").read_text(encoding="utf-8", errors="ignore")[:6000]
    train_text, eval_text = text[:5000], text[5000:5500]

    agent = RCKAgent(
        hv_dim=1024, n_columns=2, reservoir_dim=96, n_clauses=8,
        vocab_size=80, seed=0,
    )
    print(f"training on {len(train_text)} chars ...")
    agent.observe(train_text, learn=True)
    print(f"  codebook size: {agent.codebook.size()}")

    agent.reset_temporal()
    # Prime.
    for c in eval_text[:8]:
        agent.step(c, learn=False)

    in_bigram = 0
    in_lsm = 0
    in_fep = 0
    in_combined = 0
    efe_pick = 0
    total = 0

    for i in range(8, len(eval_text) - 1):
        c = eval_text[i]
        target = eval_text[i + 1]

        # Probe each source independently BEFORE taking the step.
        bg_top = [s for s, _ in agent.bigram.query(agent.codebook, top_k=5)]
        lsm_top = [s for s, _ in agent.codebook.fast_cleanup(agent.lsm.readout(), top_k=5)]
        fep_top = [s for s, _ in agent.codebook.fast_cleanup(agent.fep.predict(agent._last_ws), top_k=5)]

        in_bigram += target in bg_top
        in_lsm += target in lsm_top
        in_fep += target in fep_top
        in_combined += target in (set(bg_top) | set(lsm_top) | set(fep_top))

        tr = agent.step(c, learn=False)
        efe_pick += tr.emitted_symbol == target
        total += 1

    print(f"\nover {total} held-out positions:")
    print(f"  target in bigram.top5:    {in_bigram/total:.3f}")
    print(f"  target in lsm.top5:       {in_lsm/total:.3f}")
    print(f"  target in fep.top5:       {in_fep/total:.3f}")
    print(f"  target in COMBINED top5*: {in_combined/total:.3f}")
    print(f"  target = EFE argmin:      {efe_pick/total:.3f}")
    print(f"  (* = union of bigram, lsm, fep top-5 sets)")


if __name__ == "__main__":
    main()
