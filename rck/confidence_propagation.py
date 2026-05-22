"""Confidence propagation through reasoning chains.

When RCK answers a multi-hop question, the final confidence should
reflect the WEAKEST link in the chain, not the strongest. Currently
the `infer()` engine returns `min(score1, score2) * 0.8` heuristically;
this module implements the principled version.

Two combination rules:
  * **Product rule (independent)**: chain_conf = prod(link_confs).
    Use when each step is independent (no shared uncertainty).
  * **Min rule (worst-link)**: chain_conf = min(link_confs) * decay.
    Use when steps share a common epistemic source.

For most KB chains, product rule with a small floor is appropriate.
The decay term penalises long chains -- each hop adds uncertainty
even if each individual link is near-certain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PropagationConfig:
    rule: str = "geometric_mean"  # "product", "min", or "geometric_mean"
    chain_decay: float = 0.95     # multiplied per hop
    floor: float = 0.01           # never report below this

    def combine(self, link_confs: list[float], chain_length: int) -> float:
        if not link_confs:
            return 0.0
        if self.rule == "product":
            base = 1.0
            for c in link_confs:
                base *= max(c, 0.0)
        elif self.rule == "min":
            base = min(link_confs)
        elif self.rule == "geometric_mean":
            # Geometric mean: (prod(scores))^(1/n). Normalises for chain
            # length so long but uniformly-strong chains stay confident.
            log_sum = 0.0
            for c in link_confs:
                log_sum += math.log(max(c, 1e-9))
            base = math.exp(log_sum / len(link_confs))
        else:
            raise ValueError(f"unknown rule {self.rule!r}")
        decayed = base * (self.chain_decay ** max(0, chain_length - 1))
        return max(self.floor, decayed)


def propagate(link_confidences: list[float],
              *, config: PropagationConfig | None = None) -> dict:
    """Compute the propagated confidence for a chain.

    Returns dict with:
      * final_confidence
      * weakest_link_index
      * verbal hedge ("strong" / "moderate" / "weak" / "uncertain")
    """
    if not link_confidences:
        return {
            "final_confidence": 0.0,
            "weakest_link_index": -1,
            "hedge": "unknown",
        }
    cfg = config or PropagationConfig()
    final = cfg.combine(link_confidences, len(link_confidences))
    weakest = min(range(len(link_confidences)), key=lambda i: link_confidences[i])
    if final >= 0.30:
        hedge = "strong"
    elif final >= 0.10:
        hedge = "moderate"
    elif final >= 0.03:
        hedge = "weak"
    else:
        hedge = "uncertain"
    return {
        "final_confidence": final,
        "weakest_link_index": weakest,
        "hedge": hedge,
    }


def verbalize_chain_confidence(propagation: dict, base_answer: str) -> str:
    """Phrase the answer with calibrated hedging based on the propagation."""
    hedge = propagation["hedge"]
    if hedge == "strong":
        return f"I'm confident: {base_answer}"
    if hedge == "moderate":
        return f"It's likely that {base_answer.lower()}"
    if hedge == "weak":
        return f"My best guess (with low confidence) is {base_answer.lower()}"
    return f"I don't have strong evidence, but possibly {base_answer.lower()}"
