"""Calibrated confidence for retrieval, layered over the HRR substrate.

The KB returns cosine scores directly from HRR cleanup. Those scores
say nothing about WHERE a fact came from. An induced fact and a
first-hand observed fact look identical at retrieval time.

This module applies a discount AT THE CALLER level based on the
provenance store:

  * `source="user"`     -- full confidence (no discount).
  * `source="induced"`  -- discounted by `via_hops` * `per_hop_decay`.
  * `source="multi"`    -- slight bonus (multiple sources reinforce).
  * `source="external"` -- moderate discount (third-party ingest).

The discount is applied multiplicatively to the raw HRR score, so the
relative ordering of candidates remains stable as long as their
provenances don't differ wildly.

This is the cleanest place to plug in any future calibration work
(e.g. learned reliability scores from `CalibrationTally`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from rck.knowledge_base import ShardedKnowledgeBase
from rck.provenance import ProvenanceStore


@dataclass
class CalibrationPolicy:
    user_factor: float = 1.0
    induced_per_hop_decay: float = 0.9      # 2-hop induced -> 0.81; 3-hop -> 0.73
    induced_floor: float = 0.5              # never below this
    multi_bonus: float = 1.05               # very mild
    external_factor: float = 0.85
    unknown_factor: float = 1.0             # no record -> trust the substrate


def calibrated_score(raw_score: float, source: str, via_hops: int,
                     *, policy: CalibrationPolicy | None = None) -> float:
    """Apply provenance-based discount to a raw HRR score."""
    cfg = policy or CalibrationPolicy()
    if source == "user":
        factor = cfg.user_factor
    elif source == "induced":
        factor = max(cfg.induced_floor,
                     cfg.induced_per_hop_decay ** max(1, via_hops))
    elif source == "multi":
        factor = cfg.multi_bonus
    elif source == "external":
        factor = cfg.external_factor
    else:
        factor = cfg.unknown_factor
    return max(0.0, raw_score * factor)


@dataclass
class CalibratedAnswer:
    symbol: Hashable
    raw_score: float
    calibrated_score: float
    source: str
    via_hops: int


def calibrated_lookup(kb: ShardedKnowledgeBase,
                      provenance: ProvenanceStore,
                      known: dict[str, Hashable], unknown_role: str,
                      *, top_k: int = 5,
                      policy: CalibrationPolicy | None = None
                      ) -> list[CalibratedAnswer]:
    """Query the KB and apply per-fact provenance calibration."""
    cfg = policy or CalibrationPolicy()
    raw = kb.query(known, unknown_role, top_k=top_k * 2)
    out: list[CalibratedAnswer] = []
    for sym, score in raw:
        # Build the full triple to look up provenance.
        s = str(known.get("S", "")).lower()
        r = str(known.get("R", "")).lower()
        o = str(known.get("O", "")).lower()
        if unknown_role == "O":
            o = str(sym).lower()
        elif unknown_role == "S":
            s = str(sym).lower()
        elif unknown_role == "R":
            r = str(sym).lower()
        rec = provenance.get(s, r, o)
        if rec is None:
            source, hops = "unknown", 1
        else:
            source = rec.source
            hops = 1
            for tag in rec.tags:
                if tag.startswith("via_") and tag.endswith("_hops"):
                    try:
                        hops = int(tag.split("_")[1])
                    except (ValueError, IndexError):
                        pass
                    break
        cal = calibrated_score(float(score), source, hops, policy=cfg)
        out.append(CalibratedAnswer(
            symbol=sym, raw_score=float(score),
            calibrated_score=cal, source=source, via_hops=hops,
        ))
    out.sort(key=lambda x: -x.calibrated_score)
    return out[:top_k]
