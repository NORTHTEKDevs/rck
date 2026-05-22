"""MCP server exposing RCK as a tool to Claude Code and other MCP clients.

Tools:
  rck_observe(text, learn=True)        -- stream text into the agent
  rck_generate(prompt, max_new, temp)  -- generate continuation + trace
  rck_reset_temporal()                 -- clear LSM / workspace state
  rck_state()                          -- model summary (codebook, position)
  rck_save(path), rck_load(path)       -- persistence
  rck_one_shot(symbol)                 -- mint a new codebook atom
  rck_explain()                        -- last reasoning trace

Run:
  python -m rck.mcp_server [--load checkpoints/rck_100k]

Register with Claude Code via:
  claude mcp add rck python -m rck.mcp_server --load /path/to/checkpoint
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from rck.agent import RCKAgent
from rck.persist import load as persist_load, save as persist_save


_AGENT: RCKAgent | None = None
_LAST_TRACE = None


def _require_agent() -> RCKAgent:
    if _AGENT is None:
        raise RuntimeError("RCK agent not initialised")
    return _AGENT


def build(server: FastMCP) -> FastMCP:
    @server.tool()
    def rck_observe(text: str, learn: bool = True) -> dict:
        """Feed `text` (any string) to RCK for learning or just to update state."""
        a = _require_agent()
        a.observe(list(text), learn=learn)
        return {"steps": len(text), "codebook_size": a.codebook.size()}

    @server.tool()
    def rck_generate(prompt: str, max_new: int = 40, temperature: float = 0.0) -> dict:
        """Generate up to `max_new` symbols after `prompt`. temperature>0 enables sampling."""
        global _LAST_TRACE
        a = _require_agent()
        a.stochastic_decode = temperature > 1e-4
        a.fep.temperature = max(temperature, 1e-3)
        out, traces = a.generate(list(prompt), max_new=max_new)
        text = "".join(str(c) for c in out)
        tr = traces[-1] if traces else None
        _LAST_TRACE = tr
        return {
            "emitted": text,
            "trace": _trace_to_dict(tr),
        }

    @server.tool()
    def rck_reset_temporal() -> dict:
        """Clear the LSM / workspace temporal state. Codebook + weights survive."""
        a = _require_agent()
        a.reset_temporal()
        return {"ok": True}

    @server.tool()
    def rck_state() -> dict:
        """Model summary -- codebook size, position, hyperparameters."""
        a = _require_agent()
        return {
            "codebook_size": a.codebook.size(),
            "position": a._position,
            "hv_dim": a.hv_dim,
            "n_columns": a.n_columns,
            "reservoir_dim": a.reservoir_dim,
            "n_clauses": a.n_clauses,
            "fep_rank": a.fep_rank,
            "bigram_order": a.bigram_order,
            "version": "1.0.0",
        }

    @server.tool()
    def rck_one_shot(symbol: str) -> dict:
        """Mint a brand-new codebook atom for `symbol`. Returns codebook size."""
        a = _require_agent()
        existed = a.codebook.has(symbol)
        a.codebook.encode(symbol)
        return {
            "symbol": symbol,
            "already_existed": existed,
            "codebook_size": a.codebook.size(),
        }

    @server.tool()
    def rck_explain() -> dict:
        """Return the most recent step's reasoning trace."""
        if _LAST_TRACE is None:
            return {"error": "no step taken yet"}
        return _trace_to_dict(_LAST_TRACE)

    @server.tool()
    def rck_save(path: str) -> dict:
        a = _require_agent()
        persist_save(a, path)
        return {"ok": True, "path": path}

    @server.tool()
    def rck_load(path: str) -> dict:
        global _AGENT
        _AGENT = persist_load(path)
        return {"ok": True, "path": path, "codebook_size": _AGENT.codebook.size()}

    return server


def _trace_to_dict(tr) -> dict:
    if tr is None:
        return {}
    return {
        "input": str(tr.input_symbol),
        "emitted": str(tr.emitted_symbol),
        "workspace_winner": tr.workspace_winner,
        "workspace_score": float(tr.workspace_score),
        "column_uncertainty": float(tr.column_uncertainty),
        "tsetlin_score": float(tr.tsetlin_score),
        "tsetlin_clauses": list(tr.tsetlin_clauses or []),
        "bigram_top": [[str(s), float(score)] for s, score in (tr.bigram_top or [])],
        "pred_err": float(tr.pred_err),
    }


def main(argv: list[str] | None = None) -> int:
    global _AGENT
    p = argparse.ArgumentParser(prog="rck.mcp_server")
    p.add_argument("--load", default=None, help="checkpoint path (no extension)")
    p.add_argument("--hv-dim", type=int, default=1024)
    p.add_argument("--vocab", type=int, default=80)
    p.add_argument("--columns", type=int, default=2)
    p.add_argument("--reservoir", type=int, default=96)
    p.add_argument("--clauses", type=int, default=16)
    p.add_argument("--fep-rank", type=int, default=64)
    p.add_argument("--bigram-order", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.load and Path(args.load).with_suffix(".npz").exists():
        _AGENT = persist_load(args.load)
        print(f"[rck-mcp] loaded {args.load}", file=sys.stderr)
    else:
        _AGENT = RCKAgent(
            vocab_size=args.vocab, hv_dim=args.hv_dim,
            n_columns=args.columns, reservoir_dim=args.reservoir,
            n_clauses=args.clauses, fep_rank=args.fep_rank,
            bigram_order=args.bigram_order, seed=args.seed,
        )
        print(f"[rck-mcp] fresh agent (no checkpoint)", file=sys.stderr)

    server = build(FastMCP("rck", instructions=(
        "Resonant Cognitive Kernel -- a compact non-LLM AI. "
        "Use rck_observe to teach it text, rck_generate to ask for completions, "
        "rck_one_shot to mint new vocabulary atoms, rck_explain for reasoning traces."
    )))
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
