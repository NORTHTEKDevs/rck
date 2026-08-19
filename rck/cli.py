"""Command-line interface for RCK.

Usage:
    rck train <text_file> [--chars N] [--save model.pkl]
    rck chat [--load model.pkl] [--show-reasoning]
    rck eval <text_file> [--load model.pkl] [--chars N]
    rck demo continual
    rck demo one-shot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rck import persist
from rck.agent import RCKAgent
from rck.train import recall_score, stream_text, train_on_text


def _save(agent: RCKAgent, path: str | Path) -> None:
    # v1.5+: rck.persist's explicit, versioned format -- not pickle.
    # pickle.loads on a caller-supplied path is an arbitrary-code-execution
    # sink; this closes that hole for the CLI's save/load round trip.
    persist.save(agent, path)


def _load(path: str | Path) -> RCKAgent:
    return persist.load(path)


def _model_exists(path: str | Path) -> bool:
    # rck.persist.save writes <path>.json + <path>.npz, not <path> itself.
    return Path(path).with_suffix('.json').exists()


def _new_agent(args: argparse.Namespace) -> RCKAgent:
    return RCKAgent(
        vocab_size=args.vocab,
        hv_dim=args.hv_dim,
        n_columns=args.columns,
        reservoir_dim=args.reservoir,
        n_clauses=args.clauses,
        seed=args.seed,
    )


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hv-dim", type=int, default=2048)
    p.add_argument("--vocab", type=int, default=128)
    p.add_argument("--columns", type=int, default=4)
    p.add_argument("--reservoir", type=int, default=256)
    p.add_argument("--clauses", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)


def cmd_train(args: argparse.Namespace) -> int:
    if args.load and _model_exists(args.load):
        agent = _load(args.load)
        print(f"[rck] loaded agent from {args.load}")
    else:
        agent = _new_agent(args)
        print(f"[rck] new agent: hv_dim={agent.hv_dim} cols={agent.n_columns}")

    def on_log(i, tr):
        print(f"  step {i}: input={tr.input_symbol!r} emit={tr.emitted_symbol!r} "
              f"err={tr.pred_err:.3f} unc={tr.column_uncertainty:.4f} "
              f"win={tr.workspace_winner}")

    text = list(stream_text(args.path, lowercase=args.lowercase))
    train_on_text(agent, text, max_chars=args.chars, log_every=args.log_every, on_log=on_log)
    if args.save:
        _save(agent, args.save)
        print(f"[rck] saved -> {args.save}")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    if args.load and _model_exists(args.load):
        agent = _load(args.load)
        print(f"[rck] loaded {args.load}")
    else:
        agent = _new_agent(args)
        print(f"[rck] fresh agent, no training. Try `rck train` first for fluency.")

    print("Type 'exit' to quit. 'why' to dump last reasoning trace. 'reset' to clear temporal state.")
    last_traces = []
    while True:
        try:
            line = input(">> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line == "exit":
            return 0
        if line == "reset":
            agent.reset_temporal()
            print("[rck] temporal state cleared")
            continue
        if line == "why":
            if not last_traces:
                print("(nothing to explain yet)")
                continue
            tr = last_traces[-1]
            print(f"  workspace_winner={tr.workspace_winner} score={tr.workspace_score:.3f}")
            print(f"  column_uncertainty={tr.column_uncertainty:.4f}")
            print(f"  tsetlin_score={tr.tsetlin_score:.2f}")
            if tr.tsetlin_clauses:
                print("  active clauses:")
                for c in tr.tsetlin_clauses:
                    print(f"    " + c)
            else:
                print("  (no Tsetlin clauses have fired yet -- needs more training)")
            continue
        prompt = list(line)
        out, traces = agent.generate(prompt, max_new=args.max_new)
        last_traces = traces
        text = "".join(str(c) for c in out)
        print("<<", text)
        if args.show_reasoning and traces:
            tr = traces[-1]
            print(f"   [unc={tr.column_uncertainty:.4f} winner={tr.workspace_winner}]")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    if args.load and _model_exists(args.load):
        agent = _load(args.load)
    else:
        agent = _new_agent(args)
    text = Path(args.path).read_text(encoding="utf-8", errors="ignore")
    if args.lowercase:
        text = text.lower()
    score = recall_score(agent, text, n_eval=args.chars)
    print(f"next-char accuracy: {score:.3f}  ({args.chars} positions)")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    if args.which == "continual":
        from examples.continual_learning import main as run
        return run()
    if args.which == "one-shot":
        from examples.one_shot_vocab import main as run
        return run()
    print(f"unknown demo: {args.which}")
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rck")
    sub = p.add_subparsers(dest="cmd", required=True)

    tp = sub.add_parser("train", help="train on a text file")
    tp.add_argument("path")
    tp.add_argument("--chars", type=int, default=2000)
    tp.add_argument("--log-every", type=int, default=500)
    tp.add_argument("--lowercase", action="store_true")
    tp.add_argument("--load", default=None)
    tp.add_argument("--save", default=None)
    _add_model_args(tp)
    tp.set_defaults(func=cmd_train)

    cp = sub.add_parser("chat", help="chat with the agent")
    cp.add_argument("--load", default=None)
    cp.add_argument("--max-new", type=int, default=40)
    cp.add_argument("--show-reasoning", action="store_true")
    _add_model_args(cp)
    cp.set_defaults(func=cmd_chat)

    ep = sub.add_parser("eval", help="next-char accuracy on a text")
    ep.add_argument("path")
    ep.add_argument("--chars", type=int, default=400)
    ep.add_argument("--lowercase", action="store_true")
    ep.add_argument("--load", default=None)
    _add_model_args(ep)
    ep.set_defaults(func=cmd_eval)

    dp = sub.add_parser("demo", help="run a packaged demo")
    dp.add_argument("which", choices=["continual", "one-shot"])
    dp.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
