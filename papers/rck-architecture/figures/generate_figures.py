"""Generate all paper figures from the empirical data files.

Reads data/*.json (relative to the repository root) and writes
PDF figures into the same directory as this script.

Usage (from the repo root):
    python papers/rck-architecture/figures/generate_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
DATA = REPO / "data"


def _load(name: str) -> dict:
    path = DATA / name
    return json.loads(path.read_text())


def _save(name: str) -> None:
    out = HERE / name
    plt.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 1: stack diagram (text-based; LaTeX will render properly)
# ---------------------------------------------------------------------------

def stack_diagram() -> None:
    """Render a clean four-layer stack diagram."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.axis("off")

    layers = [
        (6.5, "Surface", "IDK / calibrated ask / set reasoning / explain-why",
         "#5b8def"),
        (4.8, "Knowledge mgmt",
         "provenance / skills / query memory / contradictions / belief revision",
         "#7eb77f"),
        (3.1, "Reasoning",
         "chain walker / discoverer / induction / rules / analogy / causal",
         "#f0b942"),
        (1.4, "Substrate",
         "HRR memory  +  codebook  +  sharded KB",
         "#e07b6b"),
    ]
    for y, title, sub, colour in layers:
        rect = plt.Rectangle((0.5, y - 0.6), 9, 1.2, facecolor=colour,
                              edgecolor="black", linewidth=1, alpha=0.85)
        ax.add_patch(rect)
        ax.text(5, y + 0.15, title, ha="center", va="center",
                fontsize=13, fontweight="bold")
        ax.text(5, y - 0.25, sub, ha="center", va="center", fontsize=9)

    ax.text(5, 7.5, "ConsciousAgent (integrated API)",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="#222")
    ax.annotate("", xy=(5, 7.05), xytext=(5, 7.2),
                arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.5))

    _save("stack-diagram.pdf")


# ---------------------------------------------------------------------------
# Figure 2: chain-depth study
# ---------------------------------------------------------------------------

def chain_depth() -> None:
    data = _load("chain_depth_study.json")
    rows = data["rows"]
    depths = sorted({r["depth"] for r in rows})

    plt.figure(figsize=(7, 4.5))
    for rule, marker, colour in [
        ("product",        "o", "#e07b6b"),
        ("min",            "s", "#7eb77f"),
        ("geometric_mean", "D", "#5b8def"),
    ]:
        ys = []
        for d in depths:
            match = [r for r in rows if r["rule"] == rule and r["depth"] == d]
            if match:
                ys.append(match[0]["confidence"])
            else:
                ys.append(None)
        plt.plot(depths, ys, marker=marker, label=rule,
                 linewidth=2, markersize=7, color=colour)

    # Threshold lines.
    plt.axhline(0.30, color="#aaa", linestyle="--", linewidth=0.8)
    plt.axhline(0.10, color="#aaa", linestyle="--", linewidth=0.8)
    plt.axhline(0.03, color="#aaa", linestyle="--", linewidth=0.8)
    plt.text(50, 0.305, "strong", color="#666", fontsize=8, ha="right")
    plt.text(50, 0.105, "moderate", color="#666", fontsize=8, ha="right")
    plt.text(50, 0.033, "weak", color="#666", fontsize=8, ha="right")

    plt.xlabel("Chain depth (hops)")
    plt.ylabel("Reported confidence")
    plt.title("Propagation rule controls reported confidence "
              "(retrieval is 100% at every depth)")
    plt.legend(loc="upper right", fontsize=10)
    plt.yscale("log")
    plt.grid(True, linewidth=0.3, alpha=0.5)
    _save("chain-depth.pdf")


# ---------------------------------------------------------------------------
# Figure 3: sparse vs dense capacity
# ---------------------------------------------------------------------------

def sparse_vs_dense() -> None:
    sparse = _load("sparse_capacity_study.json")
    dense = _load("capacity_study.json")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    # Dense capacity points (from capacity_study.json).
    # Format may vary; defensively pull data.
    dense_xs, dense_ys = [], []
    if isinstance(dense, dict) and "results" in dense:
        for row in dense["results"]:
            if row.get("dim") == 4096:
                dense_xs.append(row.get("n_facts", row.get("facts", 0)))
                dense_ys.append(row.get("recall_at_1", row.get("recall", 0)))
    if dense_xs:
        ax.plot(dense_xs, dense_ys, marker="o", color="#5b8def",
                label="Dense bipolar D=4096", linewidth=2, markersize=7)

    # Sparse: read configurations + cliffs from the sparse study.
    cliffs = sparse.get("cliffs_at_90pct_recall", {})
    label_done = False
    for key, cliff in cliffs.items():
        # key like "D8192_k320"
        # Plot single point at the cliff value, recall ~0.90.
        ax.plot(cliff, 0.90, marker="D", color="#e07b6b", markersize=9,
                label="Sparse cliff (various D, k)" if not label_done else None)
        label_done = True
        ax.annotate(key, (cliff, 0.90),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color="#666")

    ax.axhline(0.90, color="#aaa", linestyle="--", linewidth=0.8)
    ax.text(ax.get_xlim()[1] * 0.98, 0.905,
            "90% recall threshold",
            color="#666", fontsize=8, ha="right")

    ax.set_xlabel("Facts per shard")
    ax.set_ylabel("Recall @ 1")
    ax.set_title("Per-shard capacity: dense bipolar vs.\\ sparse-binary HRR")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    _save("sparse-vs-dense.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Generating figures from data/*.json ...")
    stack_diagram()
    try:
        chain_depth()
    except Exception as e:
        print(f"chain-depth failed: {e}", file=sys.stderr)
    try:
        sparse_vs_dense()
    except Exception as e:
        print(f"sparse-vs-dense failed: {e}", file=sys.stderr)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
