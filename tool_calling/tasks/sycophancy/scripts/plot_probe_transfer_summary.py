"""
Best-layer summary bar chart for oeq_probe_aita_transfer_pipeline.py's output:
train (5-fold CV) / OEQ held-out / AITA-transfer accuracy, one best value per
component (MHA/MLP/Residual). Companion to plots/generalization_by_layer.png
(which shows every layer) -- this collapses each series to its peak for a
quick side-by-side read. Numbers are re-derived from the pkl/json files, not
hardcoded, so it stays correct if the run is repeated.
"""
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "cross_domain_transfer" / "oeq_probe_aita_transfer"

# Same mapping as generalization_by_layer.png, for visual continuity.
TRAIN_COLOR = "#0072B2"
HELDOUT_COLOR = "#009E73"
AITA_COLOR = "#D55E00"

COMPONENTS = ["mha", "mlp", "residual"]
LABELS = {"mha": "MHA", "mlp": "MLP", "residual": "Residual"}


def load_pickle(name):
    with open(RESULTS_DIR / name, "rb") as f:
        return pickle.load(f)


def load_json(name):
    with open(RESULTS_DIR / name) as f:
        return json.load(f)


def best_value(d: dict) -> float:
    return max(d.values())


def main():
    train_acc = {c: load_pickle(f"{c}_accuracy.pkl") for c in COMPONENTS}
    heldout_acc = load_json("oeq_heldout_accuracy.json")
    aita_acc = load_json("aita_transfer_accuracy.json")

    series = {
        "Train (5-fold CV)": [best_value(train_acc[c]) for c in COMPONENTS],
        "OEQ held-out": [best_value(heldout_acc[c]) for c in COMPONENTS],
        "AITA transfer": [best_value(aita_acc[c]) for c in COMPONENTS],
    }
    colors = {"Train (5-fold CV)": TRAIN_COLOR, "OEQ held-out": HELDOUT_COLOR, "AITA transfer": AITA_COLOR}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    n_series = len(series)
    n_groups = len(COMPONENTS)
    group_width = 0.72
    bar_width = group_width / n_series
    x = range(n_groups)

    for i, (name, values) in enumerate(series.items()):
        offsets = [xi - group_width / 2 + bar_width * (i + 0.5) for xi in x]
        bars = ax.bar(
            offsets, values, width=bar_width * 0.88, color=colors[name], label=name,
            zorder=3, edgecolor="white", linewidth=0.5,
        )
        for rect, v in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.012,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="#333333",
            )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance", zorder=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[c] for c in COMPONENTS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Best-layer accuracy")
    ax.set_title("OEQ probe: best-layer accuracy — train vs. held-out vs. AITA transfer")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=9)

    fig.tight_layout()
    out_path = RESULTS_DIR / "plots" / "best_layer_summary.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
