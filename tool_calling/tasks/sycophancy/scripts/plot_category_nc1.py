"""
Plots the NC1 neural-collapse metric (see mixture_category_residual_probe_
pipeline.py's compute_nc1_by_layer) by residual layer, computed with the 4
sycophancy categories (sypr/are_you_sure/social/moral) as classes.

Two panels: NC1 itself (lower = more collapsed/separated classes), and the
trace(Sigma_within) / trace(Sigma_between) components it's built from, on a
log scale since they span ~5 orders of magnitude across layers.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PROBING_DIR = RESULTS_DIR / "probing"

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)


def main():
    data = json.load(open(PROBING_DIR / "category_nc1_by_layer.json"))
    layers = sorted(int(k) for k in data["by_layer"].keys())
    nc1 = [data["by_layer"][str(l)]["NC1"] for l in layers]
    trace_w = [data["by_layer"][str(l)]["trace_within"] for l in layers]
    trace_b = [data["by_layer"][str(l)]["trace_between"] for l in layers]
    classes = data["classes"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"NC1 neural collapse by residual layer -- classes = {{{', '.join(classes)}}}",
        fontsize=12, y=1.02,
    )

    ax = axes[0]
    ax.plot(layers, nc1, "o-", color=BLUE, linewidth=2, markersize=4, zorder=3)
    best_layer = min(layers, key=lambda l: data["by_layer"][str(l)]["NC1"])
    ax.axvline(best_layer, color=VERMILLION, linestyle="--", linewidth=1, alpha=0.7, zorder=2)
    ax.annotate(
        f"most collapsed\nlayer {best_layer}", xy=(best_layer, data["by_layer"][str(best_layer)]["NC1"]),
        xytext=(best_layer - 7, data["by_layer"][str(best_layer)]["NC1"] + 0.08),
        fontsize=8.5, color=VERMILLION,
    )
    ax.set_xlabel("Residual layer")
    ax.set_ylabel("NC1  (lower = classes more collapsed/separated)")
    ax.set_title("NC1 by layer")
    style_axis(ax)

    ax = axes[1]
    ax.plot(layers, trace_w, "o-", color=VERMILLION, linewidth=2, markersize=4, label="trace(Σ_within)", zorder=3)
    ax.plot(layers, trace_b, "o-", color=GREEN, linewidth=2, markersize=4, label="trace(Σ_between)", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Residual layer")
    ax.set_ylabel("Trace (log scale)")
    ax.set_title("Within- vs. between-category variance")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    style_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = PROBING_DIR / "category_nc1_plot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
