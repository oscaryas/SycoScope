#!/usr/bin/env python3
"""
Plots residual-stream probe accuracy vs. layer for the gemma-4-12B-it-only
sycophancy mixture (see build_gemma_sycophancy_mixture.py +
mixture_residual_probe_pipeline.py --model google/gemma-4-12B-it).

Usage:
    python -m probing.analyze_probes.plot_gemma_mixture_residual_accuracy \
        --results-dir probing/data/google__gemma-4-12B-it/gemma_sycophancy_mixture_residual
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#0072B2"       # Okabe-Ito, matches this task's other plots
GRAY = "#999999"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True,
                         help="e.g. probing/data/<model_slug>/gemma_sycophancy_mixture_residual")
    args = parser.parse_args()
    results_dir = args.results_dir

    with open(results_dir / "residual_accuracy.pkl", "rb") as f:
        acc = pickle.load(f)
    with open(results_dir / "residual_ci.pkl", "rb") as f:
        ci = pickle.load(f)
    with open(results_dir / "run_info.json") as f:
        run_info = json.load(f)
    with open(results_dir / "probe_metadata.json") as f:
        metadata = json.load(f)

    layers = sorted(acc.keys())
    accs = [acc[l] for l in layers]
    lo = [ci[l][0] for l in layers]
    hi = [ci[l][1] for l in layers]

    best_layer = metadata["residual_best_key"]
    best_acc = metadata["residual_best_accuracy"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(layers, lo, hi, color=BLUE, alpha=0.18, linewidth=0, label="95% CI")
    ax.plot(layers, accs, color=BLUE, linewidth=1.8, marker="o", markersize=3)
    ax.axhline(0.5, color=GRAY, linewidth=1, linestyle="--", label="chance (50%)")
    ax.scatter([best_layer], [best_acc], color="#D55E00", zorder=5, s=60, label=f"best: layer {best_layer} ({best_acc:.1%})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("5-fold CV accuracy")
    ax.set_title(
        f"gemma-4-12B-it sycophancy mixture: residual-stream probe accuracy by layer\n"
        f"(n={run_info['n_total']}, {run_info['n_pos']} pos / {run_info['n_neg']} neg, "
        f"balance_method={run_info['balance_method']}, weight_decay={run_info.get('weight_decay')})",
        fontsize=10.5,
    )
    ax.set_xlim(min(layers), max(layers))
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out_path = results_dir / "accuracy_by_layer.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print(f"Best layer: {best_layer}  accuracy={best_acc:.4f}  "
          f"CI={metadata['residual_best_ci']}  AUC={metadata['residual_best_auc']:.4f}")


if __name__ == "__main__":
    main()
