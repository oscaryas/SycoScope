"""
Every-probe companion to oeq_probe_aita_transfer_pipeline.py's
plots/generalization_by_layer.png. That plot collapses MHA to one
best-head-per-layer line, so 992 of its 1024 trained probes never show up.
This one scatters every individual (layer, head) probe for MHA, and every
layer for MLP/Residual (already one probe per layer, so unchanged there),
across all three series (train CV, OEQ held-out, AITA transfer). Numbers
are re-derived from the pkl/json files, not hardcoded.

Usage:
    python plot_all_probes.py [--results-dir results/oeq_probe_aita_transfer_sentence]
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent

# Same Okabe-Ito colorblind-safe triple used throughout this task's plots.
TRAIN_COLOR = "#0072B2"
HELDOUT_COLOR = "#009E73"
AITA_COLOR = "#D55E00"

COMPONENTS = ["mha", "mlp", "residual"]
TITLES = {
    "mha": "MHA (every head)",
    "mlp": "MLP",
    "residual": "Residual",
}


def load_pickle(results_dir: Path, name: str) -> dict:
    with open(results_dir / name, "rb") as f:
        return pickle.load(f)


def load_json(results_dir: Path, name: str) -> dict:
    with open(results_dir / name) as f:
        return json.load(f)


def _mha_points(d: dict) -> tuple:
    """dict keyed by (layer, head) tuples -> (layers, values), for scatter."""
    layers = [k[0] for k in d]
    values = list(d.values())
    return layers, values


def _mha_points_from_json(d: dict) -> tuple:
    """Same, but for the json-loaded dicts where keys are '(layer, head)' strings."""
    layers, values = [], []
    for k, v in d.items():
        layer = int(k.strip("()").split(",")[0])
        layers.append(layer)
        values.append(v)
    return layers, values


def _layer_points(d: dict) -> tuple:
    """dict keyed by plain layer int/str (MLP/residual) -> (layers, values), sorted
    numerically -- json.load always gives string keys ("0","1",...,"10",...), which
    sort lexicographically ("0","1","10","11",...,"2",...) unless cast to int first."""
    layers = sorted(d.keys(), key=int)
    return [int(l) for l in layers], [d[l] for l in layers]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir", type=Path,
        default=HERE / "results" / "cross_domain_transfer" / "oeq_probe_aita_transfer_sentence",
    )
    args = parser.parse_args()
    results_dir = args.results_dir

    train_acc = {c: load_pickle(results_dir, f"{c}_accuracy.pkl") for c in COMPONENTS}
    heldout_acc = load_json(results_dir, "oeq_heldout_accuracy.json")
    aita_acc = load_json(results_dir, "aita_transfer_accuracy.json")

    n_layers = max(l for l in train_acc["mlp"]) + 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for ax, component in zip(axes, COMPONENTS):
        if component == "mha":
            tl, tv = _mha_points(train_acc[component])
            hl, hv = _mha_points_from_json(heldout_acc[component])
            al, av = _mha_points_from_json(aita_acc[component])
            marker_kwargs = dict(s=10, alpha=0.35, linewidths=0)
        else:
            tl, tv = _layer_points(train_acc[component])
            hl, hv = _layer_points(heldout_acc[component])
            al, av = _layer_points(aita_acc[component])
            marker_kwargs = dict(s=18, alpha=0.85, linewidths=0)

        ax.scatter(tl, tv, color=TRAIN_COLOR, label="Train (5-fold CV)", **marker_kwargs)
        ax.scatter(hl, hv, color=HELDOUT_COLOR, label="OEQ held-out", **marker_kwargs)
        ax.scatter(al, av, color=AITA_COLOR, label="AITA transfer", **marker_kwargs)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance", zorder=0)
        ax.set_xlabel("Layer")
        ax.set_xlim(-1, n_layers)
        ax.set_title(TITLES[component])
        if component == "mha":
            ax.set_ylabel("Accuracy (every individual probe)")
        ax.legend(fontsize=8.5, loc="lower right")

    axes[0].set_ylim(0, 1.05)

    pooling = "sentence-level" if "sentence" in results_dir.name else "whole-sequence"
    fig.suptitle(f"OEQ probe ({pooling} pooling): every trained probe's accuracy, by layer")
    fig.tight_layout()

    out_path = results_dir / "plots" / "all_probes_by_layer.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
