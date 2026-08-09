#!/usr/bin/env python3
"""
End-to-end probe-based sycophancy-direction pipeline sourced from OEQ.jsonl.

Runs as a single script rather than a notebook: import the already-tracked
modules in this repo directly, write results straight to disk. This avoids
the Colab-notebook failure modes hit earlier this session (multi-cell
interactive drift, ~85KB-per-read cell-output caps forcing partial data
transfers) -- run this once with `python oeq_probe_pipeline.py`, wherever a
GPU + the Llama weights + an ANTHROPIC_API_KEY are available, and every
output lands on disk directly.

Label source: OEQ.jsonl's "validation" metric (ELEPHANT's own definition of
unconditional emotional validation -- the response emotionally validates the
user regardless of whether validation is warranted). This is the closest
existing "purely sycophancy" signal in this repo to what was asked for: it
scores every response independently (unlike moral sycophancy, which needs a
paired correctness judgment), so no new judge was needed --
social_sycophancy_judge.generate_social_sycophancy_labels already produces a
directly-trainable label set from OEQ.jsonl's cached generations (no fresh
generation needed for labeling, unlike the SyPR pipeline).

Usage:
    python oeq_probe_pipeline.py \
        --n-label 150 --cross-dataset-n 100 \
        --output-dir results/oeq_probe --model meta-llama/Meta-Llama-3-8B-Instruct
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import anthropic

HERE = Path(__file__).resolve().parent
SYCOPHANCY_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
for p in (SYCOPHANCY_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.model import load_model_and_tokenizer, cleanup as cleanup_model
from sycophancy_model_registry import get_model_config
from sycophancy_probes import (
    collect_activations,
    train_mha_probes,
    train_mlp_probes,
    train_residual_probes,
    save_probe_results,
    load_probe_results,
)
from social_sycophancy_judge import generate_social_sycophancy_labels, build_labeled_text
from cross_dataset_generalization import (
    DEFAULT_ALPHAS as ALPHAS,
    CROSS_DATASETS,
    run_generalization_sweep,
)

BALANCE_METHODS = ["undersample", "upweight"]


# ---------------------------------------------------------------------------
# Labeling + probe training (activations extracted once, both balance
# variants trained off the same cached activations -- avoids two full
# forward-pass sweeps over the model for what's otherwise identical data).
# ---------------------------------------------------------------------------

def build_labels(tokenizer, n_label: int) -> dict:
    result = generate_social_sycophancy_labels(tokenizer, metric="validation", n_examples=n_label)
    n_pos = sum(r["label"] == 1 for r in result["records"])
    n_neg = len(result["records"]) - n_pos
    result["n_pos"] = n_pos
    result["n_neg"] = n_neg
    return result


def train_all_variants(model, tokenizer, texts: list, labels: np.ndarray, model_config: dict, output_dir: Path) -> dict:
    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]

    print(f"Extracting activations for {len(texts)} labeled examples...")
    activations = collect_activations(model, tokenizer, texts, model_config)

    variant_metadata = {}
    for balance_method in BALANCE_METHODS:
        print(f"\n=== Training probes: balance_method={balance_method} ===")
        mha_acc, mha_ci, mha_states, mha_auc, mha_auc_ci = train_mha_probes(
            activations["mha"], labels, n_layers, n_heads, balance_method=balance_method
        )
        mlp_acc, mlp_ci, mlp_states, mlp_auc, mlp_auc_ci = train_mlp_probes(
            activations["mlp"], labels, n_layers, balance_method=balance_method
        )
        res_acc, res_ci, res_states, res_auc, res_auc_ci = train_residual_probes(
            activations["residual"], labels, n_layers, balance_method=balance_method
        )
        results = {
            "mha_accuracy": mha_acc, "mha_ci": mha_ci, "mha_states": mha_states, "mha_auc": mha_auc, "mha_auc_ci": mha_auc_ci,
            "mlp_accuracy": mlp_acc, "mlp_ci": mlp_ci, "mlp_states": mlp_states, "mlp_auc": mlp_auc, "mlp_auc_ci": mlp_auc_ci,
            "residual_accuracy": res_acc, "residual_ci": res_ci, "residual_states": res_states, "residual_auc": res_auc, "residual_auc_ci": res_auc_ci,
        }
        variant_dir = output_dir / f"oeq_probe_{balance_method}"
        metadata = save_probe_results(results, str(variant_dir), model_name=model_config.get("model_name", ""))
        variant_metadata[balance_method] = {"dir": variant_dir, "metadata": metadata}

    del activations
    gc.collect()
    return variant_metadata


# ---------------------------------------------------------------------------
# Dataset sampling + judging for the generalization sweep now lives in
# cross_dataset_generalization.py (shared with sypr_probe_pipeline.py and
# aita_dim_pipeline.py) -- run_generalization_sweep is imported above.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _best_mha_per_layer(mha_dict: dict, mha_acc_for_argmax: dict, n_layers: int) -> list:
    """mha_dict can be accuracy or CI/AUC -- always picks the value at whichever
    head is accuracy-best for that layer, so a CI/AUC line stays paired with the
    same head the accuracy line reports, not independently re-maximized."""
    out = []
    for l in range(n_layers):
        heads_at_layer = [(h, acc) for (layer, h), acc in mha_acc_for_argmax.items() if layer == l]
        if not heads_at_layer:
            out.append(None)
            continue
        best_head = max(heads_at_layer, key=lambda t: t[1])[0]
        out.append(mha_dict.get((l, best_head)))
    return out


def plot_accuracy_by_layer(probe_results: dict, balance_method: str, out_path: Path):
    n_layers = 1 + max(probe_results["mlp_accuracy"].keys())
    mha_acc_dict = probe_results["mha_accuracy"]
    mha_best_per_layer = _best_mha_per_layer(mha_acc_dict, mha_acc_dict, n_layers)
    mlp_by_layer = [probe_results["mlp_accuracy"][l] for l in range(n_layers)]
    res_by_layer = [probe_results["residual_accuracy"][l] for l in range(n_layers)]

    fig, ax = plt.subplots(figsize=(10, 6))
    layers = range(n_layers)
    lines = [("MHA (best head)", mha_best_per_layer, "mha_ci"), ("MLP", mlp_by_layer, "mlp_ci"), ("Residual", res_by_layer, "residual_ci")]
    for label, values, ci_key in lines:
        (line,) = ax.plot(layers, values, "o-", label=label)
        ci_dict = probe_results.get(ci_key)
        if ci_dict:
            if ci_key == "mha_ci":
                ci_pairs = _best_mha_per_layer(ci_dict, mha_acc_dict, n_layers)
            else:
                ci_pairs = [ci_dict.get(l) for l in layers]
            lo = [p[0] if p else v for p, v in zip(ci_pairs, values)]
            hi = [p[1] if p else v for p, v in zip(ci_pairs, values)]
            ax.fill_between(layers, lo, hi, alpha=0.15, color=line.get_color())
    ax.axhline(0.5, color="gray", linestyle="--", label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe accuracy (5-fold CV, shaded = Wilson CI)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"OEQ validation-sycophancy probe accuracy by layer ({balance_method})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_auc_roc_by_layer(probe_results: dict, balance_method: str, out_path: Path):
    n_layers = 1 + max(probe_results["mlp_accuracy"].keys())
    mha_acc_dict = probe_results["mha_accuracy"]
    mha_auc_dict = probe_results.get("mha_auc", {})
    mlp_auc_dict = probe_results.get("mlp_auc", {})
    res_auc_dict = probe_results.get("residual_auc", {})
    if not (mha_auc_dict or mlp_auc_dict or res_auc_dict):
        return

    mha_by_layer = _best_mha_per_layer(mha_auc_dict, mha_acc_dict, n_layers)
    mlp_by_layer = [mlp_auc_dict.get(l) for l in range(n_layers)]
    res_by_layer = [res_auc_dict.get(l) for l in range(n_layers)]

    fig, ax = plt.subplots(figsize=(10, 6))
    layers = range(n_layers)
    lines = [
        ("MHA (best head, by accuracy)", mha_by_layer, "mha_auc_ci"),
        ("MLP", mlp_by_layer, "mlp_auc_ci"),
        ("Residual", res_by_layer, "residual_auc_ci"),
    ]
    for label, values, ci_key in lines:
        values = [v if v is not None else 0.5 for v in values]
        (line,) = ax.plot(layers, values, "o-", label=label)
        ci_dict = probe_results.get(ci_key)
        if ci_dict:
            if ci_key == "mha_auc_ci":
                ci_pairs = _best_mha_per_layer(ci_dict, mha_acc_dict, n_layers)
            else:
                ci_pairs = [ci_dict.get(l) for l in layers]
            lo = [p[0] if p else v for p, v in zip(ci_pairs, values)]
            hi = [p[1] if p else v for p, v in zip(ci_pairs, values)]
            ax.fill_between(layers, lo, hi, alpha=0.15, color=line.get_color())
    ax.axhline(0.5, color="gray", linestyle="--", label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC-ROC (5-fold CV, shaded = bootstrap CI) -- diagnostic only")
    ax.set_ylim(0.3, 1.05)
    ax.set_title(f"OEQ validation-sycophancy probe AUC-ROC by layer ({balance_method})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cross_dataset(cross_results: dict, balance_method: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    nonzero_alphas = [a for a in ALPHAS if a != 0.0]
    for dataset, per_alpha in cross_results.items():
        baseline = per_alpha[0.0]["rate"]
        deltas = [per_alpha[a]["rate"] - baseline for a in nonzero_alphas]
        ax.plot(nonzero_alphas, deltas, "o-", label=dataset)
    ax.axhline(0.0, color="gray", linestyle="--")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Sycophancy-proxy rate delta vs. baseline")
    ax.set_title(f"OEQ-derived direction: cross-dataset generalization ({balance_method})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary writing
# ---------------------------------------------------------------------------

def write_summary(
    out_dir: Path,
    balance_method: str,
    model_name: str,
    label_info: dict,
    n_label: int,
    probe_results: dict,
    best_key: dict,
    cross_results: dict,
) -> None:
    lines = [
        f"# OEQ Probe -- Validation Sycophancy -- Summary (balance_method=\"{balance_method}\")",
        "",
        f"Model: `{model_name}` * Dataset: `SAE/results/OEQ.jsonl` (cached generations, no fresh "
        "generation needed for labeling) * label: ELEPHANT \"validation\" metric (does the response "
        "emotionally validate the user, regardless of whether validation is warranted) * direction "
        f"method: linear probe (`nn.Linear` + `BCEWithLogitsLoss`, Adam lr=0.001, 25 epochs, batch "
        f"size 25, 5-fold stratified CV) * `balance_method=\"{balance_method}\"`.",
        "",
        "## Class balance",
        "",
        f"**{label_info['n_pos']} validating (label=1) / {label_info['n_neg']} not-validating (label=0) "
        f"out of {label_info['n_judged']} judged -- {label_info['n_pos']/label_info['n_judged']*100:.1f}% / "
        f"{label_info['n_neg']/label_info['n_judged']*100:.1f}%.**",
        "",
        "## Plots",
        "",
        "- `plots/01_probe_accuracy_by_layer.png` -- MHA (best head)/MLP/Residual accuracy, all layers, "
        "shaded Wilson CI bands",
        "- `plots/02_auc_roc_by_layer.png` -- same layout, AUC-ROC (diagnostic only, direction selection "
        "stays accuracy-based), shaded bootstrap CI bands",
        "- `plots/03_cross_dataset_generalization.png` -- all 5 datasets (OEQ entry is a held-out slice, "
        "disjoint from the rows used to label/train), delta vs. baseline",
        "",
        "## Best direction",
        "",
        f"- **MHA: layer {best_key['layer']}, head {best_key['head']}, "
        f"accuracy={probe_results['metadata']['mha_best_accuracy']:.3f}**",
        f"- MLP: layer {probe_results['metadata']['mlp_best_key']}, "
        f"accuracy={probe_results['metadata']['mlp_best_accuracy']:.3f}",
        f"- Residual: layer {probe_results['metadata']['residual_best_key']}, "
        f"accuracy={probe_results['metadata']['residual_best_accuracy']:.3f}",
        "",
        "## Cross-dataset generalization (MHA best direction)",
        "",
        "| Dataset | Baseline | " + " | ".join(f"alpha={a}" for a in ALPHAS if a != 0.0) + " |",
        "|---|---|" + "---|" * (len(ALPHAS) - 1),
    ]
    for dataset, per_alpha in cross_results.items():
        baseline = per_alpha[0.0]["rate"]
        row = [dataset, f"{baseline*100:.2f}%"]
        for a in ALPHAS:
            if a == 0.0:
                continue
            delta = per_alpha[a]["rate"] - baseline
            row.append(f"{delta*100:+.2f}%")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        f"Full raw generations: `generations.jsonl`. Full probe checkpoints: "
        f"`mha_probe_weights.pth`/`mha_projection_stds.pt` (and mlp/residual equivalents), "
        f"covering every (layer, head)/layer, not just the selected best.",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-label", type=int, default=150)
    parser.add_argument("--cross-dataset-n", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=str(SYCOPHANCY_DIR / "results"))
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    model_config["model_name"] = args.model

    print(f"\nBuilding OEQ 'validation' sycophancy labels (n_label={args.n_label})...")
    label_info = build_labels(tokenizer, args.n_label)
    label_info["n_judged"] = len(label_info["records"])
    print(
        f"Class balance: {label_info['n_pos']} pos / {label_info['n_neg']} neg "
        f"out of {label_info['n_judged']} judged ({label_info['n_pos']/label_info['n_judged']*100:.1f}% positive)"
    )

    texts = [r["text"] for r in label_info["records"]]
    y = np.array([r["label"] for r in label_info["records"]], dtype=np.float32)

    variant_dirs = train_all_variants(model, tokenizer, texts, y, model_config, output_dir)

    client = anthropic.Anthropic()
    all_results = {
        "model_name": args.model,
        "n_label": args.n_label,
        "cross_dataset_n": args.cross_dataset_n,
        "label_balance": {
            "n_pos": label_info["n_pos"], "n_neg": label_info["n_neg"], "n_judged": label_info["n_judged"],
        },
        "variants": {},
    }

    for balance_method, info in variant_dirs.items():
        variant_dir = info["dir"]
        probe_results = load_probe_results(str(variant_dir))
        best_key = {
            "component": "mha",
            "layer": probe_results["metadata"]["mha_best_key"][0],
            "head": probe_results["metadata"]["mha_best_key"][1],
        }

        plots_dir = variant_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_accuracy_by_layer(probe_results, balance_method, plots_dir / "01_probe_accuracy_by_layer.png")
        plot_auc_roc_by_layer(probe_results, balance_method, plots_dir / "02_auc_roc_by_layer.png")

        generations_log = []
        print(f"\n=== Generalization sweep: {balance_method} (MHA {best_key['layer']}/{best_key['head']}) ===")
        cross_results = run_generalization_sweep(
            client, model, tokenizer, model_config, best_key, variant_dir,
            args.cross_dataset_n, args.n_label, f"oeq_{balance_method}", generations_log,
        )
        plot_cross_dataset(cross_results, balance_method, plots_dir / "03_cross_dataset_generalization.png")

        with open(variant_dir / "generations.jsonl", "w") as f:
            for rec in generations_log:
                f.write(json.dumps(rec) + "\n")

        cross_results_json = {
            dataset: {str(a): v for a, v in per_alpha.items()} for dataset, per_alpha in cross_results.items()
        }
        with open(variant_dir / "results.json", "w") as f:
            json.dump({
                "balance_method": balance_method,
                "best_key": best_key,
                "probe_metadata": probe_results["metadata"],
                "cross_dataset": cross_results_json,
            }, f, indent=2)

        write_summary(variant_dir, balance_method, args.model, label_info, args.n_label, probe_results, best_key, cross_results)

        all_results["variants"][balance_method] = {
            "dir": str(variant_dir), "best_key": best_key, "probe_metadata": probe_results["metadata"],
        }
        print(f"Wrote {variant_dir}/")

    with open(output_dir / "oeq_probe_run_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    cleanup_model(model, tokenizer)
    print(f"\nDone. Results under {output_dir}/")


if __name__ == "__main__":
    main()
