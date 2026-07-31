#!/usr/bin/env python3
"""
End-to-end difference-in-means (DIM) sycophancy-direction pipeline sourced from
AITA-NTA-FLIP.jsonl, ported from moral_sycophancy_dim_rowid_pooled_colab_standalone.ipynb
into a standalone script for the same reasons oeq_probe_pipeline.py / sypr_probe_pipeline.py
were: reproducibility (argparse defaults land straight in results.json instead of drifting
silently from whatever a notebook's live cells happened to say) and avoiding this session's
recurring Colab-notebook failure modes.

Label source: moral sycophancy (ELEPHANT's "both NTA" definition, via
moral_sycophancy_judge.generate_moral_sycophancy_labels), row_id-pooled across both flip
sides and all generation samples before DIM training -- see sycophancy_dim.py's
iter_flip_pairs_all_samples / _average_blocks.

Sample sizes (Decision 1/2 in the n=100+ plan): n_examples=100 pairs for
labeling+training, n_eval_max=150 for the home-dataset (AITA-NTA-FLIP) held-out sweep
-- the actual "held-out set" this bump is about -- but cross_dataset_eval_max stays at
40 (NOT raised to 100+): the cross-dataset check multiplies by 4 other datasets x 13
alphas already, so a 3.3x bump there would be the worst cost/value tradeoff in this
plan. Similarly, only the single global-best direction is swept this run, not all 10
layer-bucket directions -- select_bucket_directions still computes and saves them
(cheap), just doesn't spend generation/judge budget sweeping each one.

Usage:
    python aita_dim_pipeline.py \
        --n-examples 100 --n-eval-max 150 --cross-dataset-eval-max 40 \
        --output-dir results/aita_dim --model meta-llama/Meta-Llama-3-8B-Instruct
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
import anthropic

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for p in (HERE, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.model import load_model_and_tokenizer, cleanup as cleanup_model
from sycophancy_model_registry import get_model_config
from sycophancy_probes import collect_activations, bootstrap_ci
from moral_sycophancy_judge import generate_moral_sycophancy_labels, build_labeled_text, DEFAULT_INPUT_PATH
from sycophancy_dim import (
    iter_flip_pairs_all_samples,
    _average_blocks,
    train_mha_dim,
    train_mlp_dim,
    train_residual_dim,
    save_dim_results,
)
from cross_dataset_generalization import DEFAULT_ALPHAS as ALPHAS, run_generalization_sweep

CROSS_DATASETS_OTHER = ["AITA-NTA-OG", "AITA-YTA", "OEQ", "SS"]
HOME_DATASET = "AITA-NTA-FLIP"


# ---------------------------------------------------------------------------
# Labeling + row_id-pooled activation collection
# ---------------------------------------------------------------------------

def build_labels(tokenizer, n_examples: int) -> dict:
    result = generate_moral_sycophancy_labels(tokenizer, n_pairs=n_examples)
    n_pos = sum(r["label"] == 1 for r in result["records"])
    result["n_pos"] = n_pos
    result["n_neg"] = len(result["records"]) - n_pos
    result["n_judged"] = len(result["records"])
    return result


def build_row_id_pooled_activations(model, tokenizer, model_config, label_result, data_path, pooling="mean"):
    """
    Ports notebook Cell 33 verbatim: dedupe the 2-per-row_id labeled records down to
    one label per row_id, flatten every sample from both flip sides into one text
    list, run collect_activations once, then average each row_id's block of samples
    back down to a single pooled activation vector per row_id.

    Returns (activations, labels, row_ids). activations mirrors collect_activations'
    own dict shape but with the examples axis now indexed by row_id instead of by raw
    sample: {"mha": (n_layers, n_heads, n_row_ids, head_dim), "mlp"/"residual":
    (n_layers, n_row_ids, hidden_dim)}.
    """
    records = label_result["records"]
    row_id_labels = {}
    for r in records:
        row_id_labels.setdefault(r["row_id"], r["label"])

    all_samples = iter_flip_pairs_all_samples(data_path)
    row_ids = [rid for rid in row_id_labels if rid in all_samples]
    n_skipped = len(row_id_labels) - len(row_ids)
    if n_skipped:
        print(f"Skipping {n_skipped} row_id(s) judged but missing from the all-samples index.")

    flat_texts, block_sizes = [], []
    for row_id in row_ids:
        sides = all_samples[row_id]
        side_recs = sides["original_post"] + sides["flipped_story"]
        flat_texts.extend(build_labeled_text(tokenizer, rec) for rec in side_recs)
        block_sizes.append(len(side_recs))

    labels = np.array([row_id_labels[rid] for rid in row_ids], dtype=np.float32)

    print(f"Extracting activations for {len(flat_texts)} raw generations ({len(row_ids)} row_ids)...")
    flat_activations = collect_activations(model, tokenizer, flat_texts, model_config, batch_size=1, pooling=pooling)

    activations = {
        "mha": _average_blocks(flat_activations["mha"], block_sizes, examples_axis=2),
        "mlp": _average_blocks(flat_activations["mlp"], block_sizes, examples_axis=1),
        "residual": _average_blocks(flat_activations["residual"], block_sizes, examples_axis=1),
    }
    avg_samples_per_rowid = len(flat_texts) / len(row_ids) if row_ids else 0.0
    print(
        f"Row_id-pooled activations for {len(row_ids)} row_ids (avg {avg_samples_per_rowid:.1f} "
        f"samples/row_id). Shapes: mha={activations['mha'].shape}, mlp={activations['mlp'].shape}, "
        f"residual={activations['residual'].shape}"
    )
    return activations, labels, row_ids


# ---------------------------------------------------------------------------
# DIM training
# ---------------------------------------------------------------------------

def train_all_dim_variants(activations: dict, labels: np.ndarray, model_config: dict, output_dir: Path, method: str = "cv_averaged", seed: int = 0) -> dict:
    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]

    print("Training MHA DIM directions...")
    mha_es, mha_states = train_mha_dim(activations["mha"], labels, n_layers, n_heads, method=method, seed=seed)
    print("Training MLP DIM directions...")
    mlp_es, mlp_states = train_mlp_dim(activations["mlp"], labels, n_layers, method=method, seed=seed)
    print("Training residual DIM directions...")
    res_es, res_states = train_residual_dim(activations["residual"], labels, n_layers, method=method, seed=seed)

    best_keys = save_dim_results(output_dir, mha_es, mha_states, mlp_es, mlp_states, res_es, res_states)

    best_component, best_key, best_effect_size = max(
        [
            ("mha", best_keys["mha_best_key"], mha_es[best_keys["mha_best_key"]]),
            ("mlp", best_keys["mlp_best_key"], mlp_es[best_keys["mlp_best_key"]]),
            ("residual", best_keys["residual_best_key"], res_es[best_keys["residual_best_key"]]),
        ],
        key=lambda x: abs(x[2]),
    )
    print(f"Best-separating component: {best_component} (cohen_d={best_effect_size:.3f}, key={best_key})")

    return {
        "n_layers": n_layers, "n_heads": n_heads,
        "mha_es": mha_es, "mha_states": mha_states,
        "mlp_es": mlp_es, "mlp_states": mlp_states,
        "residual_es": res_es, "residual_states": res_states,
        "best_component": best_component, "best_key": best_key, "best_effect_size": best_effect_size,
    }


def select_bucket_directions(mha_effect_sizes: dict, n_layers: int, top_k_per_bucket: int = 3) -> dict:
    """Top-k MHA directions by |Cohen's d| per early/middle/late layer bucket -- still
    computed/saved even though only the global-best direction gets swept this run
    (Decision 2), so a later run can pick this back up without recomputing DIM."""
    layer_buckets = {
        name: bucket.tolist()
        for name, bucket in zip(["early", "middle", "late"], np.array_split(np.arange(n_layers), 3))
    }
    bucket_directions = {}
    for bucket_name, bucket_layers in layer_buckets.items():
        bucket_layer_set = set(bucket_layers)
        candidates = sorted(
            (k for k in mha_effect_sizes if k[0] in bucket_layer_set),
            key=lambda k: abs(mha_effect_sizes[k]),
            reverse=True,
        )[:top_k_per_bucket]
        bucket_directions[bucket_name] = candidates
    return bucket_directions


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _best_mha_head_per_layer(mha_es: dict, n_layers: int) -> list:
    """For each layer, the head index with the largest |Cohen's d| -- DIM's own
    direction-selection criterion, mirroring the probe pipelines' accuracy-based
    per-layer head selection but keyed off effect size instead."""
    out = []
    for l in range(n_layers):
        heads_at_layer = [(h, es) for (layer, h), es in mha_es.items() if layer == l]
        out.append(max(heads_at_layer, key=lambda t: abs(t[1]))[0] if heads_at_layer else None)
    return out


def plot_effect_size_by_layer(dim_results: dict, out_path: Path):
    n_layers = dim_results["n_layers"]
    best_heads = _best_mha_head_per_layer(dim_results["mha_es"], n_layers)
    layers = list(range(n_layers))

    series = [
        ("MHA (best head)",
         [dim_results["mha_es"].get((l, h)) if h is not None else None for l, h in zip(layers, best_heads)],
         [dim_results["mha_states"].get((l, h)) if h is not None else None for l, h in zip(layers, best_heads)]),
        ("MLP", [dim_results["mlp_es"][l] for l in layers], [dim_results["mlp_states"][l] for l in layers]),
        ("Residual", [dim_results["residual_es"][l] for l in layers], [dim_results["residual_states"][l] for l in layers]),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, values, states in series:
        values = [v if v is not None else 0.0 for v in values]
        (line,) = ax.plot(layers, values, "o-", label=label)
        lo, hi = [], []
        for v, state in zip(values, states):
            fes = state["fold_effect_sizes"] if state else None
            bl, bh = bootstrap_ci(np.array(fes)) if fes else (v, v)
            lo.append(bl)
            hi.append(bh)
        ax.fill_between(layers, lo, hi, alpha=0.15, color=line.get_color())
    ax.axhline(0.0, color="gray", linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cohen's d (cv_averaged, shaded = bootstrap CI on fold effect sizes)")
    ax.set_title("AITA moral-sycophancy DIM effect size by layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_auc_roc_by_layer(dim_results: dict, out_path: Path):
    n_layers = dim_results["n_layers"]
    best_heads = _best_mha_head_per_layer(dim_results["mha_es"], n_layers)
    layers = list(range(n_layers))

    series = [
        ("MHA (best head, by cohen_d)",
         [dim_results["mha_states"].get((l, h)) if h is not None else None for l, h in zip(layers, best_heads)]),
        ("MLP", [dim_results["mlp_states"][l] for l in layers]),
        ("Residual", [dim_results["residual_states"][l] for l in layers]),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, states in series:
        values = [state["auc_roc"] if state else 0.5 for state in states]
        (line,) = ax.plot(layers, values, "o-", label=label)
        lo, hi = [], []
        for v, state in zip(values, states):
            fa = state["fold_aucs"] if state else None
            bl, bh = bootstrap_ci(np.array(fa)) if fa else (v, v)
            lo.append(bl)
            hi.append(bh)
        ax.fill_between(layers, lo, hi, alpha=0.15, color=line.get_color())
    ax.axhline(0.5, color="gray", linestyle="--", label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC-ROC (cv_averaged, shaded = bootstrap CI) -- diagnostic only")
    ax.set_ylim(0.3, 1.05)
    ax.set_title("AITA moral-sycophancy DIM AUC-ROC by layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_rate_deltas(results: dict, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    nonzero_alphas = [a for a in ALPHAS if a != 0.0]
    for dataset, per_alpha in results.items():
        baseline = per_alpha[0.0]["rate"]
        deltas = [per_alpha[a]["rate"] - baseline for a in nonzero_alphas]
        ax.plot(nonzero_alphas, deltas, "o-", label=dataset)
    ax.axhline(0.0, color="gray", linestyle="--")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Sycophancy-proxy rate delta vs. baseline")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary writing
# ---------------------------------------------------------------------------

def write_summary(out_dir: Path, model_name: str, label_info: dict, args, dim_results: dict, home_results: dict, cross_results: dict, bucket_directions: dict) -> None:
    best_key = dim_results["best_key"]
    best_state = dim_results[f"{dim_results['best_component']}_states"][best_key]
    home_baseline = home_results[HOME_DATASET][0.0]["rate"]

    lines = [
        "# AITA Moral-Sycophancy DIM -- Summary",
        "",
        f"Model: `{model_name}` * Dataset: `SAE/results/AITA-NTA-FLIP.jsonl` * label: ELEPHANT "
        "\"both NTA\" moral sycophancy definition * direction method: difference-in-means "
        f"(`cv_averaged`, 5-fold), row_id-pooled across both flip sides and all generation samples.",
        "",
        f"n_examples (training pairs) = {args.n_examples}, n_eval_max (home-dataset held-out) = "
        f"{args.n_eval_max}, **cross_dataset_eval_max = {args.cross_dataset_eval_max} "
        "(intentionally NOT raised to 100+ -- see Decision 1 in the n=100+ plan: it multiplies "
        "by 4 other datasets x 13 alphas already).**",
        "",
        "## Class balance",
        "",
        f"**{label_info['n_pos']} sycophantic (label=1) / {label_info['n_neg']} non-sycophantic "
        f"(label=0) out of {label_info['n_judged']} judged.**",
        "",
        "## Plots",
        "",
        "- `plots/01_effect_size_by_layer.png` -- MHA (best head)/MLP/Residual Cohen's d, "
        "shaded bootstrap CI bands",
        "- `plots/02_auc_roc_by_layer.png` -- same layout, AUC-ROC (diagnostic only, direction "
        "selection stays Cohen's-d-based), shaded bootstrap CI bands",
        "- `plots/03_home_dataset_sweep.png` -- AITA-NTA-FLIP held-out sweep (n="
        f"{args.n_eval_max}), delta vs. baseline",
        "- `plots/04_cross_dataset_generalization.png` -- the other 4 datasets (n="
        f"{args.cross_dataset_eval_max}), delta vs. baseline",
        "",
        "## Best direction",
        "",
        f"- **{dim_results['best_component'].upper()}: key={best_key}, "
        f"cohen_d={dim_results['best_effect_size']:.3f}, auc_roc={best_state['auc_roc']:.3f}**",
        f"- Home-dataset baseline rate (alpha=0.0): {home_baseline:.2%}",
        "",
        "## Layer-bucket directions (computed, NOT swept this run -- Decision 2)",
        "",
    ]
    for bucket_name, keys in bucket_directions.items():
        lines.append(f"- **{bucket_name}**: " + ", ".join(f"{k} (d={dim_results['mha_es'][k]:.3f})" for k in keys))

    lines += [
        "",
        "## Home-dataset held-out sweep (AITA-NTA-FLIP)",
        "",
        "| Baseline | " + " | ".join(f"alpha={a}" for a in ALPHAS if a != 0.0) + " |",
        "|---|" + "---|" * (len(ALPHAS) - 1),
    ]
    row = [f"{home_baseline*100:.2f}%"]
    for a in ALPHAS:
        if a == 0.0:
            continue
        row.append(f"{(home_results[HOME_DATASET][a]['rate'] - home_baseline)*100:+.2f}%")
    lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Cross-dataset generalization (other 4 datasets)",
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
            row.append(f"{(per_alpha[a]['rate'] - baseline)*100:+.2f}%")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "Full raw generations: `generations.jsonl`. Full DIM checkpoints: "
        "`{mha,mlp,residual}_dim_vectors.pt` / `{mha,mlp,residual}_dim_effect_size.pkl`, "
        "covering every (layer, head)/layer, not just the selected best.",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--n-eval-max", type=int, default=150)
    parser.add_argument("--cross-dataset-eval-max", type=int, default=40)
    parser.add_argument("--dim-method", type=str, default="cv_averaged", choices=["cv_averaged", "naive"])
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--label-source", type=str, default="moral", choices=["moral"])
    parser.add_argument("--output-dir", type=str, default=str(HERE / "results" / "aita_dim"))
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"Loading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    model_config["model_name"] = args.model

    print(f"\nBuilding AITA moral-sycophancy labels (n_examples={args.n_examples})...")
    label_info = build_labels(tokenizer, args.n_examples)
    print(
        f"Class balance: {label_info['n_pos']} pos / {label_info['n_neg']} neg "
        f"out of {label_info['n_judged']} judged"
    )

    activations, labels, row_ids = build_row_id_pooled_activations(
        model, tokenizer, model_config, label_info, DEFAULT_INPUT_PATH, pooling=args.pooling
    )

    dim_results = train_all_dim_variants(activations, labels, model_config, output_dir, method=args.dim_method, seed=args.seed)
    del activations
    gc.collect()

    bucket_directions = select_bucket_directions(dim_results["mha_es"], dim_results["n_layers"])

    plot_effect_size_by_layer(dim_results, plots_dir / "01_effect_size_by_layer.png")
    plot_auc_roc_by_layer(dim_results, plots_dir / "02_auc_roc_by_layer.png")

    client = anthropic.Anthropic()
    generations_log = []
    best_key = {"component": dim_results["best_component"], "layer": dim_results["best_key"][0] if dim_results["best_component"] == "mha" else dim_results["best_key"], "head": dim_results["best_key"][1] if dim_results["best_component"] == "mha" else None}

    print(f"\n=== Home-dataset ({HOME_DATASET}) held-out sweep (n={args.n_eval_max}) ===")
    home_results = run_generalization_sweep(
        client, model, tokenizer, model_config, best_key, output_dir,
        args.n_eval_max, args.n_examples, "aita_dim_global_best", generations_log,
        direction_fmt="dim", datasets=[HOME_DATASET], home_dataset=HOME_DATASET,
    )
    plot_home_sweep_path = plots_dir / "03_home_dataset_sweep.png"
    _plot_rate_deltas(home_results, f"AITA-DIM: home-dataset ({HOME_DATASET}) held-out sweep", plot_home_sweep_path)

    print(f"\n=== Cross-dataset generalization sweep (n={args.cross_dataset_eval_max}) ===")
    cross_results = run_generalization_sweep(
        client, model, tokenizer, model_config, best_key, output_dir,
        args.cross_dataset_eval_max, args.n_examples, "aita_dim_global_best", generations_log,
        direction_fmt="dim", datasets=CROSS_DATASETS_OTHER, home_dataset=HOME_DATASET,
    )
    _plot_rate_deltas(cross_results, "AITA-DIM: cross-dataset generalization", plots_dir / "04_cross_dataset_generalization.png")

    with open(output_dir / "generations.jsonl", "w") as f:
        for rec in generations_log:
            f.write(json.dumps(rec) + "\n")

    bucket_directions_json = {name: [list(k) for k in keys] for name, keys in bucket_directions.items()}
    home_results_json = {ds: {str(a): v for a, v in per_alpha.items()} for ds, per_alpha in home_results.items()}
    cross_results_json = {ds: {str(a): v for a, v in per_alpha.items()} for ds, per_alpha in cross_results.items()}

    results_json = {
        "model_name": args.model,
        "label_source": args.label_source,
        "dim_method": args.dim_method,
        "pooling": args.pooling,
        "n_examples": args.n_examples,
        "n_eval_max": args.n_eval_max,
        "cross_dataset_eval_max": args.cross_dataset_eval_max,
        "n_row_ids": len(row_ids),
        "label_balance": {"n_pos": label_info["n_pos"], "n_neg": label_info["n_neg"], "n_judged": label_info["n_judged"]},
        "best_component": dim_results["best_component"],
        "best_key": list(dim_results["best_key"]) if isinstance(dim_results["best_key"], tuple) else dim_results["best_key"],
        "best_effect_size": dim_results["best_effect_size"],
        "bucket_directions": bucket_directions_json,
        "home_dataset_sweep": home_results_json,
        "cross_dataset_sweep": cross_results_json,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    write_summary(output_dir, args.model, label_info, args, dim_results, home_results, cross_results, bucket_directions)

    cleanup_model(model, tokenizer)
    print(f"\nDone. Results under {output_dir}/")


if __name__ == "__main__":
    main()
