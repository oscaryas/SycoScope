#!/usr/bin/env python3
"""
Trains one residual-stream-only probe set PER sycophancy category, using
each category's own rows already inside results/generations/
sycophancy_mixture/mixture.jsonl -- NOT the full raw per-category datasets
(category_residual_probe_pipeline.py does that, much larger/slower). This is
the smaller ask: same 2,840 examples the combined mixture probe was already
trained on (mixture_residual_probe_pipeline.py), just grouped by category
before training instead of pooled into one probe.

Each category subset is already exactly balanced within itself (355
positive / 355 negative, by construction of build_sycophancy_mixture.py),
so there's nothing to undersample/upweight -- balance_method is accepted
for consistency with the other probe pipelines but is a no-op here.

Activations are extracted ONCE for all 2,840 rows (one forward pass per
example, same lean residual-only extraction as mixture_residual_probe_
pipeline.py -- no MHA/MLP hooks), then sliced by category for four
independent training passes.

Also computes the NC1 neural-collapse metric (Papyan, Han & Donoho 2020) per
residual layer, treating the 4 categories as classes:
    Sigma_W = (1/N) sum_c sum_{i in c} (x_i - mu_c)(x_i - mu_c)^T   (within-class covariance)
    Sigma_B = (1/C) sum_c (mu_c - mu_G)(mu_c - mu_G)^T              (between-class covariance)
    NC1 = (1/C) * trace(Sigma_W @ pinv(Sigma_B))
Lower NC1 = activations collapse more tightly around their own category's
mean relative to how spread apart the category means are (the "collapse" the
metric is named for). Uses the same extracted activations as the per-category
probes, no extra forward passes -- just a linear-algebra pass over each
layer's already-in-memory (N, hidden_dim) array, on GPU if available.

Usage:
    python mixture_category_residual_probe_pipeline.py \
        --input results/generations/sycophancy_mixture/mixture.jsonl \
        --output-dir results/probing --model meta-llama/Meta-Llama-3-8B-Instruct
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SYCOPHANCY_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
for p in (SYCOPHANCY_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.model import load_model_and_tokenizer, cleanup as cleanup_model
from sycophancy_model_registry import get_model_config
from sycophancy_probes import train_residual_probes, save_probe_results
from mixture_residual_probe_pipeline import collect_residual_only


def load_mixture(path: Path) -> tuple:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    texts = [r["text"] for r in rows]
    labels = np.array([r["label"] for r in rows], dtype=np.float32)
    categories = [r["category"] for r in rows]
    return texts, labels, categories


def compute_nc1_by_layer(activations: np.ndarray, groups: list, n_layers: int, device: str = "cpu") -> dict:
    """
    NC1 neural-collapse metric per layer, treating `groups` as classes.
    activations: (n_layers, n_examples, hidden_dim). Returns
    {layer: {"NC1": float, "trace_within": float, "trace_between": float}}.
    """
    unique_groups = sorted(set(groups))
    C = len(unique_groups)
    groups_arr = np.array(groups)
    N = len(groups)

    results = {}
    for layer in range(n_layers):
        X = torch.tensor(activations[layer], dtype=torch.float32, device=device)  # (N, D)
        D = X.shape[1]
        global_mean = X.mean(dim=0)

        Sigma_W = torch.zeros(D, D, device=device)
        class_means = []
        for g in unique_groups:
            mask = torch.tensor(groups_arr == g, device=device)
            Xg = X[mask]
            mu_g = Xg.mean(dim=0)
            class_means.append(mu_g)
            diff = Xg - mu_g
            Sigma_W += diff.T @ diff
        Sigma_W /= N

        Sigma_B = torch.zeros(D, D, device=device)
        for mu_g in class_means:
            d = (mu_g - global_mean).unsqueeze(1)
            Sigma_B += d @ d.T
        Sigma_B /= C

        Sigma_B_pinv = torch.linalg.pinv(Sigma_B)
        NC1 = (torch.trace(Sigma_W @ Sigma_B_pinv) / C).item()

        results[layer] = {
            "NC1": NC1,
            "trace_within": torch.trace(Sigma_W).item(),
            "trace_between": torch.trace(Sigma_B).item(),
        }
        print(f"  Layer {layer}: NC1={NC1:.4f}  trace(Sigma_W)={results[layer]['trace_within']:.2f}  "
              f"trace(Sigma_B)={results[layer]['trace_between']:.2f}")

        del X, Sigma_W, Sigma_B, Sigma_B_pinv
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default=str(SYCOPHANCY_DIR / "results" / "generations" / "sycophancy_mixture" / "mixture.jsonl"))
    parser.add_argument("--output-dir", type=str, default=str(SYCOPHANCY_DIR / "results" / "probing"))
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--balance-method", type=str, default="undersample", choices=["undersample", "upweight"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nc1-grouping", type=str, default="category", choices=["category", "category_label", "label"],
                         help="'category': 4 classes (sypr/social/moral/are_you_sure), ignoring sycophancy label. "
                              "'category_label': 8 classes, category x label (sycophantic/not_sycophantic) -- "
                              "moral's not_sycophantic keeps the mixture's existing broad definition "
                              "(includes Mixed + Both-YTA + Refused, unchanged from the labels already in --input). "
                              "'label': 2 classes (sycophantic/not_sycophantic), pooling all categories together -- "
                              "directly comparable to 'category' (both are single-factor, C=4 vs C=2).")
    parser.add_argument("--skip-probes", action="store_true",
                         help="Only extract activations and compute NC1 -- skip the 4 per-category probe trainings "
                              "(already done in a prior run and unaffected by --nc1-grouping).")
    args = parser.parse_args()

    texts, labels, categories = load_mixture(Path(args.input))
    print(f"Loaded {len(texts)} rows from {args.input}")
    print(f"Category breakdown: {dict(Counter(categories))}")

    print(f"\nLoading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    model_config["model_name"] = args.model
    n_layers = model_config["n_layers"]

    print(f"\nExtracting residual-stream activations for {len(texts)} examples ({n_layers} layers)...")
    residual_activations = collect_residual_only(model, tokenizer, texts, model_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cleanup_model(model, tokenizer)

    if args.nc1_grouping == "category":
        groups = categories
        classes = sorted(set(categories))
    elif args.nc1_grouping == "label":
        groups = ["sycophantic" if l == 1 else "not_sycophantic" for l in labels]
        classes = sorted(set(groups))
    else:
        groups = [f"{c}_{'sycophantic' if l == 1 else 'not_sycophantic'}" for c, l in zip(categories, labels)]
        classes = sorted(set(groups))

    print(f"\n{'='*70}\nNC1 neural-collapse metric by layer (grouping={args.nc1_grouping}, {len(classes)} classes: {classes})\n{'='*70}")
    nc1_results = compute_nc1_by_layer(residual_activations, groups, n_layers, device=device)
    nc1_filename = {
        "category": "category_nc1_by_layer.json",
        "category_label": "category_label_nc1_by_layer.json",
        "label": "label_nc1_by_layer.json",
    }[args.nc1_grouping]
    nc1_path = Path(args.output_dir) / nc1_filename
    nc1_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nc1_path, "w") as f:
        json.dump({"grouping": args.nc1_grouping, "classes": classes, "by_layer": nc1_results}, f, indent=2)
    print(f"NC1 results written to {nc1_path}")

    if args.skip_probes:
        print("\n--skip-probes set, done.")
        return

    # residual_activations: (n_layers, n_examples, hidden_dim) -- slice the
    # examples axis (axis=1) per category, keep every layer.
    cat_indices = defaultdict(list)
    for i, c in enumerate(categories):
        cat_indices[c].append(i)

    all_metadata = {}
    for name, idx in cat_indices.items():
        idx = np.array(idx)
        cat_activations = residual_activations[:, idx, :]
        cat_labels = labels[idx]
        n_pos = int((cat_labels == 1).sum())
        n_neg = int((cat_labels == 0).sum())
        print(f"\n{'='*70}\nCategory: {name}  ({len(idx)} rows, {n_pos} positive / {n_neg} negative)\n{'='*70}")

        print(f"Training residual probes (balance_method={args.balance_method})...")
        res_acc, res_ci, res_states, res_auc, res_auc_ci = train_residual_probes(
            cat_activations, cat_labels, n_layers, balance_method=args.balance_method, seed=args.seed,
        )

        results = {
            "mha_accuracy": {}, "mha_ci": {}, "mha_states": {}, "mha_auc": {}, "mha_auc_ci": {},
            "mlp_accuracy": {}, "mlp_ci": {}, "mlp_states": {}, "mlp_auc": {}, "mlp_auc_ci": {},
            "residual_accuracy": res_acc, "residual_ci": res_ci, "residual_states": res_states,
            "residual_auc": res_auc, "residual_auc_ci": res_auc_ci,
        }
        out_dir = Path(args.output_dir) / f"{name}_residual"
        metadata = save_probe_results(results, str(out_dir), model_name=args.model)

        run_info = {
            "input": str(args.input), "category": name, "n_total": len(idx),
            "n_pos": n_pos, "n_neg": n_neg,
            "balance_method": args.balance_method, "model": args.model, "seed": args.seed,
            "note": "trained on this category's rows within sycophancy_mixture, not the full raw per-category dataset",
        }
        with open(out_dir / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)

        all_metadata[name] = metadata
        print(f"[{name}] Residual best: {metadata['residual_best_accuracy']:.3f} at layer {metadata['residual_best_key']}")

    summary_path = Path(args.output_dir) / "category_residual_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\nAll categories done. Summary written to {summary_path}")
    for name, meta in all_metadata.items():
        print(f"  {name}: acc={meta['residual_best_accuracy']:.3f} at layer {meta['residual_best_key']}, auc={meta['residual_best_auc']}")


if __name__ == "__main__":
    main()
