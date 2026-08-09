#!/usr/bin/env python3
"""
Bootstrap confidence intervals on the NC1 neural-collapse metric (see
mixture_category_residual_probe_pipeline.py), to quantify how much the
category (C=4) vs label (C=2) vs category_label (C=8) NC1 gap depends on
sampling noise rather than being a robust effect.

Extracts residual activations ONCE (same lean extraction as
mixture_residual_probe_pipeline.py), then for each (grouping, layer):
  - computes the point-estimate NC1 on the real data (no resampling)
  - runs --n-bootstrap iterations of: for each class, resample that class's
    original member indices WITH replacement (same per-class N as the real
    data, so class balance is fixed across iterations -- this isolates
    within-class sampling noise in the activation distribution, not
    class-size variation), recompute NC1 on the resampled set
  - reports the point estimate plus the bootstrap mean and a percentile CI

--layers lets you restrict to a subset (e.g. "0,5,10,15,20,25,31") to bound
runtime -- each bootstrap iteration needs a fresh pinv() on a (D,D) matrix,
which is the dominant cost once activations are cached in memory.

Usage:
    python bootstrap_nc1_pipeline.py \
        --input results/generations/sycophancy_mixture/mixture.jsonl \
        --output-dir results/probing \
        --groupings category,label,category_label \
        --layers 0,5,10,15,20,25,31 \
        --n-bootstrap 50 --seed 0
"""
import argparse
import json
import sys
import time
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
from mixture_residual_probe_pipeline import collect_residual_only


def load_mixture(path: Path) -> tuple:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    texts = [r["text"] for r in rows]
    labels = np.array([r["label"] for r in rows], dtype=np.float32)
    categories = [r["category"] for r in rows]
    return texts, labels, categories


def build_groups(grouping: str, categories: list, labels: np.ndarray) -> list:
    if grouping == "category":
        return list(categories)
    if grouping == "label":
        return ["sycophantic" if l == 1 else "not_sycophantic" for l in labels]
    if grouping == "category_label":
        return [f"{c}_{'sycophantic' if l == 1 else 'not_sycophantic'}" for c, l in zip(categories, labels)]
    raise ValueError(f"unknown grouping {grouping!r}")


def compute_nc1(X: torch.Tensor, groups_arr: np.ndarray, unique_groups: list) -> float:
    """X: (n_examples, D) on some device. Returns a single NC1 scalar."""
    C = len(unique_groups)
    N = X.shape[0]
    D = X.shape[1]
    device = X.device
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
    return (torch.trace(Sigma_W @ Sigma_B_pinv) / C).item()


def bootstrap_nc1_for_layer(
    layer_activations: np.ndarray, groups: list, n_bootstrap: int, seed: int, device: str,
) -> dict:
    """layer_activations: (n_examples, D) for one layer. Returns point estimate + bootstrap stats."""
    groups_arr = np.array(groups)
    unique_groups = sorted(set(groups))
    class_indices = {g: np.nonzero(groups_arr == g)[0] for g in unique_groups}

    X_full = torch.tensor(layer_activations, dtype=torch.float32, device=device)
    point_estimate = compute_nc1(X_full, groups_arr, unique_groups)
    del X_full

    rng = np.random.default_rng(seed)
    boot_values = []
    for _ in range(n_bootstrap):
        resampled_idx = np.concatenate([
            rng.choice(idx, size=len(idx), replace=True) for idx in class_indices.values()
        ])
        resampled_groups = groups_arr[resampled_idx]
        X_boot = torch.tensor(layer_activations[resampled_idx], dtype=torch.float32, device=device)
        boot_values.append(compute_nc1(X_boot, resampled_groups, unique_groups))
        del X_boot

    boot_values = np.array(boot_values)
    return {
        "point_estimate": point_estimate,
        "boot_mean": float(boot_values.mean()),
        "boot_std": float(boot_values.std()),
        "ci_lower_2.5": float(np.percentile(boot_values, 2.5)),
        "ci_upper_97.5": float(np.percentile(boot_values, 97.5)),
        "n_bootstrap": n_bootstrap,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default=str(SYCOPHANCY_DIR / "results" / "generations" / "sycophancy_mixture" / "mixture.jsonl"))
    parser.add_argument("--output-dir", type=str, default=str(SYCOPHANCY_DIR / "results" / "probing"))
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--groupings", type=str, default="category,label,category_label")
    parser.add_argument("--layers", type=str, default="", help="Comma-separated layer indices, e.g. '0,5,10'. Empty = all layers.")
    parser.add_argument("--n-bootstrap", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timing-test", action="store_true",
                         help="Only run 1 layer x 1 grouping x 3 bootstrap iters, print per-iteration timing, then exit -- "
                              "use this first to estimate full-run wall-clock before committing to it.")
    args = parser.parse_args()

    texts, labels, categories = load_mixture(Path(args.input))
    print(f"Loaded {len(texts)} rows from {args.input}")

    print(f"\nLoading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    n_layers = model_config["n_layers"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nExtracting residual-stream activations for {len(texts)} examples ({n_layers} layers)...")
    residual_activations = collect_residual_only(model, tokenizer, texts, model_config)
    cleanup_model(model, tokenizer)

    groupings = args.groupings.split(",")
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(n_layers))

    if args.timing_test:
        grouping = groupings[0]
        layer = layers[0]
        groups = build_groups(grouping, categories, labels)
        print(f"\nTiming test: grouping={grouping}, layer={layer}, 3 bootstrap iterations...")
        t0 = time.time()
        result = bootstrap_nc1_for_layer(residual_activations[layer], groups, n_bootstrap=3, seed=args.seed, device=device)
        elapsed = time.time() - t0
        print(f"3 bootstrap iters + point estimate took {elapsed:.2f}s ({elapsed/4:.2f}s/iteration incl. point estimate)")
        print(result)
        est_full = elapsed / 4 * (args.n_bootstrap + 1) * len(layers) * len(groupings)
        print(f"\nEstimated full run ({len(layers)} layers x {len(groupings)} groupings x {args.n_bootstrap} bootstrap): "
              f"{est_full/60:.1f} minutes")
        return

    all_results = {}
    for grouping in groupings:
        groups = build_groups(grouping, categories, labels)
        print(f"\n{'='*70}\nGrouping: {grouping}\n{'='*70}")
        grouping_results = {}
        for layer in layers:
            t0 = time.time()
            result = bootstrap_nc1_for_layer(
                residual_activations[layer], groups, args.n_bootstrap, args.seed, device,
            )
            elapsed = time.time() - t0
            grouping_results[layer] = result
            print(f"  layer {layer:2d}: point={result['point_estimate']:.4f} "
                  f"boot_mean={result['boot_mean']:.4f} "
                  f"CI=[{result['ci_lower_2.5']:.4f}, {result['ci_upper_97.5']:.4f}] "
                  f"({elapsed:.1f}s)")
        all_results[grouping] = grouping_results

    out_path = Path(args.output_dir) / "nc1_bootstrap_ci.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "layers": layers, "n_bootstrap": args.n_bootstrap, "seed": args.seed,
            "results": all_results,
        }, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
