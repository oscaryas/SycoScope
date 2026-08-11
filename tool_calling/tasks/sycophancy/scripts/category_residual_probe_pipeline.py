#!/usr/bin/env python3
"""
Trains residual-stream-only linear probes separately for each sycophancy
category (not mixed, unlike mixture_residual_probe_pipeline.py) -- one probe
set per category, each on that category's own full pooled/generated dataset:
    sypr          results/generations/sypr_praise_llama31_full/checkpoint.jsonl
    are_you_sure  results/generations/are_you_sure_freeform/checkpoint.jsonl
    social        results/generations/social_sycophancy_pooled/pooled.jsonl
    moral         results/generations/moral_sycophancy_pooled/pooled.jsonl

Model is loaded once and reused across all four categories (each just needs
a fresh residual-activation extraction pass) rather than reloading the 8B
model four times. Same lean residual-only extraction as
mixture_residual_probe_pipeline.py -- no MHA/MLP hooks, no MHA/MLP probe
training.

Each category keeps its own natural class imbalance (sypr is ~1:10.7,
social ~7.9:1, moral ~1:3.3, are_you_sure ~1:2.1) and is balanced at train
time via train_probe's balance_method (default "undersample").

Usage:
    python category_residual_probe_pipeline.py \
        --output-dir results/probing --model meta-llama/Meta-Llama-3-8B-Instruct
"""
import argparse
import json
import sys
from collections import Counter
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

GENERATIONS_DIR = SYCOPHANCY_DIR / "results" / "generations"

CATEGORIES = {
    "sypr": GENERATIONS_DIR / "sypr_praise_llama31_full" / "checkpoint.jsonl",
    "are_you_sure": GENERATIONS_DIR / "are_you_sure_freeform" / "checkpoint.jsonl",
    "social": GENERATIONS_DIR / "social_sycophancy_pooled" / "pooled.jsonl",
    "moral": GENERATIONS_DIR / "moral_sycophancy_pooled" / "pooled.jsonl",
}


def load_dataset(path: Path) -> tuple:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    texts = [r["text"] for r in rows]
    labels = np.array([r["label"] for r in rows], dtype=np.float32)
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=str, default=str(SYCOPHANCY_DIR / "results" / "probing"))
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--balance-method", type=str, default="undersample", choices=["undersample", "upweight"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--categories", type=str, default=",".join(CATEGORIES.keys()),
                         help="Comma-separated subset of: " + ",".join(CATEGORIES.keys()))
    args = parser.parse_args()

    selected = args.categories.split(",")
    for name in selected:
        if name not in CATEGORIES:
            print(f"ERROR: unknown category {name!r}, must be one of {list(CATEGORIES)}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    model_config["model_name"] = args.model
    n_layers = model_config["n_layers"]

    all_metadata = {}
    for name in selected:
        path = CATEGORIES[name]
        print(f"\n{'='*70}\nCategory: {name}  ({path})\n{'='*70}")
        texts, labels = load_dataset(path)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        print(f"Loaded {len(texts)} rows ({n_pos} positive / {n_neg} negative)")

        print(f"Extracting residual-stream activations for {len(texts)} examples ({n_layers} layers)...")
        residual_activations = collect_residual_only(model, tokenizer, texts, model_config)

        print(f"Training residual probes (balance_method={args.balance_method})...")
        res_acc, res_ci, res_states, res_auc, res_auc_ci = train_residual_probes(
            residual_activations, labels, n_layers, balance_method=args.balance_method, seed=args.seed,
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
            "input": str(path), "category": name, "n_total": len(texts),
            "n_pos": n_pos, "n_neg": n_neg,
            "balance_method": args.balance_method, "model": args.model, "seed": args.seed,
        }
        with open(out_dir / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)

        all_metadata[name] = metadata
        print(f"[{name}] Residual best: {metadata['residual_best_accuracy']:.3f} at layer {metadata['residual_best_key']}")

        del residual_activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cleanup_model(model, tokenizer)

    summary_path = Path(args.output_dir) / "category_residual_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\nAll categories done. Summary written to {summary_path}")
    for name, meta in all_metadata.items():
        print(f"  {name}: acc={meta['residual_best_accuracy']:.3f} at layer {meta['residual_best_key']}, auc={meta['residual_best_auc']}")


if __name__ == "__main__":
    main()
