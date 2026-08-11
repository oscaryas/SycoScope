#!/usr/bin/env python3
"""
Retrains the combined (all 4 categories pooled) residual-stream probe from
mixture_residual_probe_pipeline.py, fixing the same issue moral_avg_residual_
probe_pipeline.py fixed for the moral-only probe: the mixture's `moral` rows
store one concatenated (original post + "---" + flipped story) text, pooled
over a single ~1024-token-truncated forward pass -- for ~39% of pairs the
flipped-story half never enters the activation at all.

Uses the EXACT SAME 2,840 rows/labels/categories already in
sycophancy_mixture/mixture.jsonl (not a different sample -- this isolates
"does the extraction bug affect the combined probe" from any change to
which examples are included). Only how the `moral` subset's activations are
computed changes:
  - sypr / social / are_you_sure rows: single forward pass over row["text"],
    same as before.
  - moral rows: row["meta"]["row_id"] is looked up in AITA-NTA-FLIP_moral_
    sycophancy_judged.jsonl to get the original post and flipped story's
    prompt/response separately, TWO forward passes are run (one per side),
    and the resulting per-layer residual vectors are averaged -- same fix
    as moral_avg_residual_probe_pipeline.py, just folded into the combined
    2,840-example extraction instead of a moral-only run.

Usage:
    python mixture_residual_probe_pipeline_v2.py \
        --mixture results/generations/sycophancy_mixture/mixture.jsonl \
        --judged results/generations/AITA-NTA-FLIP_moral_sycophancy_judged.jsonl \
        --output-dir results/probing/sycophancy_mixture_residual_moralfix \
        --model meta-llama/Meta-Llama-3-8B-Instruct
"""
import argparse
import json
import sys
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
from utils.inference import build_chat_prompt
from sycophancy_model_registry import get_model_config
from sycophancy_probes import train_residual_probes, save_probe_results
from mixture_residual_probe_pipeline import collect_residual_only


def load_mixture(path: Path) -> list:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def load_judged_by_row_id(path: Path) -> dict:
    return {rec["row_id"]: rec for line in open(path, encoding="utf-8") for rec in [json.loads(line)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mixture", type=str, default=str(SYCOPHANCY_DIR / "results" / "generations" / "sycophancy_mixture" / "mixture.jsonl"))
    parser.add_argument("--judged", type=str, default=str(SYCOPHANCY_DIR / "results" / "generations" / "AITA-NTA-FLIP_moral_sycophancy_judged.jsonl"))
    parser.add_argument("--output-dir", type=str, default=str(SYCOPHANCY_DIR / "results" / "probing" / "sycophancy_mixture_residual_moralfix"))
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--balance-method", type=str, default="undersample", choices=["undersample", "upweight"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_mixture(Path(args.mixture))
    labels = np.array([r["label"] for r in rows], dtype=np.float32)
    print(f"Loaded {len(rows)} rows from {args.mixture}")
    from collections import Counter
    print(f"Category breakdown: {dict(Counter(r['category'] for r in rows))}")

    moral_idx = [i for i, r in enumerate(rows) if r["category"] == "moral"]
    other_idx = [i for i, r in enumerate(rows) if r["category"] != "moral"]
    print(f"moral rows needing two-pass fix: {len(moral_idx)}, other rows (single-pass, unchanged): {len(other_idx)}")

    judged_by_row_id = load_judged_by_row_id(Path(args.judged))
    missing = [i for i in moral_idx if rows[i]["meta"]["row_id"] not in judged_by_row_id]
    if missing:
        raise ValueError(f"{len(missing)} moral row_ids not found in {args.judged}")

    print(f"\nLoading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    model_config["model_name"] = args.model
    n_layers = model_config["n_layers"]

    other_texts = [rows[i]["text"] for i in other_idx]
    print(f"\n[1/3] Extracting residual activations for {len(other_texts)} non-moral rows (single pass each)...")
    other_activations = collect_residual_only(model, tokenizer, other_texts, model_config)

    moral_og_texts = [
        build_chat_prompt(tokenizer, judged_by_row_id[rows[i]["meta"]["row_id"]]["original_post_prompt"], system_prompt=None)
        + judged_by_row_id[rows[i]["meta"]["row_id"]]["original_post_response"]
        for i in moral_idx
    ]
    moral_flip_texts = [
        build_chat_prompt(tokenizer, judged_by_row_id[rows[i]["meta"]["row_id"]]["flipped_story_prompt"], system_prompt=None)
        + judged_by_row_id[rows[i]["meta"]["row_id"]]["flipped_story_response"]
        for i in moral_idx
    ]
    print(f"\n[2/3] Extracting residual activations for {len(moral_og_texts)} moral rows, original-post side...")
    moral_og_activations = collect_residual_only(model, tokenizer, moral_og_texts, model_config)
    print(f"\n[3/3] Extracting residual activations for {len(moral_flip_texts)} moral rows, flipped-story side...")
    moral_flip_activations = collect_residual_only(model, tokenizer, moral_flip_texts, model_config)
    moral_activations = (moral_og_activations + moral_flip_activations) / 2.0

    cleanup_model(model, tokenizer)

    # Reassemble into original row order: (n_layers, n_total_rows, hidden_dim)
    hidden_dim = model_config["hidden_dim"]
    combined = np.zeros((n_layers, len(rows), hidden_dim), dtype=np.float32)
    for slot, i in enumerate(other_idx):
        combined[:, i, :] = other_activations[:, slot, :]
    for slot, i in enumerate(moral_idx):
        combined[:, i, :] = moral_activations[:, slot, :]

    print(f"\nTraining combined residual probe on all {len(rows)} rows (balance_method={args.balance_method})...")
    res_acc, res_ci, res_states, res_auc, res_auc_ci = train_residual_probes(
        combined, labels, n_layers, balance_method=args.balance_method, seed=args.seed,
    )

    results = {
        "mha_accuracy": {}, "mha_ci": {}, "mha_states": {}, "mha_auc": {}, "mha_auc_ci": {},
        "mlp_accuracy": {}, "mlp_ci": {}, "mlp_states": {}, "mlp_auc": {}, "mlp_auc_ci": {},
        "residual_accuracy": res_acc, "residual_ci": res_ci, "residual_states": res_states,
        "residual_auc": res_auc, "residual_auc_ci": res_auc_ci,
    }
    metadata = save_probe_results(results, args.output_dir, model_name=args.model)

    run_info = {
        "mixture": str(args.mixture), "judged": str(args.judged),
        "n_total": len(rows), "n_moral_fixed": len(moral_idx), "n_other_unchanged": len(other_idx),
        "n_pos": int((labels == 1).sum()), "n_neg": int((labels == 0).sum()),
        "balance_method": args.balance_method, "model": args.model, "seed": args.seed,
        "method": "same 2840 mixture rows/labels as mixture_residual_probe_pipeline.py's original run; "
                  "moral rows re-extracted as two separate forward passes (original post, flipped story) "
                  "averaged per layer, instead of one concatenated/truncated pass",
    }
    with open(Path(args.output_dir) / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"\nDone. Residual best: {metadata['residual_best_accuracy']:.3f} at layer {metadata['residual_best_key']}")


if __name__ == "__main__":
    main()
