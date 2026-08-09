#!/usr/bin/env python3
"""
Trains residual-stream-only linear probes on results/generations/
sycophancy_mixture/mixture.jsonl (the balanced 4-way sycophancy/not-
sycophancy mixture built by build_sycophancy_mixture.py).

Unlike oeq_probe_pipeline.py / sypr_probe_pipeline.py, which call
sycophancy_probes.collect_activations (always extracts MHA + MLP + residual
via register_hooks, then trains all three), this skips MHA/MLP entirely --
both the hook registration and the probe training for them are wasted work
(1024 MHA probes + 32 MLP probes) when only the residual stream is wanted.
Residual activations come for free from output_hidden_states, no hooks
needed, so the lean extraction loop below only does that.

The mixture is already exactly balanced (n positive == n negative per
category, and therefore overall), so there's no undersample/upweight
distinction to sweep -- both would behave identically on already-balanced
data. Trained once with the (default) undersample balance_method.

Usage:
    python mixture_residual_probe_pipeline.py \
        --input results/generations/sycophancy_mixture/mixture.jsonl \
        --output-dir results/probing/sycophancy_mixture_residual \
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
from sycophancy_model_registry import get_model_config
from sycophancy_probes import _pool, train_residual_probes, save_probe_results


def load_mixture(path: Path) -> tuple:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    texts = [r["text"] for r in rows]
    labels = np.array([r["label"] for r in rows], dtype=np.float32)
    categories = [r["category"] for r in rows]
    return texts, labels, categories


def collect_residual_only(model, tokenizer, texts: list, model_config: dict, max_length: int = 1024, pooling: str = "mean") -> np.ndarray:
    """
    Residual-stream activations only -- no MHA/MLP hooks registered, since
    residual comes directly from output_hidden_states. Same pooling
    convention as sycophancy_probes.collect_activations (mean over the
    response span after the answer_token_id delimiter, or whole sequence if
    not found) so results are comparable to the MHA/MLP-inclusive pipelines.

    Returns (n_layers, n_examples, hidden_dim).
    """
    n_layers = model_config["n_layers"]
    answer_token_id = model_config.get("answer_token_id")
    device = next(model.parameters()).device

    all_res = []
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"]
            if str(device) != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)

            if answer_token_id is not None:
                token_list = input_ids[0].tolist()
                positions = [j for j, t in enumerate(token_list) if t == answer_token_id]
                pos = positions[-1] if positions else -1
            else:
                pos = -1

            res_example = np.zeros((n_layers, model_config["hidden_dim"]), dtype=np.float32)
            hidden_states = outputs.hidden_states
            for layer_idx in range(n_layers):
                hs = _pool(hidden_states[layer_idx + 1][0], pos, pooling).cpu().float().numpy().astype(np.float32)
                res_example[layer_idx] = hs
            all_res.append(res_example)

            if (i + 1) % 50 == 0:
                print(f"  Extracted {i+1}/{len(texts)} examples")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    res_arr = np.stack(all_res, axis=0)       # (n_examples, n_layers, hidden_dim)
    return res_arr.transpose(1, 0, 2)         # (n_layers, n_examples, hidden_dim)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default=str(SYCOPHANCY_DIR / "results" / "generations" / "sycophancy_mixture" / "mixture.jsonl"))
    parser.add_argument("--output-dir", type=str, default=str(SYCOPHANCY_DIR / "results" / "probing" / "sycophancy_mixture_residual"))
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--balance-method", type=str, default="undersample", choices=["undersample", "upweight"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    texts, labels, categories = load_mixture(Path(args.input))
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    print(f"Loaded {len(texts)} rows from {args.input} ({n_pos} positive / {n_neg} negative)")
    from collections import Counter
    print(f"Category breakdown: {dict(Counter(categories))}")

    print(f"\nLoading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    model_config["model_name"] = args.model
    n_layers = model_config["n_layers"]

    print(f"\nExtracting residual-stream activations for {len(texts)} examples ({n_layers} layers)...")
    residual_activations = collect_residual_only(model, tokenizer, texts, model_config)

    cleanup_model(model, tokenizer)

    print(f"\nTraining residual probes ({n_layers} layers, balance_method={args.balance_method})...")
    res_acc, res_ci, res_states, res_auc, res_auc_ci = train_residual_probes(
        residual_activations, labels, n_layers, balance_method=args.balance_method, seed=args.seed,
    )

    results = {
        "mha_accuracy": {}, "mha_ci": {}, "mha_states": {}, "mha_auc": {}, "mha_auc_ci": {},
        "mlp_accuracy": {}, "mlp_ci": {}, "mlp_states": {}, "mlp_auc": {}, "mlp_auc_ci": {},
        "residual_accuracy": res_acc, "residual_ci": res_ci, "residual_states": res_states,
        "residual_auc": res_auc, "residual_auc_ci": res_auc_ci,
    }
    metadata = save_probe_results(results, args.output_dir, model_name=args.model)

    run_info = {
        "input": str(args.input),
        "n_total": len(texts),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "category_breakdown": dict(Counter(categories)),
        "balance_method": args.balance_method,
        "model": args.model,
        "seed": args.seed,
    }
    with open(Path(args.output_dir) / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"\nDone. Residual best: {metadata['residual_best_accuracy']:.3f} at layer {metadata['residual_best_key']}")


if __name__ == "__main__":
    main()
