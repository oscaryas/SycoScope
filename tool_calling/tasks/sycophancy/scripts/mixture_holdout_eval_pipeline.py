#!/usr/bin/env python3
"""
Genuine held-out evaluation of the residual probe on sycophancy_mixture --
unlike sycophancy_probes.train_probe (used everywhere else in this batch of
runs), which does 5-fold CV for reported accuracy but then REFITS the saved
probe on 100% of the data (no example is ever truly held out from the
deployed probe), this script does a real train/test split: fits the probe
on train only, then reports per-example correct/incorrect on test examples
the probe never saw during fitting.

Split is stratified by (category, label) -- 8 strata of 355 examples each
in the full mixture -- so both train and test stay exactly balanced across
all 4 categories and both labels, not just balanced in aggregate.

Uses the same moral-fix-aware extraction as mixture_residual_probe_pipeline_
v2.py: moral rows get two averaged forward passes (original post, flipped
story), other rows get one.

Usage:
    python mixture_holdout_eval_pipeline.py \
        --mixture results/generations/sycophancy_mixture/mixture.jsonl \
        --judged results/generations/AITA-NTA-FLIP_moral_sycophancy_judged.jsonl \
        --layer 14 --test-frac 0.2 \
        --output-dir results/probing/mixture_holdout_eval \
        --model meta-llama/Meta-Llama-3-8B-Instruct
"""
import argparse
import json
import random
import sys
from collections import defaultdict
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
from sycophancy_probes import _fit_probe
from mixture_residual_probe_pipeline import collect_residual_only


def load_mixture(path: Path) -> list:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def load_judged_by_row_id(path: Path) -> dict:
    return {rec["row_id"]: rec for line in open(path, encoding="utf-8") for rec in [json.loads(line)]}


def stratified_split(rows: list, test_frac: float, seed: int) -> tuple:
    """Splits indices into (train_idx, test_idx), stratified by (category, label)."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for i, r in enumerate(rows):
        strata[(r["category"], r["label"])].append(i)

    train_idx, test_idx = [], []
    for key, idx in strata.items():
        idx = idx[:]
        rng.shuffle(idx)
        n_test = round(len(idx) * test_frac)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return sorted(train_idx), sorted(test_idx)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mixture", type=str, default=str(SYCOPHANCY_DIR / "results" / "generations" / "sycophancy_mixture" / "mixture.jsonl"))
    parser.add_argument("--judged", type=str, default=str(SYCOPHANCY_DIR / "results" / "generations" / "AITA-NTA-FLIP_moral_sycophancy_judged.jsonl"))
    parser.add_argument("--output-dir", type=str, default=str(SYCOPHANCY_DIR / "results" / "probing" / "mixture_holdout_eval"))
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--layer", type=int, default=14, help="Residual layer to evaluate (14 was the best layer for the moral-fixed combined probe).")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--n-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_mixture(Path(args.mixture))
    labels = np.array([r["label"] for r in rows], dtype=np.float32)
    print(f"Loaded {len(rows)} rows from {args.mixture}")

    train_idx, test_idx = stratified_split(rows, args.test_frac, args.seed)
    print(f"Split: {len(train_idx)} train / {len(test_idx)} test (stratified by category x label)")
    from collections import Counter
    print(f"  train category breakdown: {dict(Counter(rows[i]['category'] for i in train_idx))}")
    print(f"  test category breakdown:  {dict(Counter(rows[i]['category'] for i in test_idx))}")
    print(f"  train label balance: {dict(Counter(int(labels[i]) for i in train_idx))}")
    print(f"  test label balance:  {dict(Counter(int(labels[i]) for i in test_idx))}")

    moral_idx = [i for i, r in enumerate(rows) if r["category"] == "moral"]
    other_idx = [i for i, r in enumerate(rows) if r["category"] != "moral"]
    judged_by_row_id = load_judged_by_row_id(Path(args.judged))

    print(f"\nLoading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    model_config["model_name"] = args.model
    n_layers = model_config["n_layers"]
    if not (0 <= args.layer < n_layers):
        raise ValueError(f"--layer must be in [0, {n_layers}), got {args.layer}")

    other_texts = [rows[i]["text"] for i in other_idx]
    print(f"\n[1/3] Extracting residual activations for {len(other_texts)} non-moral rows...")
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

    hidden_dim = model_config["hidden_dim"]
    combined = np.zeros((n_layers, len(rows), hidden_dim), dtype=np.float32)
    for slot, i in enumerate(other_idx):
        combined[:, i, :] = other_activations[:, slot, :]
    for slot, i in enumerate(moral_idx):
        combined[:, i, :] = moral_activations[:, slot, :]

    layer_activations = combined[args.layer]  # (n_rows, hidden_dim)
    X_train, y_train = layer_activations[train_idx], labels[train_idx]
    X_test, y_test = layer_activations[test_idx], labels[test_idx]

    print(f"\nFitting probe on train set only ({len(train_idx)} examples, layer {args.layer})...")
    probe = _fit_probe(X_train, y_train, input_dim=hidden_dim, n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr)

    with torch.no_grad():
        train_logits = probe(torch.FloatTensor(X_train)).numpy()
        test_logits = probe(torch.FloatTensor(X_test)).numpy()
    train_preds = (train_logits > 0).astype(np.float32)
    test_preds = (test_logits > 0).astype(np.float32)

    train_acc = float((train_preds == y_train).mean())
    test_acc = float((test_preds == y_test).mean())
    print(f"\nTrain accuracy (seen data): {train_acc:.3f}")
    print(f"Test accuracy (held-out, never seen during fitting): {test_acc:.3f}")

    per_example = []
    for slot, i in enumerate(test_idx):
        r = rows[i]
        per_example.append({
            "row_index": i,
            "category": r["category"],
            "true_label": int(y_test[slot]),
            "predicted_label": int(test_preds[slot]),
            "correct": bool(test_preds[slot] == y_test[slot]),
            "logit": float(test_logits[slot]),
            "text_snippet": r["text"][:300],
            "meta": r.get("meta", {}),
        })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "test_predictions.jsonl", "w") as f:
        for rec in per_example:
            f.write(json.dumps(rec) + "\n")

    by_category = defaultdict(lambda: {"n": 0, "n_correct": 0})
    for rec in per_example:
        by_category[rec["category"]]["n"] += 1
        by_category[rec["category"]]["n_correct"] += int(rec["correct"])
    category_summary = {
        cat: {"n": v["n"], "n_correct": v["n_correct"], "accuracy": v["n_correct"] / v["n"]}
        for cat, v in by_category.items()
    }

    misclassified = [rec for rec in per_example if not rec["correct"]]
    most_confident_wrong = sorted(misclassified, key=lambda r: abs(r["logit"]), reverse=True)[:20]

    summary = {
        "mixture": str(args.mixture), "layer": args.layer, "test_frac": args.test_frac, "seed": args.seed,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "train_accuracy": train_acc, "test_accuracy": test_acc,
        "n_test_correct": int(sum(r["correct"] for r in per_example)),
        "n_test_misclassified": len(misclassified),
        "accuracy_by_category": category_summary,
        "most_confidently_wrong_row_indices": [r["row_index"] for r in most_confident_wrong],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPer-category test accuracy:")
    for cat, v in category_summary.items():
        print(f"  {cat}: {v['n_correct']}/{v['n']} = {v['accuracy']:.3f}")
    print(f"\nWrote {out_dir / 'test_predictions.jsonl'} ({len(per_example)} rows)")
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
