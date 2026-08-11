#!/usr/bin/env python3
"""
Fixes moral_residual (trained via mixture_category_residual_probe_pipeline.py)
which pooled residual activations over ONE forward pass on the original post
+ flipped story concatenated as a single ~1024-token-truncated text -- for
~39% of pairs, the "---" separator itself fell past the truncation cutoff,
meaning the flipped-story half never entered the activation at all.

Fix: TWO separate forward passes per pair -- one over the original post's
prompt+response, one over the flipped story's prompt+response, each pooled
independently (own 1024-token budget, no truncation stealing from the other
side) -- then AVERAGE the two resulting per-layer residual vectors into one
activation per pair. That averaged vector is what the probe trains on, still
against the same pair-level label (1 iff both verdicts are NTA).

Negative class is redefined from the original moral_residual/mixture run:
NOT "everything that isn't Both-NTA" (which lumped in Mixed -- one side NTA,
one side YTA -- as if that were "not sycophantic", when it actually reflects
the model discriminating between the two framings, arguably the opposite of
sycophantic-but-inconsistent). Negative here is specifically:
    Both YTA (model consistently un-sycophantic on both sides), OR
    Refused/unclear (either side's verdict is OTHER)
Mixed pairs are excluded entirely from training, not folded into either
class. Positive stays Both-NTA (all 370 available pairs). Since this breaks
the old run's exact-pair-match (negatives are drawn from a different pool
now), this uses ALL 370 Both-NTA positives against a random --seed sample of
the redefined negative pool (matched size), pulled directly from
AITA-NTA-FLIP_moral_sycophancy_judged.jsonl (both sides' prompt/response/
verdict per row, unlike the pre-concatenated moral_sycophancy_pooled/
pooled.jsonl) -- not from sycophancy_mixture at all.

Usage:
    python moral_avg_residual_probe_pipeline.py \
        --judged results/generations/AITA-NTA-FLIP_moral_sycophancy_judged.jsonl \
        --output-dir results/probing/moral_avg_residual \
        --model meta-llama/Meta-Llama-3-8B-Instruct
"""
import argparse
import json
import random
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


def select_moral_pairs(judged_path: Path, seed: int) -> dict:
    """
    Builds the positive/negative pools directly from verdicts (not from
    sycophancy_mixture, whose negative definition included Mixed):
      positive: original_post_verdict == NTA and flipped_story_verdict == NTA
      negative: Both YTA, or either side OTHER/unclear -- excludes Mixed
                (one side NTA, other YTA) entirely.
    Samples min(len(pos), len(neg)) from each pool with the given seed.
    Returns {row_id: (record, label)} for the sampled pairs.
    """
    rows = [json.loads(line) for line in open(judged_path, encoding="utf-8")]

    pos, neg, mixed_excluded = [], [], 0
    for r in rows:
        og, flip = r["original_post_verdict"], r["flipped_story_verdict"]
        if og == "NTA" and flip == "NTA":
            pos.append(r)
        elif (og == "NTA" and flip == "YTA") or (og == "YTA" and flip == "NTA"):
            mixed_excluded += 1
        else:
            neg.append(r)  # Both YTA, or either side OTHER

    print(f"Verdict pools in {judged_path}: {len(pos)} positive (Both NTA), "
          f"{len(neg)} negative (Both YTA / unclear), {mixed_excluded} Mixed (excluded)")

    n = min(len(pos), len(neg))
    rng = random.Random(seed)
    pos_sample = rng.sample(pos, n)
    neg_sample = rng.sample(neg, n)
    print(f"Sampled {n} positive / {n} negative (seed={seed})")

    selected = {}
    for r in pos_sample:
        selected[r["row_id"]] = (r, 1)
    for r in neg_sample:
        selected[r["row_id"]] = (r, 0)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judged", type=str, default=str(SYCOPHANCY_DIR / "results" / "generations" / "AITA-NTA-FLIP_moral_sycophancy_judged.jsonl"))
    parser.add_argument("--output-dir", type=str, default=str(SYCOPHANCY_DIR / "results" / "probing" / "moral_avg_residual"))
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--balance-method", type=str, default="undersample", choices=["undersample", "upweight"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    selected = select_moral_pairs(Path(args.judged), args.seed)
    row_ids = sorted(selected.keys())
    by_row_id = {rid: rec for rid, (rec, _) in selected.items()}
    labels = np.array([selected[rid][1] for rid in row_ids], dtype=np.float32)

    print(f"\nLoading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    model_config["model_name"] = args.model
    n_layers = model_config["n_layers"]

    og_texts = [build_chat_prompt(tokenizer, by_row_id[rid]["original_post_prompt"], system_prompt=None)
                + by_row_id[rid]["original_post_response"] for rid in row_ids]
    flip_texts = [build_chat_prompt(tokenizer, by_row_id[rid]["flipped_story_prompt"], system_prompt=None)
                  + by_row_id[rid]["flipped_story_response"] for rid in row_ids]

    print(f"\n[run 1/2] Extracting residual activations for {len(og_texts)} original-post responses...")
    og_activations = collect_residual_only(model, tokenizer, og_texts, model_config)

    print(f"\n[run 2/2] Extracting residual activations for {len(flip_texts)} flipped-story responses...")
    flip_activations = collect_residual_only(model, tokenizer, flip_texts, model_config)

    cleanup_model(model, tokenizer)

    avg_activations = (og_activations + flip_activations) / 2.0

    print(f"\nTraining residual probes on averaged activations (balance_method={args.balance_method})...")
    res_acc, res_ci, res_states, res_auc, res_auc_ci = train_residual_probes(
        avg_activations, labels, n_layers, balance_method=args.balance_method, seed=args.seed,
    )

    results = {
        "mha_accuracy": {}, "mha_ci": {}, "mha_states": {}, "mha_auc": {}, "mha_auc_ci": {},
        "mlp_accuracy": {}, "mlp_ci": {}, "mlp_states": {}, "mlp_auc": {}, "mlp_auc_ci": {},
        "residual_accuracy": res_acc, "residual_ci": res_ci, "residual_states": res_states,
        "residual_auc": res_auc, "residual_auc_ci": res_auc_ci,
    }
    metadata = save_probe_results(results, args.output_dir, model_name=args.model)

    run_info = {
        "judged": str(args.judged),
        "n_total": len(row_ids), "n_pos": int((labels == 1).sum()), "n_neg": int((labels == 0).sum()),
        "balance_method": args.balance_method, "model": args.model, "seed": args.seed,
        "positive_definition": "Both NTA (original_post_verdict == NTA and flipped_story_verdict == NTA)",
        "negative_definition": "Both YTA, or either side OTHER/unclear -- Mixed (one NTA, one YTA) pairs excluded entirely",
        "method": "two separate forward passes (original post response, flipped story response), "
                  "residual activations averaged per layer before training -- fixes moral_residual's "
                  "single-concatenated-text truncation issue (see run notes)",
    }
    with open(Path(args.output_dir) / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"\nDone. Residual best: {metadata['residual_best_accuracy']:.3f} at layer {metadata['residual_best_key']}")


if __name__ == "__main__":
    main()
