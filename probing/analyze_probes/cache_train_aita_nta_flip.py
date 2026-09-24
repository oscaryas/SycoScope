#!/usr/bin/env python3
"""Cache + train a moral-sycophancy residual probe on AITA-NTA-FLIP generations,
row_id-pooled: every generation sample from BOTH flip sides of a row_id gets
its own forward pass, and the per-layer residual vectors are averaged down to
one activation per row_id (the recipe of the retired aita_dim_pipeline.py,
with its DIM training replaced by the L2 C sweep).

Labels come from the moral-sycophancy LLM judge on sample_idx=0 of the first
--n-examples pairs (ELEPHANT's definition): 1 iff both sides are judged NTA,
else 0; pairs where either side is OTHER are skipped. Needs ANTHROPIC_API_KEY.
CV groups are row_ids.

Usage:
    python -m probing.analyze_probes.cache_train_aita_nta_flip \\
        --input-path SAE/results/AITA-NTA-FLIP.jsonl --n-examples 100 \\
        --model <model> --layers 8 16 24 --output weights.pkl
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from probing.analyze_probes import probes_core as core  # noqa: E402


def build_labels(tokenizer, input_path: Path, n_examples: int) -> dict:
    from probing.evaluations.judge.moral_sycophancy_judge import generate_moral_sycophancy_labels

    result = generate_moral_sycophancy_labels(tokenizer, input_path=input_path, n_pairs=n_examples)
    n_pos = sum(r["label"] == 1 for r in result["records"])
    result["n_pos"] = n_pos
    result["n_neg"] = len(result["records"]) - n_pos
    result["n_judged"] = len(result["records"])
    return result


def iter_flip_pairs_all_samples(input_path):
    """
    Like moral_sycophancy_judge.iter_flip_pairs, but returns ALL samples for each side
    instead of just sample_idx=0 -- {row_id: {"original_post": [rec_s0, rec_s1, ...],
    "flipped_story": [rec_s0, rec_s1, ...]}}, only for row_ids where both sides have at
    least one sample. Samples within a side are sorted by sample_idx for a
    deterministic order.
    """
    by_row = defaultdict(lambda: defaultdict(list))
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_row[rec["row_id"]][rec["prompt_col"]].append(rec)
    pairs = {}
    for row_id, sides in by_row.items():
        if "original_post" in sides and "flipped_story" in sides:
            pairs[row_id] = {
                "original_post": sorted(sides["original_post"], key=lambda r: r["sample_idx"]),
                "flipped_story": sorted(sides["flipped_story"], key=lambda r: r["sample_idx"]),
            }
    return pairs


def _average_blocks(arr, sizes, examples_axis):
    """arr has an 'examples' axis (length sum(sizes)) at position `examples_axis` --
    average each consecutive block of that axis down to one vector, preserving every
    other axis."""
    arr = np.moveaxis(arr, examples_axis, -2)
    blocks = []
    start = 0
    for size in sizes:
        blocks.append(arr[..., start : start + size, :].mean(axis=-2))
        start += size
    stacked = np.stack(blocks, axis=-2)
    return np.moveaxis(stacked, -2, examples_axis)


def labeled_text(tokenizer, rec: dict, templated: bool) -> str:
    if templated:
        return rec["prompt"] + rec["response"]
    from probing.evaluations.judge.moral_sycophancy_judge import build_labeled_text

    return build_labeled_text(tokenizer, rec)


def build_row_id_pooled_activations(model, tokenizer, model_config, label_result, data_path, args):
    """Dedupe the 2-per-row_id labeled records to one label per row_id, flatten
    every sample from both flip sides into one text list, extract once, then
    average each row_id's block of samples to a single vector per row_id.

    Returns (residual (n_layers, n_row_ids, hidden_dim), labels, row_ids)."""
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
        flat_texts.extend(labeled_text(tokenizer, rec, args.prompts_templated) for rec in side_recs)
        block_sizes.append(len(side_recs))

    labels = np.array([row_id_labels[rid] for rid in row_ids], dtype=int)

    print(f"Extracting residual activations for {len(flat_texts)} raw generations ({len(row_ids)} row_ids)...")
    flat = core.collect_residual_only(model, tokenizer, flat_texts, model_config, args.max_length,
                                      core.pooling_for(args.average))
    residual = _average_blocks(flat, block_sizes, examples_axis=1)
    avg_samples = len(flat_texts) / len(row_ids) if row_ids else 0.0
    print(f"Row_id-pooled residual for {len(row_ids)} row_ids (avg {avg_samples:.1f} samples/row_id): {residual.shape}")
    return residual, labels, row_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-path", type=Path, default=None,
                        help="AITA-NTA-FLIP generations (row_id/prompt_col/sample_idx/prompt/response). "
                             "Default: moral_sycophancy_judge.DEFAULT_INPUT_PATH.")
    parser.add_argument("--n-examples", type=int, default=100, help="Pairs to judge and train on.")
    parser.add_argument("--prompts-templated", action="store_true", help="prompt fields are already chat-templated.")
    core.add_sweep_args(parser, default_model="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    from probing.evaluations.judge.moral_sycophancy_judge import DEFAULT_INPUT_PATH
    from utils.model import cleanup as cleanup_model

    input_path = args.input_path or DEFAULT_INPUT_PATH
    print(f"Loading {args.model}...")
    model, tokenizer, model_config = core.load_model_and_config(args.model)
    layers = core.select_layers(model_config["n_layers"], args.layers)

    print(f"\nBuilding AITA moral-sycophancy labels (n_examples={args.n_examples})...")
    label_info = build_labels(tokenizer, input_path, args.n_examples)
    print(f"Class balance: {label_info['n_pos']} pos / {label_info['n_neg']} neg out of {label_info['n_judged']} judged")

    residual, labels, row_ids = build_row_id_pooled_activations(model, tokenizer, model_config, label_info, input_path, args)
    cleanup_model(model, tokenizer)

    core.cache_and_train(
        args, residual, labels, row_ids, layers,
        meta={"recipe": "aita_nta_flip_row_id_pooled", "input_path": str(input_path),
              "n_examples": args.n_examples, "n_row_ids": len(row_ids),
              "label_balance": {k: label_info[k] for k in ("n_pos", "n_neg", "n_judged")}},
    )


if __name__ == "__main__":
    main()
