#!/usr/bin/env python3
"""Cache + train a moral-sycophancy residual probe from a judged AITA-NTA-FLIP
file (one row per row_id with original_post_* and flipped_story_* prompt,
response and verdict fields).

Two forward passes per pair -- original post and flipped story, each with its
own --max-length budget -- averaged per layer. A single pass over the
concatenated pair let the flipped-story half fall past truncation for ~39% of
pairs.

Labels: positive = both verdicts NTA; negative = both YTA, or either side
OTHER/unclear; Mixed (one NTA, one YTA) pairs are excluded. All positives vs
a --seed sample of the same number of negatives (random.Random(seed)).
CV groups are row_ids.

Usage:
    python -m probing.analyze_probes.cache_train_moral \\
        --judged probing/data/baseline/<model>/aita_nta_flip/judged.jsonl \\
        --model <model> --layers 8 16 24 --output weights.pkl
"""
import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from probing.analyze_probes import probes_core as core  # noqa: E402


def select_moral_pairs(judged_path: Path, seed: int) -> dict:
    """
    Builds the positive/negative pools directly from verdicts:
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


def side_texts(tokenizer, recs: list[dict], side: str, templated: bool) -> list[str]:
    """prompt + response for one side. Raw prompts are wrapped with the chat
    template (no system prompt); --prompts-templated files (e.g. gemma's
    judged.jsonl) already hold the rendered chat prefix."""
    from utils.inference import build_chat_prompt

    if templated:
        return [r[f"{side}_prompt"] + r[f"{side}_response"] for r in recs]
    return [build_chat_prompt(tokenizer, r[f"{side}_prompt"], system_prompt=None) + r[f"{side}_response"] for r in recs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judged", required=True, type=Path)
    parser.add_argument("--prompts-templated", action="store_true",
                        help="*_prompt fields are already chat-templated (do not re-wrap).")
    core.add_sweep_args(parser, default_model="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    selected = select_moral_pairs(args.judged, args.seed)
    row_ids = sorted(selected.keys())
    recs = [selected[rid][0] for rid in row_ids]
    labels = np.array([selected[rid][1] for rid in row_ids], dtype=int)

    from utils.model import cleanup as cleanup_model

    print(f"\nLoading {args.model}...")
    model, tokenizer, model_config = core.load_model_and_config(args.model)
    layers = core.select_layers(model_config["n_layers"], args.layers)
    pooling = core.pooling_for(args.average)

    og_texts = side_texts(tokenizer, recs, "original_post", args.prompts_templated)
    flip_texts = side_texts(tokenizer, recs, "flipped_story", args.prompts_templated)
    print(f"\n[run 1/2] Extracting residual activations for {len(og_texts)} original-post responses...")
    og = core.collect_residual_only(model, tokenizer, og_texts, model_config, args.max_length, pooling)
    print(f"\n[run 2/2] Extracting residual activations for {len(flip_texts)} flipped-story responses...")
    flip = core.collect_residual_only(model, tokenizer, flip_texts, model_config, args.max_length, pooling)
    cleanup_model(model, tokenizer)

    core.cache_and_train(
        args, (og + flip) / 2.0, labels, row_ids, layers,
        meta={"recipe": "moral_two_pass_avg", "judged": str(args.judged),
              "positive_definition": "Both NTA",
              "negative_definition": "Both YTA, or either side OTHER/unclear; Mixed excluded"},
    )


if __name__ == "__main__":
    main()
