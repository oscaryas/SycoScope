#!/usr/bin/env python3
"""
Standalone CLI: how many SyPR responses did a given model generate that get
judged as sycophantic praise? Reuses sypr_data.generate_and_label_sypr (the
same generation+judging path sypr_probe_pipeline.py trains probes on) without
the probe-training step -- just the counts.

"Sycophantic praise" here follows sypr_data.py's label: praised==1 (the judge
says the response praises/validates the utterance) AND is_poor_quality(row)
(SyPR's own ground truth says the utterance was actually wrong/low-quality) --
i.e. false, unwarranted praise, not praise in general. Also reports the raw
praise rate (regardless of quality) since "how many responses were sycophantic
praise" is ambiguous between the two.

Usage:
    python sypr_praise_count.py --model meta-llama/Llama-3.1-8B-Instruct --n 120
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for p in (HERE, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n", type=int, default=120, help="Number of SyPR rows to sample+generate+judge.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    import torch
    from utils.model import load_model_and_tokenizer, cleanup as cleanup_model
    from sycophancy_model_registry import get_model_config
    from sypr_data import generate_and_label_sypr

    # device_map="auto" (utils.model's default) is tuned for CUDA's balanced
    # sharding and doesn't budget MPS memory correctly -- on this Mac it silently
    # decided to offload layers to disk (confirmed: process RSS stayed ~85MB for
    # an 8B model, i.e. it was streaming weights from disk on every forward pass
    # instead of holding them in unified memory), which makes generation
    # prohibitively slow. Forcing a single-device map skips that balancer.
    if torch.cuda.is_available():
        device_map = "auto"
    elif torch.backends.mps.is_available():
        device_map = {"": "mps"}
    else:
        device_map = {"": "cpu"}

    print(f"Loading {args.model} (device_map={device_map})...")
    model, tokenizer = load_model_and_tokenizer(args.model, device_map=device_map)
    model_config = get_model_config(args.model)

    print(f"Sampling+generating+judging {args.n} SyPR rows...")
    result = generate_and_label_sypr(
        model, tokenizer, model_config,
        n_train=args.n, seed=args.seed,
        generation_batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    cleanup_model(model, tokenizer)

    n_judged = result["n_judged"]
    n_pos = result["n_pos"]  # praised AND poor-quality utterance (sycophantic praise)
    n_raw_praise = sum(r["praised"] == 1 for r in result["records"])

    print()
    print(f"Model: {args.model}")
    print(f"Sampled: {args.n}  Judged: {n_judged}  Skipped (judge parse failure): {result['n_skipped']}")
    print(f"Raw praise (praised==1, any utterance quality): {n_raw_praise}/{n_judged} ({n_raw_praise/n_judged:.1%})")
    print(f"Sycophantic praise (praised==1 AND utterance was poor-quality): {n_pos}/{n_judged} ({n_pos/n_judged:.1%})")
    print(f"  Praise rate on poor-quality utterances: {result['praise_rate_on_poor']:.1%}")
    print(f"  Praise rate on good-quality utterances: {result['praise_rate_on_good']:.1%}")


if __name__ == "__main__":
    main()
