#!/usr/bin/env python3
"""
OpenRouter port of social_generate.py: generate model responses for one of the
ELEPHANT social-sycophancy source datasets (OEQ, SS, AITA-YTA) via OpenRouter
instead of local/Colab GPU. Same row selection (seeded shuffle, --n cap) and
output schema as social_generate.py, so run_social_sycophancy_judge_oeq.py and
run_moral_sycophancy_judge_aita.py (--mode single) read the checkpoint unchanged:
    {dataset, row_id, prompt_col, prompt, response, sample_idx, model}

"prompt" is the raw source text (not a chat-templated string with special
tokens -- OpenRouter's provider applies its own template server-side, and the
judge scripts only need human-readable text).

Usage:
    python social_generate_openrouter.py --dataset aita_yta --n 2000 \
        --out results/generations/openrouter_llama31/aita_yta/checkpoint.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYCOPHANCY_DIR = HERE.parent
REPO_ROOT = SYCOPHANCY_DIR.parents[2]
for p in (SYCOPHANCY_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pipeline_scripts.generations.common import generate_via_openrouter_with_finish_reasons  # noqa: E402
from utils.datasets import iter_prompts  # noqa: E402

DATASET_FILES = {"oeq": "OEQ.csv", "ss": "SS.csv", "aita_yta": "AITA-YTA.csv"}


def _save_truncated(out_path, records):
    if not records:
        return
    with open(out_path.parent / "checkpoint.truncated.jsonl", "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASET_FILES), default="aita_yta")
    parser.add_argument("--model", type=str, default="meta-llama/llama-3.1-8b-instruct",
                         help="OpenRouter model slug (lowercase).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=32, help="Concurrent OpenRouter requests")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true", help="Off by default: matches original's greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--out", type=str, default=str(SYCOPHANCY_DIR / "results" / "generations" / "openrouter_llama31" / "social" / "checkpoint.jsonl"))
    parser.add_argument("--n", type=int, default=None, help="Cap on number of rows. Default: all rows in the dataset.")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_file = DATASET_FILES[args.dataset]
    print(f"Loading {dataset_file}...")
    rows = list(iter_prompts(dataset_file))
    random.Random(args.seed).shuffle(rows)
    if args.n is not None:
        rows = rows[: args.n]
    total = len(rows)
    print(f"{total} rows to process (seed={args.seed}).")

    already_done = 0
    if out_path.exists():
        with open(out_path) as f:
            already_done = sum(1 for _ in f)
        print(f"Resuming: {already_done}/{total} rows already checkpointed at {out_path}.")

    remaining_rows = rows[already_done:]
    if not remaining_rows:
        print("Nothing left to do -- checkpoint already covers all requested rows.")
    else:
        messages_list = [[{"role": "user", "content": row["text"]}] for row in remaining_rows]
        print(f"Dispatching {len(messages_list)} OpenRouter requests (max_workers={args.max_workers})...")
        generations = generate_via_openrouter_with_finish_reasons(messages_list, args)

        n_written = already_done
        truncated_records = []
        with open(out_path, "a") as f:
            for row, gen in zip(remaining_rows, generations):
                if gen["finish_reason"] == "length":
                    truncated_records.append({
                        "dataset": row["dataset"], "row_id": row["row_id"], "prompt_col": row["prompt_col"],
                        "prompt": row["text"], "response": gen["content"],
                        "max_new_tokens": args.max_new_tokens, "model": args.model,
                    })
                    continue
                record = {
                    "dataset": row["dataset"], "row_id": row["row_id"], "prompt_col": row["prompt_col"],
                    "prompt": row["text"], "response": gen["content"], "sample_idx": 0, "model": args.model,
                    "finish_reason": gen["finish_reason"],
                }
                f.write(json.dumps(record) + "\n")
                n_written += 1
        _save_truncated(out_path, truncated_records)
        print(f"Done. {n_written}/{total} rows written to {out_path} ({len(truncated_records)} cap-hit set aside).")

    n_lines = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    n_truncated_total = 0
    truncated_path = out_path.parent / "checkpoint.truncated.jsonl"
    if truncated_path.exists():
        n_truncated_total = sum(1 for _ in open(truncated_path))
    summary = {
        "model": args.model, "dataset": args.dataset, "seed": args.seed, "backend": "openrouter",
        "n_total_rows": total, "n_generated": n_lines, "n_cap_hit_set_aside": n_truncated_total,
    }
    summary_path = out_path.parent / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
