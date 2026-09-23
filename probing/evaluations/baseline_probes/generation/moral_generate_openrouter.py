#!/usr/bin/env python3
"""
OpenRouter port of moral_generate.py: generate model responses for one of the
ELEPHANT moral-sycophancy source datasets (AITA-NTA-FLIP, AITA-NTA-OG) via
OpenRouter instead of local/Colab GPU. Same row selection (seeded shuffle over
row_ids, --n cap keeps complete flip pairs) and output schema as
moral_generate.py, so run_moral_sycophancy_judge_aita.py reads the checkpoint
unchanged:
    {dataset, row_id, prompt_col, prompt, response, sample_idx, model}

AITA-NTA-FLIP has two prompt columns per row (original_post, flipped_story) --
both sides are generated as separate records sharing row_id, exactly as the
flip judge's iter_flip_pairs() expects.

"prompt" is the raw source text (not a chat-templated string with special
tokens -- OpenRouter's provider applies its own template server-side, and the
judge scripts only need human-readable text).

Usage:
    python -m probing.evaluations.baseline_probes.generation.moral_generate_openrouter \
        --dataset aita_nta_og --n 1591 --out <path>
    python -m probing.evaluations.baseline_probes.generation.moral_generate_openrouter \
        --dataset aita_nta_flip --n 1591 --out <path>
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.evaluations.baseline_probes.generation.common import generate_via_openrouter_with_finish_reasons  # noqa: E402
from utils.datasets import iter_prompts  # noqa: E402

DATASET_FILES = {"aita_nta_flip": "AITA-NTA-FLIP.csv", "aita_nta_og": "AITA-NTA-OG.csv"}


def _save_truncated(out_path, records):
    if not records:
        return
    with open(out_path.parent / "checkpoint.truncated.jsonl", "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASET_FILES), default="aita_nta_flip")
    parser.add_argument("--model", type=str, default="meta-llama/llama-3.1-8b-instruct",
                         help="OpenRouter model slug (lowercase).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=32, help="Concurrent OpenRouter requests")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true", help="Off by default: matches original's greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n", type=int, default=None,
                         help="Cap on number of ROWS (not generations) -- for aita_nta_flip this keeps both "
                              "sides of each of the first N rows, so the checkpoint has 2N records. "
                              "Default: all rows in the dataset.")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_file = DATASET_FILES[args.dataset]
    print(f"Loading {dataset_file}...")
    all_rows = list(iter_prompts(dataset_file))

    # Shuffle whole row_ids together (not individual prompt_col records) so
    # both sides of a flip pair stay adjacent and a --n row cap keeps
    # complete pairs, not one orphaned side.
    row_ids = sorted({r["row_id"] for r in all_rows}, key=str)
    random.Random(args.seed).shuffle(row_ids)
    if args.n is not None:
        row_ids = row_ids[: args.n]
    wanted_ids = set(row_ids)
    order = {rid: i for i, rid in enumerate(row_ids)}
    rows = sorted((r for r in all_rows if r["row_id"] in wanted_ids), key=lambda r: (order[r["row_id"]], r["prompt_col"]))
    total = len(rows)
    print(f"{len(row_ids)} rows -> {total} generations to process (seed={args.seed}).")

    already_done = 0
    if out_path.exists():
        with open(out_path) as f:
            already_done = sum(1 for _ in f)
        print(f"Resuming: {already_done}/{total} generations already checkpointed at {out_path}.")

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
        print(f"Done. {n_written}/{total} generations written to {out_path} ({len(truncated_records)} cap-hit set aside).")

    n_lines = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    n_truncated_total = 0
    truncated_path = out_path.parent / "checkpoint.truncated.jsonl"
    if truncated_path.exists():
        n_truncated_total = sum(1 for _ in open(truncated_path))
    summary = {
        "model": args.model, "dataset": args.dataset, "seed": args.seed, "backend": "openrouter",
        "n_rows": len(row_ids), "n_total_generations": total,
        "n_generated": n_lines, "n_cap_hit_set_aside": n_truncated_total,
    }
    summary_path = out_path.parent / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
