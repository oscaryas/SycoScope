#!/usr/bin/env python3
"""
Generate model responses for one of the ELEPHANT social-sycophancy source
datasets (OEQ, SS, AITA-YTA), checkpointed like sypr_praise_full_generate.py.
Generation only -- judging is a separate step via the existing
run_social_sycophancy_judge_oeq.py runner (all three metrics: validation,
indirectness, framing), which already has --resume and retry/backoff.

Output schema matches SAE/pipeline/generations.py, so the existing judge
runner's iter_dataset_records() reads it unchanged:
    {dataset, row_id, prompt_col, prompt, response, sample_idx, model}

Usage:
    python social_generate.py --dataset oeq --model Qwen/Qwen3-8B \
        --out results/generations/multimodel/Qwen__Qwen3-8B/oeq/checkpoint.jsonl
    python run_social_sycophancy_judge_oeq.py \
        --input-path results/generations/multimodel/Qwen__Qwen3-8B/oeq/checkpoint.jsonl \
        --output-path results/generations/multimodel/Qwen__Qwen3-8B/oeq/judged.jsonl

    # smoke test (small, fast, no GPU required):
    python social_generate.py --dataset ss --n 8 --out /tmp/smoke.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYCOPHANCY_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
for p in (SYCOPHANCY_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATASET_FILES = {"oeq": "OEQ.csv", "ss": "SS.csv", "aita_yta": "AITA-YTA.csv"}


def _save_truncated(out_path, records):
    """Append cap-hit (INCOMPLETE) generations to a sidecar next to the checkpoint
    so they are kept rather than silently discarded; they are excluded from the
    labeled checkpoint (and therefore from judging)."""
    if not records:
        return
    with open(out_path.parent / "checkpoint.truncated.jsonl", "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASET_FILES), default="oeq")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--seed", type=int, default=0, help="Shuffles row order (also the resume order) -- keep fixed across resumes.")
    parser.add_argument("--system-prompt", type=str, default=None,
                         help="Optional system prompt for every generation turn -- e.g. 'detailed thinking on' "
                              "to enable Nemotron's reasoning mode (its thinking is OFF without it).")
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", type=str, default=str(SYCOPHANCY_DIR / "results" / "generations" / "social" / "checkpoint.jsonl"))
    parser.add_argument("--n", type=int, default=None, help="Cap on number of rows, for smoke-testing. Default: all rows in the dataset.")
    args = parser.parse_args()
    if "nemotron" in args.model.lower() and not args.system_prompt:
        print("WARNING: Nemotron models run with thinking OFF unless --system-prompt "
              "'detailed thinking on' is passed.")

    import torch
    from utils.model import load_model_and_tokenizer, cleanup as cleanup_model
    from sycophancy_model_registry import get_model_config
    from sycophancy_steering import ActivationSteerer
    from utils.inference import build_chat_prompt
    from utils.datasets import iter_prompts

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
        if torch.cuda.is_available():
            device_map = "auto"
        elif torch.backends.mps.is_available():
            device_map = {"": "mps"}
        else:
            device_map = {"": "cpu"}

        print(f"Loading {args.model} (device_map={device_map})...")
        model, tokenizer = load_model_and_tokenizer(args.model, device_map=device_map)
        model_config = get_model_config(args.model)
        steerer = ActivationSteerer(model, tokenizer, model_config)

        n_written = already_done
        with open(out_path, "a") as f:
            for start in range(0, len(remaining_rows), args.generation_batch_size):
                chunk = remaining_rows[start : start + args.generation_batch_size]
                prompts = [build_chat_prompt(tokenizer, row["text"], system_prompt=args.system_prompt) for row in chunk]
                responses = steerer.generate_batch(prompts, max_new_tokens=args.max_new_tokens, batch_size=args.generation_batch_size)
                truncated = steerer.last_truncated

                truncated_records = [
                    {"dataset": row["dataset"], "row_id": row["row_id"], "prompt_col": row["prompt_col"],
                     "prompt": prompt, "response": response, "max_new_tokens": args.max_new_tokens, "model": args.model}
                    for row, prompt, response, trunc in zip(chunk, prompts, responses, truncated) if trunc
                ]
                _save_truncated(out_path, truncated_records)

                for row, prompt, response, trunc in zip(chunk, prompts, responses, truncated):
                    if trunc:
                        continue
                    record = {
                        "dataset": row["dataset"], "row_id": row["row_id"], "prompt_col": row["prompt_col"],
                        "prompt": prompt, "response": response, "sample_idx": 0, "model": args.model,
                    }
                    f.write(json.dumps(record) + "\n")
                    n_written += 1
                f.flush()
                n_truncated_this_batch = len(truncated_records)
                print(f"[{n_written}/{total}] checkpointed ({len(chunk) - n_truncated_this_batch} generated, "
                      f"{n_truncated_this_batch} cap-hit set aside this batch)")

        steerer.cleanup()
        cleanup_model(model, tokenizer)
        print(f"Done. {n_written}/{total} rows written to {out_path}.")

    n_lines = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    n_truncated_total = 0
    truncated_path = out_path.parent / "checkpoint.truncated.jsonl"
    if truncated_path.exists():
        n_truncated_total = sum(1 for _ in open(truncated_path))
    summary = {
        "model": args.model, "dataset": args.dataset, "seed": args.seed,
        "n_total_rows": total, "n_generated": n_lines, "n_cap_hit_set_aside": n_truncated_total,
    }
    summary_path = out_path.parent / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
