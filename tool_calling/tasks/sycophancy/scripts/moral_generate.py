#!/usr/bin/env python3
"""
Generate model responses for one of the ELEPHANT moral-sycophancy source
datasets (AITA-NTA-FLIP, AITA-NTA-OG), checkpointed like
sypr_praise_full_generate.py. Generation only -- judging is a separate step
via the existing run_moral_sycophancy_judge_aita.py runner (--mode flip for
AITA-NTA-FLIP, --mode single for AITA-NTA-OG), which already has --resume
and retry/backoff.

AITA-NTA-FLIP has two prompt columns per row (original_post, flipped_story)
-- both sides of the same conflict are generated (one row in the checkpoint
per column) so the flip judge can pair them back up by row_id.

Output schema matches SAE/pipeline/generations.py, so the existing judge
runner's iter_flip_pairs()/iter_dataset_records() read it unchanged:
    {dataset, row_id, prompt_col, prompt, response, sample_idx, model}

Usage:
    python moral_generate.py --dataset aita_nta_flip --model Qwen/Qwen3-8B \
        --out results/generations/multimodel/Qwen__Qwen3-8B/aita_nta_flip/checkpoint.jsonl
    python run_moral_sycophancy_judge_aita.py --mode flip \
        --input-path results/generations/multimodel/Qwen__Qwen3-8B/aita_nta_flip/checkpoint.jsonl \
        --output-path results/generations/multimodel/Qwen__Qwen3-8B/aita_nta_flip/judged.jsonl

    # smoke test (small, fast, no GPU required):
    python moral_generate.py --dataset aita_nta_flip --n 8 --out /tmp/smoke.jsonl
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

DATASET_FILES = {"aita_nta_flip": "AITA-NTA-FLIP.csv", "aita_nta_og": "AITA-NTA-OG.csv"}


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
    parser.add_argument("--dataset", choices=sorted(DATASET_FILES), default="aita_nta_flip")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--seed", type=int, default=0, help="Shuffles row order (also the resume order) -- keep fixed across resumes.")
    parser.add_argument("--system-prompt", type=str, default=None,
                         help="Optional system prompt for every generation turn -- e.g. 'detailed thinking on' "
                              "to enable Nemotron's reasoning mode (its thinking is OFF without it).")
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", type=str, default=str(SYCOPHANCY_DIR / "results" / "generations" / "moral" / "checkpoint.jsonl"))
    parser.add_argument("--n", type=int, default=None,
                         help="Cap on number of ROWS (not generations) -- for aita_nta_flip this keeps both "
                              "sides of each of the first N rows, so the checkpoint has 2N records. "
                              "Default: all rows in the dataset.")
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
        print(f"Done. {n_written}/{total} generations written to {out_path}.")

    n_lines = sum(1 for _ in open(out_path)) if out_path.exists() else 0
    n_truncated_total = 0
    truncated_path = out_path.parent / "checkpoint.truncated.jsonl"
    if truncated_path.exists():
        n_truncated_total = sum(1 for _ in open(truncated_path))
    summary = {
        "model": args.model, "dataset": args.dataset, "seed": args.seed,
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
