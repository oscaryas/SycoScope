#!/usr/bin/env python3
"""
Full-dataset SyPR generation + judging: every one of the ~10,800 label-eligible
rows (see sypr_data.all_eligible_indices), not a stratified sample. Same
generate -> judge -> label path as sypr_data.generate_and_label_sypr /
sypr_praise_count.py, but checkpointed to a JSONL file so a crash mid-run only
loses the batch in flight, not hours of prior progress -- rerunning the same
command resumes automatically from wherever the checkpoint left off.

Usage:
    python sypr_praise_full_generate.py --model meta-llama/Llama-3.1-8B-Instruct \
        --out results/sypr_praise_llama31_full/checkpoint.jsonl

    # smoke test (small, fast, no GPU required):
    python sypr_praise_full_generate.py --n 8 --out /tmp/smoke.jsonl
"""
import argparse
import json
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
    parser.add_argument("--seed", type=int, default=0, help="Shuffles row order (also the resume order) -- keep fixed across resumes.")
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--judge-max-workers", type=int, default=16)
    parser.add_argument("--out", type=str, default=str(HERE / "results" / "sypr_praise_llama31_full" / "checkpoint.jsonl"))
    parser.add_argument("--n", type=int, default=None, help="Cap on number of rows, for smoke-testing. Default: all label-eligible rows.")
    args = parser.parse_args()

    import torch
    from utils.model import load_model_and_tokenizer, cleanup as cleanup_model
    from sycophancy_model_registry import get_model_config
    from sycophancy_steering import ActivationSteerer
    from utils.inference import build_chat_prompt_multiturn
    from sycophantic_praise_judge import judge_praise_batch
    from sypr_data import load_sypr_dataset, all_eligible_indices, is_poor_quality, build_chat_messages, _row_from_index

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading SyPR dataset...")
    dataset = load_sypr_dataset()

    import random

    indices = all_eligible_indices(dataset)
    random.Random(args.seed).shuffle(indices)
    if args.n is not None:
        indices = indices[: args.n]
    total = len(indices)
    print(f"{total} label-eligible rows to process (seed={args.seed}).")

    already_done = 0
    if out_path.exists():
        with open(out_path) as f:
            already_done = sum(1 for _ in f)
        print(f"Resuming: {already_done}/{total} rows already checkpointed at {out_path}.")

    remaining_indices = indices[already_done:]
    if not remaining_indices:
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
            for start in range(0, len(remaining_indices), args.generation_batch_size):
                chunk_indices = remaining_indices[start : start + args.generation_batch_size]
                rows = [_row_from_index(dataset, i) for i in chunk_indices]
                prompts = [build_chat_prompt_multiturn(tokenizer, build_chat_messages(r)) for r in rows]
                responses = steerer.generate_batch(prompts, max_new_tokens=args.max_new_tokens)
                for row, prompt, response in zip(rows, prompts, responses):
                    row["prompt"] = prompt
                    row["response"] = response

                verdicts = judge_praise_batch(rows, max_workers=args.judge_max_workers)
                for row, verdict in zip(rows, verdicts):
                    row["praised"] = verdict

                n_skipped_this_batch = 0
                for row in rows:
                    if row["praised"] is None:
                        n_skipped_this_batch += 1
                        continue
                    record = {
                        "text": row["prompt"] + row["response"],
                        "label": 1 if (row["praised"] == 1 and is_poor_quality(row)) else 0,
                        "domain": row["domain"],
                        "utterance_text": row["utterance_text"],
                        "response": row["response"],
                        "prompt": row["prompt"],
                        "is_poor_quality": is_poor_quality(row),
                        "praised": row["praised"],
                    }
                    f.write(json.dumps(record) + "\n")
                    n_written += 1
                f.flush()
                print(f"[{n_written}/{total}] checkpointed ({len(rows)} generated, {n_skipped_this_batch} judge-skipped this batch)")

        steerer.cleanup()
        cleanup_model(model, tokenizer)
        print(f"Done. {n_written}/{total} rows written to {out_path}.")

    records = [json.loads(line) for line in open(out_path)]
    n_judged = len(records)
    n_pos = sum(r["label"] == 1 for r in records)
    n_raw_praise = sum(r["praised"] == 1 for r in records)
    n_poor = sum(r["is_poor_quality"] for r in records)
    n_good = n_judged - n_poor
    praise_rate_on_poor = sum(r["praised"] == 1 for r in records if r["is_poor_quality"]) / n_poor if n_poor else 0.0
    praise_rate_on_good = sum(r["praised"] == 1 for r in records if not r["is_poor_quality"]) / n_good if n_good else 0.0

    summary = {
        "model": args.model,
        "seed": args.seed,
        "n_total_eligible": total,
        "n_judged": n_judged,
        "n_skipped": total - n_judged,
        "n_raw_praise": n_raw_praise,
        "n_sycophantic_praise": n_pos,
        "praise_rate_on_poor": praise_rate_on_poor,
        "praise_rate_on_good": praise_rate_on_good,
    }
    summary_path = out_path.parent / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
