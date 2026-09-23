#!/usr/bin/env python3
"""
OpenRouter variant of generate_response.py: same paired-prompt design, same
per-cell checkpointed jsonl schema, but generation goes through OpenRouter's
chat completions API (concurrent HTTP requests) instead of a local HF model
forward pass. Used for meta-llama/Llama-3.1-8B-Instruct, which is not the
pipeline's default Llama-3-8B-Instruct (that model isn't on OpenRouter at
all) -- so output from this script lives under its own --run-name and is not
directly comparable to results under the "main" run without its own separate
extraction/training pass.

Reuses build_work_list/load_done/classify_degenerate/group_key/report_cells/
print_report from generate_response.py verbatim -- only the generation loop
itself (local model vs. OpenRouter) differs.

Usage:
    python generate_response_openrouter.py --run-name llama31_5k \\
        --user-prompts ../data/perez_user_prompts_5k.jsonl \\
        --model meta-llama/llama-3.1-8b-instruct
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.utils import common  # noqa: E402
from probing.evaluations.prompt_probes.generation.generate_response import (  # noqa: E402
    build_work_list,
    classify_degenerate,
    group_key,
    load_done,
    print_report,
    report_cells,
)
from utils.inference import iter_batches  # noqa: E402

from probing.evaluations.baseline_probes.generation.common import generate_via_openrouter_with_finish_reasons  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model", type=str, default="meta-llama/llama-3.1-8b-instruct",
                        help="OpenRouter model slug.")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HF repo id for the tokenizer used only to count response tokens (CPU, no weights).")
    parser.add_argument("--user-prompts", type=Path, required=True,
                        help="jsonl of user prompts to use -- NOT common.USER_PROMPTS_PATH, to avoid "
                        "touching the shared default file other runs read from.")
    common.add_cells_arg(parser)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-workers", type=int, default=32, help="Concurrent OpenRouter requests per chunk.")
    parser.add_argument("--chunk-size", type=int, default=250,
                        help="Items per generate_via_openrouter_with_finish_reasons call. Results are written "
                        "and fsync'd after each chunk, bounding how much work a crash mid-group can lose "
                        "(unlike waiting for an entire group of up to thousands of items to finish).")
    parser.add_argument("--min-response-chars", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    # generate_via_openrouter_with_finish_reasons reads args.do_sample directly.
    args.do_sample = True

    run_dir = common.resolve_run_dir(args.run_name)
    (run_dir / "generations").mkdir(parents=True, exist_ok=True)

    pairs = common.load_prompt_pairs()
    prompts = common.read_jsonl(args.user_prompts)
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    common.write_jsonl(run_dir / "user_prompts.jsonl", prompts)

    work = build_work_list(pairs, prompts, args.cells)
    groups: dict[tuple, list[dict]] = {}
    for item in work:
        groups.setdefault(group_key(item), []).append(item)
    print(f"{len(work)} generations across {len(groups)} (cell, polarity) groups, {len(prompts)} user prompts")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    todo = {}
    for key, items in groups.items():
        slug = key[0]
        done = load_done(run_dir / "generations" / f"{slug}.jsonl")
        pending = [i for i in items if i["example_id"] not in done]
        if pending:
            todo[key] = pending
    n_pending = sum(len(v) for v in todo.values())
    print(f"{n_pending} pending after resume ({len(work) - n_pending} already on disk)")

    n_done = 0
    for key, pending in todo.items():
        slug, polarity = key
        system_prompt = pending[0]["system_prompt"]
        out_path = run_dir / "generations" / f"{slug}.jsonl"
        print(f"\n[{slug} / {polarity}] {len(pending)} to generate via OpenRouter "
              f"(max_workers={args.max_workers}, chunk_size={args.chunk_size})", flush=True)

        with open(out_path, "a", encoding="utf-8") as f:
            for chunk in iter_batches(pending, args.chunk_size):
                messages_list = []
                for item in chunk:
                    msgs = []
                    if system_prompt:
                        msgs.append({"role": "system", "content": system_prompt})
                    msgs.append({"role": "user", "content": item["user_prompt"]})
                    messages_list.append(msgs)

                results = generate_via_openrouter_with_finish_reasons(messages_list, args)

                for item, result in zip(chunk, results):
                    response = result["content"]
                    n_tok = len(tokenizer(response, add_special_tokens=False)["input_ids"]) if response else 0
                    reason = classify_degenerate(response, n_tok, args.max_new_tokens, args.min_response_chars)
                    if result["finish_reason"] == "length" and reason != "empty":
                        reason = reason or "truncated"
                    if result["finish_reason"] == "request_failed":
                        reason = "request_failed"
                    record = {
                        **{k: v for k, v in item.items()},
                        "response": response,
                        "n_response_chars": len(response),
                        "n_response_tokens": n_tok,
                        "degenerate": reason,
                        "model": args.model,
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "seed": args.seed,
                        "finish_reason": result["finish_reason"],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                n_done += len(chunk)
                print(f"  {n_done}/{n_pending} total", flush=True)

    touched = sorted({k[0] for k in groups})
    report = report_cells(run_dir, touched)
    print_report(report)
    (run_dir / "checks").mkdir(exist_ok=True)
    (run_dir / "checks" / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    common.write_run_info(
        run_dir,
        "generate_response_openrouter",
        args,
        {"n_prompts": len(prompts), "n_generations_target": len(work), "cells": touched},
    )
    print(f"\nDone. Records under {run_dir / 'generations'}")


if __name__ == "__main__":
    main()
