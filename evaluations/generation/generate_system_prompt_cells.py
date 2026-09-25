#!/usr/bin/env python3
"""
Generate responses on prompts from Perez et al. (2022) under contrastive system prompts.

Paired design: the same user prompts are used for both polarities of a cell and
for all cells. Nothing about prompt content can then separate the classes, and
each prompt yields one label-1 / label-0 pair -- which is why train_probes.py has
to split by prompt_id rather than by row. The system prompt is inherent here
(each cell's sycophantic / non_sycophantic text from
data/system_prompt/sycophancy_probe_prompt_pairs.json); pick cells with
--cells. The `neutral` pseudo-cell generates with no system prompt.

Backends (formerly generate_response.py and generate_response_openrouter.py):
  local       HF model, sampled (temperature/top_p) via utils.inference.generate_batch,
              --batch-size prompts per forward pass
  openrouter  concurrent chat-completions requests, sampled, --chunk-size items
              per checkpoint step; --tokenizer counts response tokens (CPU only)

Checkpointed: every chunk is appended and fsync'd immediately, so a crash costs
at most one batch, and re-running the same command resumes. Per-cell output
files (<run>/generations/<slug>.jsonl) mean one cell can be regenerated without
touching the others.

Usage:
    # smoke test, no GPU needed
    python -m evaluations.generation.generate_system_prompt_cells --run-name smoke \\
        --model meta-llama/Llama-3.2-1B-Instruct --cells general_baseline neutral \\
        --limit-prompts 4 --max-new-tokens 48 --batch-size 4

    # render prompts and count tokens without loading any weights
    python -m evaluations.generation.generate_system_prompt_cells --run-name smoke \\
        --cells general_baseline --limit-prompts 2 --dry-run

    # full local run
    python -m evaluations.generation.generate_system_prompt_cells --run-name main --batch-size 8

    # OpenRouter run
    python -m evaluations.generation.generate_system_prompt_cells --run-name llama31_5k \\
        --backend openrouter --model meta-llama/llama-3.1-8b-instruct \\
        --user-prompts data/system_prompt/perez_user_prompts_5k.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import common  # noqa: E402
from utils.inference import build_chat_prompt, iter_batches  # noqa: E402
from evaluations.generation.common import add_backend_args, generate_openrouter  # noqa: E402

DEFAULT_OPENROUTER_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
# --model default per backend: an HF repo id is not a valid OpenRouter slug
# (OpenRouter answers it with a non-retryable 400).
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"

# Diagnostics, not labels: they flag individual records and feed a per-cell rate.
REFUSAL_MARKERS = ("i can't", "i cannot", "i won't", "i'm not able to", "i am not able to")


def classify_degenerate(response: str, n_response_tokens: int, max_new_tokens: int, min_chars: int) -> str | None:
    """First matching degeneracy reason, or None. Flags only -- never drops."""
    text = response.strip()
    if not text:
        return "empty"
    if len(text) < min_chars:
        return "too_short"
    # Hit the token cap. Flagged but not a drop reason in train_probes: the
    # activations are still valid, and dropping these would condition on length.
    if n_response_tokens >= max_new_tokens - 2:
        return "truncated"
    words = text.lower().split()
    if len(words) >= 20:
        if len(set(words)) / len(words) < 0.35:
            return "repetitive"
        if len(words) >= 8:
            grams = Counter(tuple(words[i : i + 8]) for i in range(len(words) - 7))
            if grams.most_common(1)[0][1] > 4:
                return "repetitive"
    low = text.lower()
    if any(low.startswith(m) or f" {m}" in low[:200] for m in REFUSAL_MARKERS):
        # Interesting rather than broken: kept by default, flagged separately.
        return "refusal"
    return None


def build_work_list(pairs: list[dict], prompts: list[dict], slugs: list[str] | None) -> list[dict]:
    """One entry per (cell, polarity, prompt) still to be generated.

    The `neutral` pseudo-cell has no pair and no polarity: it is a single
    response per prompt with no system prompt, serving as the control set that
    probe scores are centred against (paper section 5.5).
    """
    work = []
    for pair in common.select_cells(pairs, slugs):
        for polarity in common.POLARITIES:
            for prompt in prompts:
                work.append(
                    {
                        "slug": pair["slug"],
                        "cell": pair["cell"],
                        "pair_type": pair["type"],
                        "pair_index": pair["pair_index"],
                        "polarity": polarity,
                        "label": common.POLARITY_LABEL[polarity],
                        "system_prompt": pair[polarity],
                        "prompt_id": prompt["prompt_id"],
                        "prompt_source": prompt["source"],
                        "user_prompt": prompt["user_prompt"],
                        "example_id": f"{pair['slug']}__{common.POLARITY_TAG[polarity]}__{prompt['prompt_id']}__s0",
                    }
                )
    if slugs is None or common.NEUTRAL_SLUG in slugs:
        for prompt in prompts:
            work.append(
                {
                    "slug": common.NEUTRAL_SLUG,
                    "cell": "Neutral (no system prompt)",
                    "pair_type": "neutral",
                    "pair_index": -1,
                    "polarity": "neutral",
                    "label": None,
                    "system_prompt": None,
                    "prompt_id": prompt["prompt_id"],
                    "prompt_source": prompt["source"],
                    "user_prompt": prompt["user_prompt"],
                    "example_id": f"{common.NEUTRAL_SLUG}__neu__{prompt['prompt_id']}__s0",
                }
            )
    return work


def load_done(path: Path) -> set[str]:
    """example_ids already generated. Rows from a failed OpenRouter request
    (degenerate == "request_failed", written by older runs) are not done: they
    are removed from the file here so the rerun retries them without leaving
    duplicate example_ids behind."""
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    ok = [r for r in rows if r.get("degenerate") != "request_failed"]
    if len(ok) != len(rows):
        print(f"{path.name}: dropping {len(rows) - len(ok)} request_failed rows so they are retried")
        common.write_jsonl(path, ok)
    return {r["example_id"] for r in ok}


def group_key(item: dict) -> tuple:
    """Batching unit. generate_batch takes one system_prompt per call, so the
    system prompt must be constant within a batch."""
    return (item["slug"], item["polarity"])


def report_cells(run_dir: Path, slugs: list[str]) -> dict:
    """Per-cell diagnostics over everything on disk for that cell."""
    report = {}
    for slug in slugs:
        path = run_dir / "generations" / f"{slug}.jsonl"
        if not path.exists():
            continue
        rows = common.read_jsonl(path)
        by_pol: dict[str, list[dict]] = {}
        for r in rows:
            by_pol.setdefault(r["polarity"], []).append(r)
        entry = {"n": len(rows), "degenerate_rate": round(sum(bool(r["degenerate"]) for r in rows) / len(rows), 4)}
        for pol, rs in sorted(by_pol.items()):
            texts = [r["response"].lower() for r in rs]
            entry[pol] = {
                "n": len(rs),
                "mean_chars": round(sum(len(t) for t in texts) / len(rs), 1),
                "refusal_rate": round(sum(any(m in t for m in REFUSAL_MARKERS) for t in texts) / len(rs), 4),
            }
        report[slug] = entry
    return report


def print_report(report: dict) -> None:
    print(f"\n{'cell':<26} {'n':>5} {'degen':>7} {'chars+':>8} {'chars-':>8}")
    print("-" * 76)
    for slug, e in report.items():
        pos, neg = e.get("sycophantic") or e.get("neutral") or {}, e.get("non_sycophantic") or {}
        print(
            f"{slug:<26} {e['n']:>5} {e['degenerate_rate']:>7.3f} "
            f"{pos.get('mean_chars', float('nan')):>8.1f} {neg.get('mean_chars', float('nan')):>8.1f} "
        )
    print(
        "\n'+' = sycophantic slot ('neutral' for the control cell), "
        "'-' = non_sycophantic. Matching columns => the prompt changed nothing."
    )


def _record(item: dict, response: str, n_tok: int, degenerate, args) -> dict:
    return {
        **item,
        "response": response,
        "n_response_chars": len(response),
        "n_response_tokens": n_tok,
        "degenerate": degenerate,
        "model": args.model,
        "backend": args.backend,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }


def _write_chunk(out_path: Path, records: list[dict]) -> None:
    with open(out_path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_local(todo: dict, run_dir: Path, n_pending: int, args) -> None:
    import torch

    from utils.inference import generate_batch
    from utils.model import cleanup as cleanup_model
    from evaluations.generation.common import load_local_model

    model, tokenizer = load_local_model(args.model)
    # Seeded once. Note that resuming mid-run shifts the RNG relative to an
    # uninterrupted run, so post-resume samples differ; recorded in run_info.
    torch.manual_seed(args.seed)
    n_done = 0
    for (slug, polarity), pending in todo.items():
        system_prompt = pending[0]["system_prompt"]
        print(f"\n[{slug} / {polarity}] {len(pending)} to generate")
        for chunk in iter_batches(pending, args.batch_size):
            responses = generate_batch(
                model, tokenizer, [c["user_prompt"] for c in chunk], system_prompt=system_prompt,
                max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature, top_p=args.top_p,
            )
            records = []
            for item, response in zip(chunk, responses):
                n_tok = len(tokenizer(response, add_special_tokens=False)["input_ids"])
                degenerate = classify_degenerate(response, n_tok, args.max_new_tokens, args.min_response_chars)
                records.append(_record(item, response, n_tok, degenerate, args))
            _write_chunk(run_dir / "generations" / f"{slug}.jsonl", records)
            n_done += len(chunk)
            print(f"  {n_done}/{n_pending}", flush=True)
    cleanup_model(model, tokenizer)


def run_openrouter(todo: dict, run_dir: Path, n_pending: int, args) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or DEFAULT_OPENROUTER_TOKENIZER)
    args.do_sample = True  # generate_openrouter reads args.do_sample; this design always samples.
    n_done = 0
    n_failed_total = 0
    for (slug, polarity), pending in todo.items():
        system_prompt = pending[0]["system_prompt"]
        print(f"\n[{slug} / {polarity}] {len(pending)} to generate via OpenRouter "
              f"(max_workers={args.max_workers}, chunk_size={args.chunk_size})", flush=True)
        for chunk in iter_batches(pending, args.chunk_size):
            messages_list = []
            for item in chunk:
                msgs = [{"role": "system", "content": system_prompt}] if system_prompt else []
                msgs.append({"role": "user", "content": item["user_prompt"]})
                messages_list.append(msgs)
            results = generate_openrouter(messages_list, args)
            records = []
            n_failed = 0
            for item, result in zip(chunk, results):
                if result["finish_reason"] == "request_failed":
                    # Not persisted: load_done would treat it as finished and a
                    # rerun would never retry it.
                    n_failed += 1
                    continue
                response = result["content"]
                n_tok = len(tokenizer(response, add_special_tokens=False)["input_ids"]) if response else 0
                reason = classify_degenerate(response, n_tok, args.max_new_tokens, args.min_response_chars)
                if result["finish_reason"] == "length" and reason != "empty":
                    reason = reason or "truncated"
                records.append({**_record(item, response, n_tok, reason, args), "finish_reason": result["finish_reason"]})
            _write_chunk(run_dir / "generations" / f"{slug}.jsonl", records)
            n_done += len(chunk)
            print(f"  {n_done}/{n_pending} total", flush=True)
            if n_failed:
                n_failed_total += n_failed
                print(f"  WARNING: {n_failed}/{len(chunk)} requests failed and were NOT saved; rerun to retry them",
                      flush=True)
    if n_failed_total:
        print(f"\nWARNING: {n_failed_total} failed requests not saved -- rerun the same command to retry them.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model", type=str, default=None,
                        help=f"HF repo id (local; default {common.DEFAULT_MODEL}) or OpenRouter slug "
                             f"(openrouter; default {DEFAULT_OPENROUTER_MODEL})")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="HF repo id used to count response tokens / render --dry-run prompts "
                             f"(default: --model for local, {DEFAULT_OPENROUTER_TOKENIZER} for openrouter)")
    parser.add_argument("--user-prompts", type=Path, default=common.USER_PROMPTS_PATH,
                        help="jsonl of user prompts (default: the shared perez_user_prompts.jsonl)")
    common.add_cells_arg(parser)
    parser.add_argument("--limit-prompts", type=int, default=None, help="Use only the first N user prompts.")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="High enough that EOS fires naturally; response length is measured, not capped.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    add_backend_args(parser, default_backend="local", system_prompt=False)
    parser.add_argument("--max-workers", type=int, default=32, help="Concurrent OpenRouter requests per chunk.")
    parser.add_argument("--chunk-size", type=int, default=250,
                        help="OpenRouter items per checkpoint step (bounds work lost to a crash mid-group).")
    parser.add_argument("--min-response-chars", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the first prompt of each group, count tokens, load no weights, write nothing.",
    )
    args = parser.parse_args()
    if args.model is None:
        args.model = DEFAULT_OPENROUTER_MODEL if args.backend == "openrouter" else common.DEFAULT_MODEL

    pairs = common.load_prompt_pairs()
    prompts = common.read_jsonl(args.user_prompts)
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]

    work = build_work_list(pairs, prompts, args.cells)
    groups: dict[tuple, list[dict]] = {}
    for item in work:
        groups.setdefault(group_key(item), []).append(item)
    print(f"{len(work)} generations across {len(groups)} (cell, polarity) groups, {len(prompts)} user prompts")

    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer or (DEFAULT_OPENROUTER_TOKENIZER if args.backend == "openrouter" else args.model))
        for key, items in groups.items():
            item = items[0]
            rendered = build_chat_prompt(tokenizer, item["user_prompt"], item["system_prompt"])
            n_tok = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            print(f"\n===== {key} | {n_tok} prompt tokens | {len(items)} items =====")
            print(rendered)
        return

    run_dir = common.resolve_run_dir(args.run_name)
    (run_dir / "generations").mkdir(parents=True, exist_ok=True)
    # Freeze the exact prompts this run used, so the run is self-describing even
    # if the user-prompts file is later regenerated with a different n.
    common.write_jsonl(run_dir / "user_prompts.jsonl", prompts)

    todo = {}
    for key, items in groups.items():
        done = load_done(run_dir / "generations" / f"{key[0]}.jsonl")
        pending = [i for i in items if i["example_id"] not in done]
        if pending:
            todo[key] = pending
    n_pending = sum(len(v) for v in todo.values())
    print(f"{n_pending} pending after resume ({len(work) - n_pending} already on disk)")

    if n_pending:
        (run_local if args.backend == "local" else run_openrouter)(todo, run_dir, n_pending, args)

    touched = sorted({k[0] for k in groups})
    report = report_cells(run_dir, touched)
    print_report(report)
    (run_dir / "checks").mkdir(exist_ok=True)
    (run_dir / "checks" / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    common.write_run_info(
        run_dir,
        # Stage keys kept from the two scripts this replaces, so existing run_info.json files stay comparable.
        "generate_response" if args.backend == "local" else "generate_response_openrouter",
        args,
        {"n_prompts": len(prompts), "n_generations_target": len(work), "cells": touched},
    )
    print(f"\nDone. Records under {run_dir / 'generations'}")


if __name__ == "__main__":
    main()
