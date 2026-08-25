#!/usr/bin/env python3
"""
Generate on-policy responses under each side of every contrastive system-prompt
pair. Stage 1 of the prompt_probes pipeline.

Paired design: the *same* user prompts are used for both polarities of a cell
and for all 14 cells. Nothing about user-prompt content can then separate the
classes (with disjoint prompt sets a probe could score highly by classifying
topic and you would never know), each prompt yields one label-1 / label-0 pair
for grouped cross-validation, and a common prompt support across cells is what
makes the cross-cell transfer matrices comparable.

The consequence is that reported accuracy is *paired* accuracy, and that
train/test splits must group by prompt_id -- see train_probes.py.

This deviates from Natarajan et al. (2026), who token-force an identical
response under both system prompts (making the response byte-identical across
classes, so the probe can only read the instruction's effect). Going on-policy
means the two classes differ in the instruction *and* the response text; the
`first5` activation position is what recovers the intent measurement, since
five response tokens are too sparse to carry the label on their own.

Checkpointed: every record is appended immediately, so a crash costs at most
one batch, and re-running the same command resumes. Per-cell output files mean
one cell can be regenerated without touching the other 13.

Usage:
    # smoke test, no GPU needed
    python generate_response.py --run-name smoke --model meta-llama/Llama-3.2-1B-Instruct \\
        --cells general_baseline neutral --limit-prompts 4 --max-new-tokens 48 --batch-size 4

    # render prompts and count tokens without loading any weights
    python generate_response.py --run-name smoke --cells general_baseline --limit-prompts 2 --dry-run

    # full run
    python generate_response.py --run-name main --batch-size 16
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
from utils.inference import build_chat_prompt, generate_batch, iter_batches  # noqa: E402

# Cheap lexical probes of "did the system prompt change anything at all",
# reported per cell at the end of the run. Not labels -- diagnostics. A cell
# whose two polarities look identical on all of these is a cell whose probe
# result is uninterpretable however high its AUC, and that matters more here
# than it would under token-forcing, because the on-policy design is premised
# on the behaviour differing.
AGREEMENT_MARKERS = (
    "you're right", "you are right", "great point", "absolutely", "i agree",
    "well said", "excellent point", "spot on", "exactly right", "couldn't agree",
)
REFUSAL_MARKERS = ("i can't", "i cannot", "i won't", "i'm not able to", "i am not able to")


def classify_degenerate(response: str, n_response_tokens: int, max_new_tokens: int, min_chars: int) -> str | None:
    """First matching degeneracy reason, or None. Flags only -- never drops."""
    text = response.strip()
    if not text:
        return "empty"
    if len(text) < min_chars:
        return "too_short"
    # generate_batch decodes with skip_special_tokens=True, so an absent EOT is
    # not directly visible; hitting the token budget is the usable proxy.
    # Reported but NOT a drop reason: hitting the cap does not corrupt the
    # activations (last_prompt and first5 are untouched, response is a mean
    # over a valid prefix), and dropping these would condition the sample on
    # length rather than remove the length effect. With max_new_tokens=1024
    # this should be near-zero anyway; a high rate means the cap is too low.
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


def build_work_list(pairs: list[dict], prompts: list[dict], slugs: list[str] | None, samples: int) -> list[dict]:
    """One entry per (cell, polarity, prompt, sample) still to be generated.

    The `neutral` pseudo-cell has no pair and no polarity: it is a single
    response per prompt with no system prompt, serving as the control set that
    probe scores are centred against (paper section 5.5).
    """
    work = []
    for pair in common.select_cells(pairs, slugs):
        for polarity in common.POLARITIES:
            for prompt in prompts:
                for s in range(samples):
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
                            "sample_idx": s,
                            "example_id": f"{pair['slug']}__{common.POLARITY_TAG[polarity]}__{prompt['prompt_id']}__s{s}",
                        }
                    )
    if slugs is None or common.NEUTRAL_SLUG in slugs:
        for prompt in prompts:
            for s in range(samples):
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
                        "sample_idx": s,
                        "example_id": f"{common.NEUTRAL_SLUG}__neu__{prompt['prompt_id']}__s{s}",
                    }
                )
    return work


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["example_id"] for line in open(path, encoding="utf-8") if line.strip()}


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
                "agreement_marker_rate": round(
                    sum(any(m in t for m in AGREEMENT_MARKERS) for t in texts) / len(rs), 4
                ),
                "refusal_rate": round(sum(any(m in t for m in REFUSAL_MARKERS) for t in texts) / len(rs), 4),
            }
        report[slug] = entry
    return report


def print_report(report: dict) -> None:
    print(f"\n{'cell':<26} {'n':>5} {'degen':>7} {'chars+':>8} {'chars-':>8} {'agree+':>8} {'agree-':>8}")
    print("-" * 76)
    for slug, e in report.items():
        pos, neg = e.get("sycophantic") or e.get("neutral") or {}, e.get("non_sycophantic") or {}
        print(
            f"{slug:<26} {e['n']:>5} {e['degenerate_rate']:>7.3f} "
            f"{pos.get('mean_chars', float('nan')):>8.1f} {neg.get('mean_chars', float('nan')):>8.1f} "
            f"{pos.get('agreement_marker_rate', float('nan')):>8.3f} {neg.get('agreement_marker_rate', float('nan')):>8.3f}"
        )
    # One line. The control-polarity caution lives in common.py beside the
    # label definitions, and the length columns are reported again by
    # analyze_probes.length_analysis; no need to restate either here.
    print(
        "\n'+' = sycophantic slot ('neutral' for the control cell), "
        "'-' = non_sycophantic. Matching columns => the prompt changed nothing."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model", type=str, default=common.DEFAULT_MODEL)
    parser.add_argument("--user-prompts", type=Path, default=common.USER_PROMPTS_PATH)
    common.add_cells_arg(parser)
    parser.add_argument("--limit-prompts", type=int, default=None, help="Use only the first N user prompts.")
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="High enough that EOS fires naturally; response length is measured, not capped.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-response-chars", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the first prompt of each group, count tokens, load no weights.",
    )
    args = parser.parse_args()

    run_dir = common.resolve_run_dir(args.run_name)
    (run_dir / "generations").mkdir(parents=True, exist_ok=True)

    pairs = common.load_prompt_pairs()
    prompts = common.read_jsonl(args.user_prompts)
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    # Freeze the exact prompts this run used, so the run is self-describing even
    # if data/perez_user_prompts.jsonl is later regenerated with a different n.
    common.write_jsonl(run_dir / "user_prompts.jsonl", prompts)

    work = build_work_list(pairs, prompts, args.cells, args.samples_per_prompt)
    groups: dict[tuple, list[dict]] = {}
    for item in work:
        groups.setdefault(group_key(item), []).append(item)
    print(f"{len(work)} generations across {len(groups)} (cell, polarity) groups, {len(prompts)} user prompts")

    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        for key, items in groups.items():
            item = items[0]
            rendered = build_chat_prompt(tokenizer, item["user_prompt"], item["system_prompt"])
            n_tok = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            print(f"\n===== {key} | {n_tok} prompt tokens | {len(items)} items =====")
            print(rendered)
        return

    import torch

    from utils.model import cleanup as cleanup_model, load_model_and_tokenizer

    todo = {}
    for key, items in groups.items():
        slug = key[0]
        done = load_done(run_dir / "generations" / f"{slug}.jsonl")
        pending = [i for i in items if i["example_id"] not in done]
        if pending:
            todo[key] = pending
    n_pending = sum(len(v) for v in todo.values())
    print(f"{n_pending} pending after resume ({len(work) - n_pending} already on disk)")

    if n_pending:
        print(f"\nLoading {args.model} ...")
        model, tokenizer = load_model_and_tokenizer(args.model)
        # Seeded once. Note that resuming mid-run shifts the RNG relative to an
        # uninterrupted run, so post-resume samples differ; recorded in run_info.
        torch.manual_seed(args.seed)

        n_done = 0
        for key, pending in todo.items():
            slug, polarity = key
            system_prompt = pending[0]["system_prompt"]
            out_path = run_dir / "generations" / f"{slug}.jsonl"
            print(f"\n[{slug} / {polarity}] {len(pending)} to generate")
            with open(out_path, "a", encoding="utf-8") as f:
                for chunk in iter_batches(pending, args.batch_size):
                    responses = generate_batch(
                        model,
                        tokenizer,
                        [c["user_prompt"] for c in chunk],
                        system_prompt=system_prompt,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    for item, response in zip(chunk, responses):
                        n_tok = len(tokenizer(response, add_special_tokens=False)["input_ids"])
                        record = {
                            **{k: v for k, v in item.items()},
                            "response": response,
                            "n_response_chars": len(response),
                            "n_response_tokens": n_tok,
                            "degenerate": classify_degenerate(
                                response, n_tok, args.max_new_tokens, args.min_response_chars
                            ),
                            "model": args.model,
                            "max_new_tokens": args.max_new_tokens,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "seed": args.seed,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    n_done += len(chunk)
                    print(f"  {n_done}/{n_pending}", flush=True)

        cleanup_model(model, tokenizer)

    touched = sorted({k[0] for k in groups})
    report = report_cells(run_dir, touched)
    print_report(report)
    (run_dir / "checks").mkdir(exist_ok=True)
    (run_dir / "checks" / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    common.write_run_info(
        run_dir,
        "generate_response",
        args,
        {"n_prompts": len(prompts), "n_generations_target": len(work), "cells": touched},
    )
    print(f"\nDone. Records under {run_dir / 'generations'}")


if __name__ == "__main__":
    main()
