#!/usr/bin/env python3
"""Generate fresh responses for an SAE/ELEPHANT JSONL prompt dataset.

Usage:
    python -m probing.evaluations.baseline_probes.generation.generate_sae \
        --input <path> --dataset-type social --model meta-llama/llama-3.1-8b-instruct --output <path>
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.utils.probes_common import read_jsonl, seed_everything, write_jsonl  # noqa: E402
from probing.evaluations.baseline_probes.generation.common import (  # noqa: E402
    add_generation_args, generate_via_openrouter_with_finish_reasons, write_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL with prompt, row_id, and prompt_col")
    parser.add_argument("--dataset-name", help="Defaults to each input row's dataset field")
    parser.add_argument("--dataset-type", choices=["social", "moral"], required=True)
    add_generation_args(parser, 512)
    args = parser.parse_args()
    seed_everything(args.seed)
    rows = read_jsonl(Path(args.input))
    # A pre-generated SAE file often has several sample_idx rows. A fresh run
    # uses each logical prompt once and never inherits the old response.
    unique = {}
    for row in rows:
        key = (str(row.get("row_id")), row.get("prompt_col", "prompt"))
        unique.setdefault(key, row)
    rows = list(unique.values())
    random.Random(args.seed).shuffle(rows)
    if args.limit is not None:
        rows = rows[: args.limit]

    messages = [[{"role": "user", "content": row["prompt"]}] for row in rows]
    generations = generate_via_openrouter_with_finish_reasons(messages, args)
    output_rows = []
    for row, chat, generation in zip(rows, messages, generations):
        response = generation["content"]
        output_rows.append({
            "dataset": args.dataset_name or row.get("dataset"),
            "row_id": row.get("row_id"), "prompt_col": row.get("prompt_col", "prompt"),
            "prompt": row["prompt"], "response": response, "finish_reason": generation["finish_reason"],
            "reasoning": generation["reasoning"],
            "messages": chat + [{"role": "assistant", "content": response}], "model": args.model,
        })
    output = Path(args.output)
    write_jsonl(output, output_rows)
    write_metadata(output, args, args.dataset_type, {"path": str(Path(args.input).resolve())})


if __name__ == "__main__":
    main()
