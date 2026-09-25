#!/usr/bin/env python3
"""Generate SyPR (vennemeyerd/sycophantic-praise) responses only; praise judging is a separate stage.

Every label-eligible row is used (seeded shuffle, optional --limit). Each
prompt is the row's persona-calibration history plus the final utterance
(sypr_data.build_chat_messages). Judge with
evaluations.judge.judge_dataset --dataset-type sypr.

Usage:
    python -m evaluations.generation.generate_sypr \\
        --model meta-llama/llama-3.1-8b-instruct --backend openrouter --output <path>
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.probes_common import seed_everything, write_jsonl  # noqa: E402
from evaluations.generation.common import (  # noqa: E402
    add_backend_args, add_generation_args, make_generator, resolve_system_prompt, with_system_prompt, write_metadata,
)
from evaluations.generation.sypr_data import (  # noqa: E402
    HF_DATASET_ID, _row_from_index, all_eligible_indices, build_chat_messages, load_sypr_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_generation_args(parser, 200)
    add_backend_args(parser)
    args = parser.parse_args()
    seed_everything(args.seed)
    system_prompt = resolve_system_prompt(args)
    dataset = load_sypr_dataset()
    indices = all_eligible_indices(dataset)
    random.Random(args.seed).shuffle(indices)
    if args.limit is not None:
        indices = indices[: args.limit]
    rows = [_row_from_index(dataset, index) for index in indices]
    messages = [with_system_prompt(build_chat_messages(row), system_prompt) for row in rows]
    generations = make_generator(args)(messages)
    output_rows = []
    for index, row, chat, generation in zip(indices, rows, messages, generations):
        response = generation["content"]
        output_rows.append({
            **{k: v for k, v in row.items() if k not in {"conversation_history_json", "utterance_json"}},
            "id": str(index), "row_id": str(index),
            "response": response, "finish_reason": generation["finish_reason"],
            "reasoning": generation["reasoning"],
            "system_prompt": system_prompt, "backend": args.backend,
            "messages": chat + [{"role": "assistant", "content": response}],
            "model": args.model,
        })
    output = Path(args.output)
    write_jsonl(output, output_rows)
    write_metadata(output, args, "sypr", {"huggingface_dataset": HF_DATASET_ID})


if __name__ == "__main__":
    main()
