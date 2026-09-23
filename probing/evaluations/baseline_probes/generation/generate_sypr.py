#!/usr/bin/env python3
"""Generate SyPR responses only; praise judging is a separate stage.

Usage:
    python -m probing.evaluations.baseline_probes.generation.generate_sypr \
        --model meta-llama/llama-3.1-8b-instruct --output <path>
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.utils.baseline_probes_common import seed_everything, write_jsonl  # noqa: E402
from probing.evaluations.baseline_probes.generation.common import (  # noqa: E402
    add_generation_args, generate_via_openrouter_with_finish_reasons, write_metadata,
)
from probing.data.sypr_data import (  # noqa: E402
    HF_DATASET_ID, _row_from_index, all_eligible_indices, build_chat_messages, load_sypr_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_generation_args(parser, 200)
    args = parser.parse_args()
    seed_everything(args.seed)
    dataset = load_sypr_dataset()
    indices = all_eligible_indices(dataset)
    random.Random(args.seed).shuffle(indices)
    if args.limit is not None:
        indices = indices[: args.limit]
    rows = [_row_from_index(dataset, index) for index in indices]
    messages = [build_chat_messages(row) for row in rows]
    generations = generate_via_openrouter_with_finish_reasons(messages, args)
    output_rows = []
    for index, row, chat, generation in zip(indices, rows, messages, generations):
        response = generation["content"]
        output_rows.append({
            **{k: v for k, v in row.items() if k not in {"conversation_history_json", "utterance_json"}},
            "id": str(index), "row_id": str(index),
            "response": response, "finish_reason": generation["finish_reason"],
            "reasoning": generation["reasoning"],
            "messages": chat + [{"role": "assistant", "content": response}],
            "model": args.model,
        })
    output = Path(args.output)
    write_jsonl(output, output_rows)
    write_metadata(output, args, "sypr", {"huggingface_dataset": HF_DATASET_ID})


if __name__ == "__main__":
    main()
