#!/usr/bin/env python3
"""Run one normalized social, moral, correctness, MCQ, or SyPR judge."""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.utils.probes_common import json_dump, read_jsonl, write_jsonl
from probing.evaluations.judge.scoring import score_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dataset-type",
        choices=["social", "moral", "sypr", "mcq", "are_you_sure_mcq", "correctness", "are_you_sure"],
        required=True,
    )
    parser.add_argument("--judge-model", default="claude-sonnet-5")
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()
    summary, rows = score_rows(read_jsonl(Path(args.input)), args.dataset_type, args.judge_model, args.max_workers)
    output = Path(args.output)
    write_jsonl(output, rows)
    json_dump(output.with_suffix(".summary.json"), summary)


if __name__ == "__main__":
    main()
