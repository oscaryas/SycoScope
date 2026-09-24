#!/usr/bin/env python3
"""Generate responses for ELEPHANT AITA-YTA (single prompt column, no flip pairing).

Judge afterwards with probing.evaluations.judge.run_moral_sycophancy_judge_aita --mode single.

Usage:
    python -m probing.evaluations.generation.generate_aita_yta \\
        --model meta-llama/llama-3.1-8b-instruct --backend openrouter \\
        --output probing/data/baseline/meta-llama__llama-3.1-8b-instruct/aita_yta/checkpoint.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.utils.probes_common import seed_everything  # noqa: E402
from probing.evaluations.generation.common import add_backend_args, add_generation_args  # noqa: E402
from probing.evaluations.generation.elephant import add_chunk_arg, run_elephant  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_generation_args(parser, 512)
    add_backend_args(parser)
    add_chunk_arg(parser)
    args = parser.parse_args()
    seed_everything(args.seed)
    run_elephant(args, "aita_yta", keep_pairs=False, dataset_type="aita_yta")


if __name__ == "__main__":
    main()
