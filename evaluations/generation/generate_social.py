#!/usr/bin/env python3
"""Generate responses for the ELEPHANT social-sycophancy sources (OEQ, SS).

Judge afterwards with evaluations.judge.run_social_sycophancy_judge_oeq.
AITA-YTA (single-prompt, judged by the moral judge's --mode single) has its
own script, generate_aita_yta.py.

Usage:
    python -m evaluations.generation.generate_social --dataset oeq \\
        --model meta-llama/llama-3.1-8b-instruct --backend openrouter \\
        --output data/baseline/meta-llama__llama-3.1-8b-instruct/oeq/checkpoint.jsonl

    python -m evaluations.generation.generate_social --dataset ss \\
        --model meta-llama/Meta-Llama-3-8B-Instruct --backend local --system-prompt pe_explicit \\
        --output data/system_prompt/meta-llama__Meta-Llama-3-8B-Instruct/pe_explicit/ss.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.probes_common import seed_everything  # noqa: E402
from evaluations.generation.common import add_backend_args, add_generation_args  # noqa: E402
from evaluations.generation.elephant import add_chunk_arg, run_elephant  # noqa: E402

DATASETS = ("oeq", "ss")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="oeq")
    add_generation_args(parser, 512)
    add_backend_args(parser)
    add_chunk_arg(parser)
    args = parser.parse_args()
    seed_everything(args.seed)
    run_elephant(args, args.dataset, keep_pairs=False, dataset_type="social")


if __name__ == "__main__":
    main()
