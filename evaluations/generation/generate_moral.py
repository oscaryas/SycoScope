#!/usr/bin/env python3
"""Generate responses for the ELEPHANT moral-sycophancy sources (AITA-NTA-FLIP, AITA-NTA-OG).

AITA-NTA-FLIP has two prompt columns per row (original_post, flipped_story);
both are generated as separate records sharing row_id, as the flip judge's
pairing expects. --limit caps ROWS (keeping complete flip pairs), not records.
Judge afterwards with probing.evaluations.judge.run_moral_sycophancy_judge_aita
(--mode flip for aita_nta_flip, --mode single for aita_nta_og).

Usage:
    python -m probing.evaluations.generation.generate_moral --dataset aita_nta_flip \\
        --model meta-llama/llama-3.1-8b-instruct --backend openrouter \\
        --output probing/data/baseline/meta-llama__llama-3.1-8b-instruct/aita_nta_flip/checkpoint.jsonl

    python -m probing.evaluations.generation.generate_moral --dataset aita_nta_flip \\
        --model meta-llama/Meta-Llama-3-8B-Instruct --backend local --system-prompt ctrl_affiliation \\
        --output probing/data/system_prompt/meta-llama__Meta-Llama-3-8B-Instruct/ctrl_affiliation/aita_nta_flip.jsonl
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

DATASETS = ("aita_nta_flip", "aita_nta_og")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="aita_nta_flip")
    add_generation_args(parser, 512)
    add_backend_args(parser)
    add_chunk_arg(parser)
    args = parser.parse_args()
    seed_everything(args.seed)
    run_elephant(args, args.dataset, keep_pairs=True, dataset_type="moral")


if __name__ == "__main__":
    main()
