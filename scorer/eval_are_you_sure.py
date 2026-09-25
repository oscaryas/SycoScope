#!/usr/bin/env python3
"""
Out-of-distribution evaluation on "are you sure" caving: the model answered
correctly, was pushed back on, and either held or flipped. The label is
behavioural, and mechanical (no judge) for the multiple-choice subsets.

Inputs are the checkpoints written by scripts/are_you_sure_{mc,freeform}_generate.py.
The probed context is each record's `turn2_prompt`: the chat-formatted turn-1
question, turn-1 answer and pushback that the model saw before answering again.

Usage:
    # one GPU pass, cached thereafter
    python -m scorer.eval_are_you_sure --cache results/probes/scores/ays.npz --model <model> --extract-only
    # scoring only, CPU
    python -m scorer.eval_are_you_sure --probe weights.pkl --cache results/probes/scores/ays.npz
"""
import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import common  # noqa: E402
from scorer import eval_common  # noqa: E402

DEFAULT_CHECKPOINTS = [
    common.GENERATIONS_DIR / "are_you_sure_mc" / "checkpoint.jsonl",
    common.GENERATIONS_DIR / "are_you_sure_freeform" / "checkpoint.jsonl",
]
LABEL_FIELDS = ("caved",)


def load_checkpoints(paths: list[Path]) -> list[dict]:
    out = []
    for path in paths:
        if not path.exists():
            print(f"  missing: {path}")
            continue
        rows = common.read_jsonl(path)
        print(f"  {path.name}: {len(rows)} rows ({sum(r['label'] for r in rows)} caved)")
        for i, r in enumerate(rows):
            out.append(
                {
                    "dataset": r["domain"],
                    "example_id": f"{r['domain']}__{path.parent.name}__{i}",
                    "chat_prefix": r["turn2_prompt"],
                    "response": r["turn2_response"],
                    "caved": int(r["label"]),
                }
            )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    eval_common.add_common_args(parser)
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=DEFAULT_CHECKPOINTS)
    args = parser.parse_args()

    paths = eval_common.cache_paths(args.cache, args.positions)
    if eval_common.needs_extraction(paths, args):
        print("Loading are_you_sure checkpoints ...")
        records = load_checkpoints(list(args.checkpoints))
        if not records:
            raise SystemExit("no records loaded")
        if args.limit:
            random.Random(args.seed).shuffle(records)
            records = records[: args.limit]
        n_pos = sum(r["caved"] for r in records)
        print(f"{len(records)} records across {len(set(r['dataset'] for r in records))} subsets")
        print(f"  caved {n_pos} / held {len(records) - n_pos}  (caving rate {n_pos / len(records):.3f})")
        eval_common.build_cache(paths, records, LABEL_FIELDS, LABEL_FIELDS, args)
    if args.extract_only:
        return

    eval_common.score_and_report(
        args, paths, "eval_are_you_sure", LABEL_FIELDS,
        extra={
            "note": "`response` pools the whole turn-2 answer, which states the letter on the "
            "MC subsets -- read it as a decodability ceiling. `last_prompt` precedes any "
            "turn-2 token, so it is the prediction number.",
        },
    )


if __name__ == "__main__":
    main()
