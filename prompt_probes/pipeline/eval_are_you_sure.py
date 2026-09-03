#!/usr/bin/env python3
"""
Out-of-distribution evaluation on "are you sure" caving: the model answered
correctly, was pushed back on, and either held or flipped. The label is
behavioural, and mechanical (no judge) for the multiple-choice subsets.

Inputs are the checkpoints written by scripts/are_you_sure_{mc,freeform}_generate.py.
The probed context is each record's `turn2_prompt`: the chat-formatted turn-1
question, turn-1 answer and pushback that the model saw before answering again.

Usage:
    python eval_are_you_sure.py --run-name main --extract-only
    python eval_are_you_sure.py --run-name main
"""
import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
import eval_common  # noqa: E402

GENERATIONS_DIR = common.SYCOPHANCY_DIR / "results" / "generations"
DEFAULT_CHECKPOINTS = [
    GENERATIONS_DIR / "are_you_sure_mc_probe_eval" / "checkpoint.jsonl",
    GENERATIONS_DIR / "are_you_sure_freeform_probe_eval" / "checkpoint.jsonl",
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

    run_dir = common.resolve_run_dir(args.run_name, create=False)
    if not run_dir.exists():
        raise SystemExit(f"no such run: {run_dir}")
    out_dir = run_dir / "eval_are_you_sure"

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

    if args.overwrite or not (out_dir / "activations.npz").exists():
        eval_common.extract_activations(out_dir, records, LABEL_FIELDS, args)
    else:
        print(f"using cached activations at {out_dir / 'activations.npz'} (--overwrite to redo)")
    if args.extract_only:
        return

    positions, layers, slugs = eval_common.resolve_eval_targets(run_dir, out_dir, args)
    index = common.read_jsonl(out_dir / "activations_index.jsonl")
    selection = eval_common.selection_split(index, args.selection_frac, args.seed)
    print(f"\nselection split: {len(selection)} rows reserved, {len(index) - len(selection)} for reporting")

    rows = eval_common.score_all(
        run_dir, out_dir, slugs, positions, layers, LABEL_FIELDS, selection, args.seed
    )
    eval_common.report(
        out_dir,
        rows,
        positions,
        layers,
        LABEL_FIELDS,
        slugs,
        {
            "n_records": len(index),
            "selection_frac": args.selection_frac,
            "seed": args.seed,
            "note": "`response` pools the whole turn-2 answer, which states the letter on the "
            "MC subsets -- read it as a decodability ceiling. `last_prompt` precedes any "
            "turn-2 token, so it is the prediction number.",
        },
    )


if __name__ == "__main__":
    main()
