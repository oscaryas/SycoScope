#!/usr/bin/env python3
"""
OOD target 2 for the llama31_5k_subset initial exploration: does a probe
trained on ONE cell's system prompt (e.g. pv_implicit) also separate
sycophantic/non_sycophantic responses generated under a DIFFERENT cell's
system prompt (e.g. ctrl_obsequiousness)? Tests cross-system-prompt
generalization within the same synthetic contrastive-pair design, as opposed
to eval_moral.py/eval_social_sycophancy.py's generalization to naturalistic AITA data.

Records are built from the ~500-prompt held-out sample (cross_cell_holdout_prompt_ids.json,
drawn from the 3000 prompt_ids NOT in the 2000-prompt training pool) across ALL
of the run's cells, dataset field = source cell slug. One GPU extraction pass
covers every cell; per-probe "cross" results are read off eval_common's
existing per-dataset AUC rows (each individual dataset's AUC, skipping the
row where dataset == the probe's own cell when interpreting "cross-cell"
transfer -- the "all" pooled row DOES include the self-cell, so use the
per-dataset breakdown, not "all", for the cross-cell number).

Label field: "sycophantic" (int(row["label"]), 1 = sycophantic polarity, 0 =
non_sycophantic -- identical to POLARITY_LABEL, no external judging needed).

Usage:
    python eval_cross_cell.py --run-name llama31_5k_subset --extract-only
    python eval_cross_cell.py --run-name llama31_5k_subset
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.utils import common  # noqa: E402
from probing.analyze_probes import eval_common  # noqa: E402

LABEL_FIELDS = ("sycophantic",)
SRC_DIR = common.REPO_ROOT / "prompt_probes" / "results" / "llama31_5k" / "generations"


def load_cross_cell_records(run_dir: Path, cells: list[str]) -> list[dict]:
    holdout_path = run_dir / "cross_cell_holdout_prompt_ids.json"
    if not holdout_path.exists():
        raise SystemExit(f"missing: {holdout_path} -- run build_llama31_subset.py first")
    holdout = set(__import__("json").loads(holdout_path.read_text())["prompt_ids"])
    print(f"  held-out sample: {len(holdout)} prompt_ids")

    out = []
    for slug in cells:
        path = SRC_DIR / f"{slug}.jsonl"
        if not path.exists():
            print(f"  missing: {path}")
            continue
        rows = common.read_jsonl(path)
        kept = [r for r in rows if r["prompt_id"] in holdout]
        for r in kept:
            out.append(
                {
                    "dataset": slug,
                    "example_id": r["example_id"],
                    "user_prompt": r["user_prompt"],
                    "system_prompt": r["system_prompt"],
                    "response": r["response"],
                    "sycophantic": int(r["label"]),
                }
            )
        print(f"  {slug}: {len(kept)}/{len(rows)} rows in the held-out sample")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    eval_common.add_common_args(parser)
    args = parser.parse_args()

    run_dir = common.resolve_run_dir(args.run_name, create=False)
    if not run_dir.exists():
        raise SystemExit(f"no such run: {run_dir}")
    out_dir = run_dir / "eval_cross_cell"

    cells = args.cells or common.all_slugs(include_neutral=False)
    print("Loading held-out cross-cell records ...")
    records = load_cross_cell_records(run_dir, cells)
    if not records:
        raise SystemExit("no records loaded")
    print(f"{len(records)} records across {len(set(r['dataset'] for r in records))} cells")

    if args.overwrite or not (out_dir / "activations.npz").exists():
        eval_common.extract_activations(out_dir, records, LABEL_FIELDS, args)
    else:
        print(f"using cached activations at {out_dir / 'activations.npz'} (--overwrite to redo)")
    if args.extract_only:
        return

    positions, layers, slugs = eval_common.resolve_eval_targets(run_dir, out_dir, args)
    index = common.read_jsonl(out_dir / "activations_index.jsonl")
    selection = eval_common.selection_split(index, args.selection_frac, args.seed)
    print(f"\nselection split: {len(selection)} keys reserved for probe selection")

    rows = eval_common.score_all(run_dir, out_dir, slugs, positions, layers, LABEL_FIELDS, selection, args.seed)
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
            "note": "Cross-system-prompt generalization: dataset field = source cell slug. "
            "The 'all' row for a probe includes its OWN cell's held-out data (not a cross-cell "
            "test) -- read the per-dataset breakdown and exclude the row matching the probe's "
            "own slug to get the genuine cross-cell number.",
        },
    )


if __name__ == "__main__":
    main()
