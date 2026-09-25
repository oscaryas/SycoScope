#!/usr/bin/env python3
"""
Out-of-distribution evaluation on ELEPHANT. Stage 5, and the primary
measurement of the pipeline.

Extraction runs once (GPU, ~45-90 min for ~8.8k records) and is cached; every
probe x position x layer x metric combination then scores as a dot product.

Usage:
    # one GPU pass, cached thereafter
    python -m scorer.eval_social_sycophancy --cache results/probes/scores/elephant.npz --model <model> --extract-only

    # scoring only, CPU
    python -m scorer.eval_social_sycophancy --probe weights.pkl --cache results/probes/scores/elephant.npz
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

JUDGED_DIR = common.SYCOPHANCY_DIR / "results" / "generations"
DEFAULT_JUDGED = [
    JUDGED_DIR / "OEQ_social_sycophancy_judged.jsonl",
    JUDGED_DIR / "SS_social_sycophancy_judged.jsonl",
    JUDGED_DIR / "AITA-YTA_social_sycophancy_judged.jsonl",
]
LABEL_FIELDS = ("validation", "indirectness", "framing")

# Stated before looking at any result. ELEPHANT's metrics correspond to
# taxonomy cells, so each column has a predicted winner:
#   validation   = unconditional emotional validation -> Person-Emotions
#   framing      = accepting the user's framing       -> Position-Subjective/Implicit
#   indirectness = softened feedback / hedged delivery -> the Implicit person cells
MATCHED = {
    "validation": ("pe_explicit", "pe_implicit"),
    "framing": ("ps_implicit",),
    "indirectness": ("pe_implicit", "pt_implicit"),
}


def load_judged(paths: list[Path]) -> list[dict]:
    """Judged records carrying all three metrics, so the 14x3 grid is
    measured on one common set of rows rather than three different ones."""
    out = []
    for path in paths:
        if not path.exists():
            print(f"  missing: {path}")
            continue
        rows = common.read_jsonl(path)
        kept = [r for r in rows if all(r.get(k) is not None for k in LABEL_FIELDS)]
        print(f"  {path.name}: {len(kept)}/{len(rows)} rows with all three metrics")
        for r in kept:
            dataset = r.get("dataset") or path.stem.split("_")[0]
            out.append(
                {
                    "dataset": dataset,
                    "row_id": r.get("row_id"),
                    "user_prompt": r["prompt"],
                    "system_prompt": None,  # matches how these were generated
                    "response": r["response"],
                    "example_id": f"{dataset}__{r.get('row_id')}",
                    **{k: int(bool(r[k])) for k in LABEL_FIELDS},
                }
            )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    eval_common.add_common_args(parser)
    parser.add_argument("--judged", type=Path, nargs="+", default=DEFAULT_JUDGED)
    parser.add_argument("--label-fields", type=str, nargs="+", default=list(LABEL_FIELDS), choices=LABEL_FIELDS)
    args = parser.parse_args()

    paths = eval_common.cache_paths(args.cache, args.positions)
    if eval_common.needs_extraction(paths, args):
        print("Loading judged ELEPHANT records ...")
        records = load_judged(list(args.judged))
        if not records:
            raise SystemExit("no judged records loaded")
        if args.limit:
            # Seeded subsample, not a head: the files concatenate in order, so a
            # head takes one dataset only.
            random.Random(args.seed).shuffle(records)
            records = records[: args.limit]
        print(f"{len(records)} records across {len(set(r['dataset'] for r in records))} datasets")
        for f in args.label_fields:
            pos = sum(r[f] for r in records)
            print(f"  {f:<13} pos {pos} / neg {len(records) - pos}  (pos rate {pos / len(records):.3f})")
        eval_common.build_cache(paths, records, LABEL_FIELDS, LABEL_FIELDS, args)
    if args.extract_only:
        return

    eval_common.score_and_report(
        args, paths, "eval_social_sycophancy", args.label_fields,
        extra={
            "matched_prediction": {k: list(v) for k, v in MATCHED.items()},
            "note": "Judge labels, not ground truth, and 83-90% positive. AUC is base-rate "
            "insensitive but the minority class is small; read n_pos/n_neg alongside.",
        },
        matched=MATCHED,
    )


if __name__ == "__main__":
    main()
