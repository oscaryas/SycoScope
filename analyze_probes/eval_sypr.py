#!/usr/bin/env python3
"""
Out-of-distribution evaluation on SyPR (verifiable-domain sycophantic praise),
from tool_calling/tasks/sycophancy/scripts/sypr_praise_full_generate.py.

Unlike ELEPHANT/moral (no ground truth), SyPR has one: a row is labelled
sycophantic (1) iff the model praised the user's utterance AND that utterance
was actually poor quality (wrong on reasoning domains, low-quality on social
domains) -- i.e. praise the content doesn't merit, not warranted agreement.

One label off one extraction pass:
  sycophantic_praise  praised (1) vs not (0), restricted to rows with a clear
                       is_poor_quality/praised judgement.

Usage:
    python eval_sypr.py --run-name main --extract-only
    python eval_sypr.py --run-name main
"""
import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.utils import common  # noqa: E402
from probing.analyze_probes import eval_common  # noqa: E402

SYCOPHANCY_DIR = common.SYCOPHANCY_DIR
DEFAULT_JUDGED = SYCOPHANCY_DIR / "results" / "generations" / "sypr_praise_llama31_full" / "checkpoint.jsonl"
LABEL_FIELDS = ("sycophantic_praise",)


def load_judged(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  missing: {path}")
        return []
    rows = common.read_jsonl(path)
    out = []
    for i, r in enumerate(rows):
        if r.get("praised") is None:
            continue
        out.append(
            {
                "dataset": "SyPR",
                "example_id": f"SyPR__{i}",
                "user_prompt": r["utterance_text"],
                "system_prompt": None,
                "chat_prefix": r["prompt"],
                "response": r["response"],
                "sycophantic_praise": int(r["label"]),
                "domain": r.get("domain"),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    eval_common.add_common_args(parser)
    parser.add_argument("--judged", type=Path, default=DEFAULT_JUDGED)
    args = parser.parse_args()

    run_dir = common.resolve_run_dir(args.run_name, create=False)
    if not run_dir.exists():
        raise SystemExit(f"no such run: {run_dir}")
    out_dir = run_dir / "eval_sypr"

    print("Loading judged SyPR records ...")
    records = load_judged(args.judged)
    if not records:
        raise SystemExit("no judged records loaded")
    if args.limit:
        random.Random(args.seed).shuffle(records)
        records = records[: args.limit]
    print(f"{len(records)} records across {len(set(r['domain'] for r in records))} domains")
    pos = sum(r["sycophantic_praise"] for r in records)
    print(f"  sycophantic_praise pos {pos} / neg {len(records) - pos} (pos rate {pos / len(records):.3f})")

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
            "note": "Ground-truth label (SyPR's own is_poor_quality x praised), unlike "
            "ELEPHANT/moral's judge-derived labels. Verifiable-domain sycophancy.",
        },
    )


if __name__ == "__main__":
    main()
