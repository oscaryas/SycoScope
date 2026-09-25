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
    python -m scorer.eval_sypr --cache results/probes/scores/sypr.npz --model <model> --extract-only
    python -m scorer.eval_sypr --probe weights.pkl --cache results/probes/scores/sypr.npz
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

    paths = eval_common.cache_paths(args.cache, args.positions)
    if eval_common.needs_extraction(paths, args):
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
        eval_common.build_cache(paths, records, LABEL_FIELDS, (*LABEL_FIELDS, "domain"), args)
    if args.extract_only:
        return

    eval_common.score_and_report(
        args, paths, "eval_sypr", LABEL_FIELDS,
        extra={
            "note": "Ground-truth label (SyPR's own is_poor_quality x praised), unlike "
            "ELEPHANT/moral's judge-derived labels. Verifiable-domain sycophancy.",
        },
    )


if __name__ == "__main__":
    main()
