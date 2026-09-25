#!/usr/bin/env python3
"""
gemma-4-12B-it analogue of build_sycophancy_mixture.py -- assembles a 4-way
balanced sycophancy mixture from ONLY this session's google/gemma-4-12B-it
generations (no llama31 data mixed in):

    - sypr          probing/results/probes/google__gemma-4-12B-it/sypr_merged/checkpoint.jsonl
                    (worker1 rows 0-2290 + worker2 real rows 2500-2632 of the
                    seed=0 shuffle, already has text+label; see summary.json
                    in that dir for the merge note)
    - are_you_sure  are_you_sure_freeform/checkpoint.jsonl + are_you_sure_mc/checkpoint.jsonl
                    concatenated (both already have text+label per row)
    - social        oeq/judged.jsonl + ss/judged.jsonl, text = prompt+response,
                    label = the "validation" metric (ELEPHANT's core social-
                    sycophancy signal: does the response emotionally validate
                    the user). Matches this session's sycophancy_rates.html
                    artifact numbers exactly (OEQ 895/1193, SS 994/1200).
    - moral         aita_nta_flip/judged.jsonl, text = flipped_story_prompt +
                    flipped_story_response (flipped-story side ONLY, not both
                    sides concatenated -- avoids the 1024-token truncation bug
                    moral_avg_residual_probe_pipeline.py's docstring documents
                    for llama31, where concatenating both stories let the
                    separator/second story fall past the truncation cutoff).
                    label = 1 iff original_post_verdict == flipped_story_verdict == "NTA"
                    (both-NTA, matches the artifact's 464/1200=38.7%), label = 0
                    iff both YTA or either side OTHER/refused (matches
                    moral_avg's redefined negative class: "everything that
                    isn't Both-NTA" would wrongly lump in Mixed as if that were
                    non-sycophantic). Mixed pairs (one NTA, one YTA) are
                    EXCLUDED entirely, not folded into either class.

Each source is capped at min(len(pos), len(neg)) so the category itself is
balanced (the "minority sample" == undersample step); then takes the min of
those per-category caps across all 4 categories as the shared size n, so
every category contributes exactly n positive + n negative rows -- same
algorithm as build_sycophancy_mixture.py, just gemma-only inputs/loaders.

Usage:
    python -m probing.evaluations.generation.build_gemma_sycophancy_mixture \
        --out probing/results/probes/google__gemma-4-12B-it/gemma_sycophancy_mixture/mixture.jsonl

Other inputs (are_you_sure / oeq / ss / aita_nta_flip) default to
probing/data/baseline/google__gemma-4-12B-it/<source>/.
"""
import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GEMMA_DIR = REPO_ROOT / "probing" / "data" / "baseline" / "google__gemma-4-12B-it"
GEMMA_PROBES = REPO_ROOT / "probing" / "results" / "probes" / "google__gemma-4-12B-it"


def load_sypr(path: Path) -> list:
    rows = [json.loads(line) for line in open(path)]
    return [
        {
            "text": r["text"],
            "label": r["label"],
            "category": "sypr",
            "meta": {k: r[k] for k in ("domain", "utterance_text", "is_poor_quality", "praised") if k in r},
        }
        for r in rows
    ]


def load_are_you_sure(freeform_path: Path, mc_path: Path) -> list:
    out = []
    for path, source in ((freeform_path, "freeform"), (mc_path, "mc")):
        rows = [json.loads(line) for line in open(path)]
        for r in rows:
            out.append(
                {
                    "text": r["text"],
                    "label": r["label"],
                    "category": "are_you_sure",
                    "meta": {
                        k: r[k]
                        for k in ("domain", "question", "correct_letter", "turn1_letter", "turn2_letter", "answers", "turn2_correct")
                        if k in r
                    }
                    | {"source": source},
                }
            )
    return out


def load_social(oeq_path: Path, ss_path: Path) -> list:
    out = []
    for path in (oeq_path, ss_path):
        rows = [json.loads(line) for line in open(path)]
        for r in rows:
            if r["validation"] is None:
                continue  # judge parse failure
            out.append(
                {
                    "text": r["prompt"] + r["response"],
                    "label": r["validation"],
                    "category": "social",
                    "meta": {
                        "domain": r["dataset"],
                        "validation": r["validation"],
                        "indirectness": r.get("indirectness"),
                        "framing": r.get("framing"),
                        "row_id": r["row_id"],
                    },
                }
            )
    return out


def load_moral(path: Path) -> list:
    rows = [json.loads(line) for line in open(path)]
    out = []
    for r in rows:
        a, b = r["original_post_verdict"], r["flipped_story_verdict"]
        if a == "NTA" and b == "NTA":
            label = 1
        elif a == "YTA" and b == "YTA":
            label = 0
        elif a in ("NTA", "YTA") and b in ("NTA", "YTA"):
            continue  # Mixed -- excluded, not "not sycophantic"
        else:
            label = 0  # either side OTHER/refused -> negative pool, per moral_avg's redefinition
        out.append(
            {
                "text": r["flipped_story_prompt"] + r["flipped_story_response"],
                "label": label,
                "category": "moral",
                "meta": {
                    "domain": "aita_nta_flip",
                    "row_id": r["row_id"],
                    "original_post_verdict": a,
                    "flipped_story_verdict": b,
                },
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sypr", type=str, default=str(GEMMA_PROBES / "sypr_merged" / "checkpoint.jsonl"))
    parser.add_argument("--are-you-sure-freeform", type=str, default=str(GEMMA_DIR / "are_you_sure_freeform" / "checkpoint.jsonl"))
    parser.add_argument("--are-you-sure-mc", type=str, default=str(GEMMA_DIR / "are_you_sure_mc" / "checkpoint.jsonl"))
    parser.add_argument("--oeq", type=str, default=str(GEMMA_DIR / "oeq" / "judged.jsonl"))
    parser.add_argument("--ss", type=str, default=str(GEMMA_DIR / "ss" / "judged.jsonl"))
    parser.add_argument("--moral", type=str, default=str(GEMMA_DIR / "aita_nta_flip" / "judged.jsonl"))
    parser.add_argument("--out", type=str, default=str(GEMMA_PROBES / "gemma_sycophancy_mixture" / "mixture.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    required = [args.sypr, args.are_you_sure_freeform, args.are_you_sure_mc, args.oeq, args.ss, args.moral]
    for p in required:
        if not Path(p).exists():
            print(f"ERROR: source not found at {p}", file=sys.stderr)
            sys.exit(1)

    rng = random.Random(args.seed)

    pools = {
        "sypr": load_sypr(Path(args.sypr)),
        "are_you_sure": load_are_you_sure(Path(args.are_you_sure_freeform), Path(args.are_you_sure_mc)),
        "social": load_social(Path(args.oeq), Path(args.ss)),
        "moral": load_moral(Path(args.moral)),
    }

    per_category_n = {}
    split_pools = {}
    for name, rows in pools.items():
        pos = [r for r in rows if r["label"] == 1]
        neg = [r for r in rows if r["label"] == 0]
        split_pools[name] = (pos, neg)
        per_category_n[name] = min(len(pos), len(neg))
        print(f"{name}: {len(rows)} total, {len(pos)} positive, {len(neg)} negative -> category cap {per_category_n[name]}")

    n = min(per_category_n.values())
    k = len(pools)
    print(f"\nShared cap across all {k} categories: n={n} (2n={2*n} per category, {2*k*n} total)")

    mixture = []
    for name, (pos, neg) in split_pools.items():
        mixture.extend(rng.sample(pos, n))
        mixture.extend(rng.sample(neg, n))
    rng.shuffle(mixture)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in mixture:
            f.write(json.dumps(row) + "\n")

    written = [json.loads(line) for line in open(out_path)]
    assert len(written) == 2 * k * n, f"expected {2*k*n} rows, got {len(written)}"
    for name in pools:
        cat_rows = [r for r in written if r["category"] == name]
        assert len(cat_rows) == 2 * n, f"{name}: expected {2*n} rows, got {len(cat_rows)}"
        cat_pos = sum(r["label"] == 1 for r in cat_rows)
        assert cat_pos == n, f"{name}: expected {n} positive, got {cat_pos}"

    summary = {
        "model": "google/gemma-4-12B-it",
        "seed": args.seed,
        "n_per_category_per_label": n,
        "total_rows": len(written),
        "category_caps": per_category_n,
    }
    summary_path = out_path.parent / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {len(written)} rows to {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
