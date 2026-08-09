#!/usr/bin/env python3
"""
Assembles the final 3-way balanced sycophancy mixture from the three already-
labeled sources:
    - SyPR sycophantic praise   (results/sypr_praise_llama31_full/checkpoint.jsonl)
    - MMLU are_you_sure caving  (results/mmlu_are_you_sure_full/checkpoint.jsonl)
    - pooled social sycophancy  (results/social_sycophancy_pooled/pooled.jsonl)

Each source already has a binary `label` field (1 = sycophantic, 0 = not).
Per category, splits into positive/negative pools and caps at
min(len(pos), len(neg)) so the category itself is balanced; then takes the
minimum of those three per-category caps as the shared size n, so every
category contributes exactly n positive + n negative rows. Samples are drawn
with a fixed seed, tagged with `category`, normalized to a common
{text, label, category, meta} schema, concatenated, and shuffled once more
(a genuine mixture, not blocked by source) before writing out.

Usage:
    python build_sycophancy_mixture.py \
        --sypr results/sypr_praise_llama31_full/checkpoint.jsonl \
        --mmlu results/mmlu_are_you_sure_full/checkpoint.jsonl \
        --social results/social_sycophancy_pooled/pooled.jsonl \
        --out results/sycophancy_mixture/mixture.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Fields specific to each source, preserved under the row's "meta" dict --
# everything else in the raw record is source-specific bookkeeping (e.g.
# SyPR's "prompt"/"response" vs. MMLU's "turn1_response"/"turn2_response")
# that doesn't belong in a unified schema.
META_FIELDS = {
    "sypr": ("domain", "utterance_text", "is_poor_quality", "praised"),
    "mmlu": ("domain", "topic", "question", "correct_letter", "turn1_letter", "turn2_letter"),
    "social": ("domain", "validation", "indirectness", "framing"),
}


def load_category(path: Path, category: str) -> list:
    records = [json.loads(line) for line in open(path)]
    normalized = []
    for r in records:
        normalized.append(
            {
                "text": r["text"],
                "label": r["label"],
                "category": category,
                "meta": {k: r[k] for k in META_FIELDS[category] if k in r},
            }
        )
    return normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sypr", type=str, default=str(HERE / "results" / "sypr_praise_llama31_full" / "checkpoint.jsonl"))
    parser.add_argument("--mmlu", type=str, default=str(HERE / "results" / "mmlu_are_you_sure_full" / "checkpoint.jsonl"))
    parser.add_argument("--social", type=str, default=str(HERE / "results" / "social_sycophancy_pooled" / "pooled.jsonl"))
    parser.add_argument("--out", type=str, default=str(HERE / "results" / "sycophancy_mixture" / "mixture.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sources = {
        "sypr": Path(args.sypr),
        "mmlu": Path(args.mmlu),
        "social": Path(args.social),
    }
    for name, path in sources.items():
        if not path.exists():
            print(f"ERROR: {name} source not found at {path}", file=sys.stderr)
            sys.exit(1)

    rng = random.Random(args.seed)

    pools = {}
    per_category_n = {}
    for name, path in sources.items():
        rows = load_category(path, name)
        pos = [r for r in rows if r["label"] == 1]
        neg = [r for r in rows if r["label"] == 0]
        pools[name] = (pos, neg)
        per_category_n[name] = min(len(pos), len(neg))
        print(f"{name}: {len(rows)} total, {len(pos)} positive, {len(neg)} negative -> category cap {per_category_n[name]}")

    n = min(per_category_n.values())
    print(f"\nShared cap across all 3 categories: n={n} (2n={2*n} per category, {6*n} total)")

    mixture = []
    for name, (pos, neg) in pools.items():
        sampled_pos = rng.sample(pos, n)
        sampled_neg = rng.sample(neg, n)
        mixture.extend(sampled_pos)
        mixture.extend(sampled_neg)

    rng.shuffle(mixture)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in mixture:
            f.write(json.dumps(row) + "\n")

    # Verification: re-read and assert the invariants the docstring promises.
    written = [json.loads(line) for line in open(out_path)]
    assert len(written) == 6 * n, f"expected {6*n} rows, got {len(written)}"
    for name in sources:
        cat_rows = [r for r in written if r["category"] == name]
        assert len(cat_rows) == 2 * n, f"{name}: expected {2*n} rows, got {len(cat_rows)}"
        cat_pos = sum(r["label"] == 1 for r in cat_rows)
        assert cat_pos == n, f"{name}: expected {n} positive, got {cat_pos}"

    summary = {
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
    print(f"Summary written to {summary_path}")

    print("\n--- one example row per category ---")
    for name in sources:
        example = next(r for r in written if r["category"] == name)
        print(f"\n[{name}] label={example['label']}")
        print(example["text"][:500])


if __name__ == "__main__":
    main()
