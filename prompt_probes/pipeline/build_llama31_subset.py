#!/usr/bin/env python3
"""
One-off: build the llama31_5k_subset run directory for the 5-cell initial
exploration (L2 sweep + dual OOD). Reads the existing 5000-prompt
llama31_5k generations (read-only) and:

  1. Picks a fixed, seeded 2000-prompt_id subsample (same prompt_ids across
     all 5 cells -- the paired sycophantic/non_sycophantic design requires
     this) out of the 5000 available. Records the split (train pool /
     held-out pool) to prompt_pool_split.json.
  2. Writes each of the 5 cells' 2000-prompt subsample (4000 rows: 2000
     sycophantic + 2000 non_sycophantic) to
     llama31_5k_subset/generations/<slug>.jsonl -- this is get_activations.py's
     expected input.
  3. Writes prompt_split.json (train/test WITHIN the 2000-prompt pool, for
     train_probes.py's holdout gate) directly, so train_probes.py's
     load_or_make_split() finds it pre-built and doesn't need
     user_prompts.jsonl to exist in this run dir at all.
  4. Also picks a ~500-prompt_id sample from the 3000 held-out prompts, for
     the cross-cell OOD target (step 5) -- written to
     cross_cell_holdout_prompt_ids.json. Since the SAME held-out sample is
     reused across all "other" cells, this is a separate file from the
     prompt pool split above rather than baked into it.

Does not touch prompt_probes/results/main/ or llama31_5k/generations/ (read
only).
"""
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "prompt_probes" / "results" / "llama31_5k" / "generations"
RUN_DIR = REPO_ROOT / "prompt_probes" / "results" / "llama31_5k_subset"
CELLS = ["general_baseline", "pv_implicit", "pe_implicit", "ctrl_pure_sycophancy", "ctrl_obsequiousness"]

N_TRAIN_POOL = 2000
N_HELDOUT_SAMPLE = 500
SEED = 0


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    # Use general_baseline as the reference for the full 5000-prompt_id pool
    # (round-robin sampling upstream means every cell shares the identical
    # prompt_id set -- verified: 5000 unique ids in each of the 20 cells).
    ref_rows = read_jsonl(SRC_DIR / "general_baseline.jsonl")
    all_prompt_ids = sorted(set(r["prompt_id"] for r in ref_rows))
    assert len(all_prompt_ids) == 5000, f"expected 5000 unique prompt_ids, got {len(all_prompt_ids)}"

    rng = random.Random(SEED)
    shuffled = all_prompt_ids[:]
    rng.shuffle(shuffled)
    train_pool = sorted(shuffled[:N_TRAIN_POOL])
    heldout_pool = sorted(shuffled[N_TRAIN_POOL:])
    assert len(train_pool) == N_TRAIN_POOL
    assert len(heldout_pool) == 3000
    assert not (set(train_pool) & set(heldout_pool))

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "prompt_pool_split.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "n_total": len(all_prompt_ids),
                "n_train_pool": len(train_pool),
                "n_heldout_pool": len(heldout_pool),
                "train_pool": train_pool,
                "heldout_pool": heldout_pool,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"prompt_pool_split.json: {len(train_pool)} train-pool / {len(heldout_pool)} held-out prompt_ids")

    # Within-train-pool split for train_probes.py's holdout gate (80/20, same
    # convention as the main pipeline's group_split via test_frac=0.2).
    train_pool_shuffled = train_pool[:]
    rng2 = random.Random(SEED)
    rng2.shuffle(train_pool_shuffled)
    n_test = max(1, round(len(train_pool_shuffled) * 0.2))
    test_split = sorted(train_pool_shuffled[:n_test])
    train_split = sorted(train_pool_shuffled[n_test:])
    (RUN_DIR / "prompt_split.json").write_text(
        json.dumps({"test_frac": 0.2, "seed": SEED, "train": train_split, "test": test_split}, indent=2),
        encoding="utf-8",
    )
    print(f"prompt_split.json: {len(train_split)} train / {len(test_split)} test (within the 2000-prompt pool)")

    # ~500-prompt held-out sample for cross-cell OOD (step 5), drawn from the
    # 3000 held-out pool, NOT overlapping the 2000-prompt training pool.
    rng3 = random.Random(SEED)
    heldout_shuffled = heldout_pool[:]
    rng3.shuffle(heldout_shuffled)
    cross_cell_sample = sorted(heldout_shuffled[:N_HELDOUT_SAMPLE])
    (RUN_DIR / "cross_cell_holdout_prompt_ids.json").write_text(
        json.dumps({"seed": SEED, "n": len(cross_cell_sample), "prompt_ids": cross_cell_sample}, indent=2),
        encoding="utf-8",
    )
    print(f"cross_cell_holdout_prompt_ids.json: {len(cross_cell_sample)} prompt_ids (subset of the held-out pool)")

    # Per-cell subset generation files (2000-prompt training pool only).
    train_pool_set = set(train_pool)
    gen_dir = RUN_DIR / "generations"
    for slug in CELLS:
        rows = read_jsonl(SRC_DIR / f"{slug}.jsonl")
        subset = [r for r in rows if r["prompt_id"] in train_pool_set]
        n_syc = sum(1 for r in subset if r["polarity"] == "sycophantic")
        n_non = sum(1 for r in subset if r["polarity"] == "non_sycophantic")
        assert n_syc == N_TRAIN_POOL and n_non == N_TRAIN_POOL, (
            f"{slug}: expected {N_TRAIN_POOL}/{N_TRAIN_POOL} syc/non, got {n_syc}/{n_non}"
        )
        write_jsonl(gen_dir / f"{slug}.jsonl", subset)
        print(f"  {slug}: {len(subset)} rows -> {gen_dir / f'{slug}.jsonl'}")

    print(f"\nDone. Run dir: {RUN_DIR}")


if __name__ == "__main__":
    main()
