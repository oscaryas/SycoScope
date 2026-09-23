"""Build strict AITA conflict-group train/validation rows for model-diff analysis.

Only NTA/NTA and NTA/YTA (either orientation) pairs are selected. Narrator
perspectives stay in the same split. The eight feasibility-pilot conflicts are
excluded. Selection is deterministic and mixed orientation is stratified.
"""
import argparse
import json
import random
from pathlib import Path


def read_rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def choose(rows, n_train, n_val, rng):
    rows = list(rows)
    rng.shuffle(rows)
    if len(rows) < n_train + n_val:
        raise ValueError(f"need {n_train+n_val} groups, found {len(rows)}")
    return [("train", row) for row in rows[:n_train]] + [
        ("validation", row) for row in rows[n_train:n_train+n_val]]


def allocate_mixed(mixed_by_orientation, n_train, n_val, rng):
    counts = {key: len(group) for key, group in mixed_by_orientation.items()}
    total = sum(counts.values())
    minor = min(counts, key=counts.get)
    minor_train = round(n_train * counts[minor] / total)
    minor_val = round(n_val * counts[minor] / total)
    major = next(x for x in counts if x != minor)
    result = []
    for orientation, group in mixed_by_orientation.items():
        nt = minor_train if orientation == minor else n_train-minor_train
        nv = minor_val if orientation == minor else n_val-minor_val
        result += choose(group, nt, nv, rng)
    return result


def write_pair(f, split, cell, orientation, row):
    for side, prefix in (("original", "original_post"), ("flipped", "flipped_story")):
        response_id = "__".join(("AITA-NTA-FLIP", str(row["row_id"]), prefix, "0"))
        x = {"response_id": response_id, "pair_id": str(row["row_id"]),
             "split": split, "cell": cell, "orientation": orientation,
             "side": side, "verdict": row[prefix + "_verdict"],
             "prompt": row[prefix + "_prompt"], "response": row[prefix + "_response"]}
        f.write(json.dumps(x, ensure_ascii=False) + "\n")


def build(judged, pilot, out, train_per_cell, val_per_cell, seed, extra_out=None):
    excluded = {x["pair_id"] for x in read_rows(pilot)}
    both = []
    mixed = {"NTA_YTA": [], "YTA_NTA": []}
    for row in read_rows(judged):
        rid = str(row["row_id"])
        if rid in excluded or not all(row.get(prefix + suffix)
                for prefix in ("original_post", "flipped_story")
                for suffix in ("_prompt", "_response")):
            continue
        a, b = row["original_post_verdict"], row["flipped_story_verdict"]
        if a == b == "NTA":
            both.append(row)
        elif (a, b) == ("NTA", "YTA"):
            mixed["NTA_YTA"].append(row)
        elif (a, b) == ("YTA", "NTA"):
            mixed["YTA_NTA"].append(row)
    rng = random.Random(seed)
    selected = [(split, "both_nta", "both_nta", row)
                for split, row in choose(both, train_per_cell, val_per_cell, rng)]
    selected += [(split, "mixed", "_".join((row["original_post_verdict"], row["flipped_story_verdict"])), row)
                 for split, row in allocate_mixed(mixed, train_per_cell, val_per_cell, rng)]
    selected_pairs = {str(row["row_id"]) for _, _, _, row in selected}
    counts = {"both_nta": len(both), **{k: len(v) for k, v in mixed.items()}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("w") as f:
        for split, cell, orientation, row in selected:
            write_pair(f, split, cell, orientation, row)
    if extra_out:
        extra = [("both_nta", "both_nta", row) for row in both]
        extra += [( "mixed", orientation, row)
                  for orientation, group in mixed.items() for row in group]
        extra = [(cell, orientation, row) for cell, orientation, row in extra
                 if str(row["row_id"]) not in selected_pairs]
        Path(extra_out).parent.mkdir(parents=True, exist_ok=True)
        with Path(extra_out).open("w") as f:
            for cell, orientation, row in extra:
                write_pair(f, "extra", cell, orientation, row)
        print({"extra_conflicts": len(extra), "extra_responses": 2 * len(extra)})
    print({"available_after_pilot_exclusion": counts,
           "selected_train_conflicts": 2*train_per_cell,
           "selected_validation_conflicts": 2*val_per_cell,
           "selected_responses": 4*(train_per_cell+val_per_cell)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judged", type=Path, required=True)
    p.add_argument("--exclude-pilot", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-per-cell", type=int, default=240)
    p.add_argument("--val-per-cell", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra-out", type=Path)
    a = p.parse_args()
    build(a.judged, a.exclude_pilot, a.out, a.train_per_cell, a.val_per_cell,
          a.seed, a.extra_out)


if __name__ == "__main__":
    main()
