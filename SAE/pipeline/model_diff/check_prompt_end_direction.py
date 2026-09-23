"""Leave-one-conflict-out feasibility check for a prompt-end category direction.

Uses pair-averaged states from both narrator perspectives. No SAE selection,
causal steering, or generalization claim follows from eight pilot conflicts.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_pairs(manifest, result_dir, kind, key):
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines()]
    grouped = {}
    for row in rows:
        rid = row["response_id"]
        filename = __import__("hashlib").sha256(rid.encode()).hexdigest() + ".npz"
        path = Path(result_dir) / (kind + "_states") / filename
        state = np.load(path)[key].astype(np.float32)
        x = grouped.setdefault(row["pair_id"], {"cell": row["cell"], "states": []})
        if x["cell"] != row["cell"]:
            raise ValueError("conflict cell mismatch")
        x["states"].append(state)
    pairs = []
    for pair_id, x in grouped.items():
        if len(x["states"]) != 2:
            raise ValueError("incomplete narrator pair: " + pair_id)
        pairs.append((pair_id, x["cell"], np.mean(x["states"], axis=0)))
    return pairs


def loo(pairs):
    results = []
    for i, (rid, truth, vec) in enumerate(pairs):
        train = pairs[:i] + pairs[i + 1:]
        pos = np.mean([v for _, c, v in train if c == "both_nta"], axis=0)
        neg = np.mean([v for _, c, v in train if c == "mixed"], axis=0)
        direction = pos - neg
        # Equal-centroid linear discriminant, trained without the held-out pair.
        threshold = np.dot(direction, (pos + neg) / 2)
        score = float(np.dot(direction, vec) - threshold)
        pred = "both_nta" if score >= 0 else "mixed"
        results.append({"pair_id": rid, "truth": truth, "prediction": pred,
                        "correct": pred == truth, "signed_score": round(score, 3)})
    return {"n_pairs": len(results), "correct": sum(x["correct"] for x in results),
            "rows": results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    report = {}
    for kind in ("chat", "base"):
        report[kind] = {}
        for key in ("pre_response_state", "early_response_state"):
            report[kind][key] = loo(load_pairs(a.manifest, a.result_dir, kind, key))
    report["limit"] = (
        "Eight conflicts, four per cell, used in unsupervised chat SAE training. "
        "A pair-level direction can encode story content or prompt format; "
        "this is not a causal sycophancy gate.")
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    for kind, tests in report.items():
        if kind == "limit":
            continue
        print(kind, {key: f"{x['correct']}/{x['n_pairs']}" for key, x in tests.items()})


if __name__ == "__main__":
    main()
