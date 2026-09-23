"""Paired uncertainty and likelihood audit for the 600-conflict GPU pass."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fit_heldout_directions import data_splits, selected_rows, state_matrix


def centroid_scores(X, y, train, val):
    pos = X[train & (y == 1)].mean(0)
    neg = X[train & (y == 0)].mean(0)
    direction = pos - neg
    center = (pos + neg) / 2
    return (X[val] - center) @ (direction / np.linalg.norm(direction))


def paired_auc_delta(y, a, b, seed=2026, draws=5000):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        idx = np.r_[rng.choice(pos, len(pos), replace=True),
                    rng.choice(neg, len(neg), replace=True)]
        deltas.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return {"auc_delta": round(float(roc_auc_score(y, a) - roc_auc_score(y, b)), 4),
            "ci95": [round(float(x), 4) for x in np.quantile(deltas, [0.025, 0.975])]}


def mean_delta(y, values, seed=2026, draws=5000):
    pos, neg = values[y == 1], values[y == 0]
    rng = np.random.default_rng(seed)
    d = [rng.choice(pos, len(pos), replace=True).mean() -
         rng.choice(neg, len(neg), replace=True).mean() for _ in range(draws)]
    return {"both_nta_mean": round(float(pos.mean()), 4),
            "mixed_mean": round(float(neg.mean()), 4),
            "both_minus_mixed": round(float(pos.mean() - neg.mean()), 4),
            "ci95": [round(float(x), 4) for x in np.quantile(d, [0.025, 0.975])]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = selected_rows(args.manifest)
    y, train, val = data_splits(rows)
    all_rows = {}
    for kind in ("base", "chat"):
        all_rows[kind] = {x["response_id"]: x for x in map(
            json.loads, (args.result_dir / (kind + ".jsonl")).read_text().splitlines())}
    if set(all_rows["base"]) != set(all_rows["chat"]):
        raise ValueError("unmatched response IDs")
    for rid in all_rows["base"]:
        if any(all_rows["base"][rid][k] != all_rows["chat"][rid][k]
               for k in ("token_ids_sha256", "response_start", "n_tokens")):
            raise ValueError("token mismatch: " + rid)

    scores = {}
    for kind in ("base", "chat"):
        for key in ("pre_response_state", "early_response_state"):
            X = state_matrix(rows, args.result_dir, kind, key)
            scores[kind + ":" + key] = centroid_scores(X, y, train, val)

    texts = [x["prompt"] for x in rows]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    tr = vec.fit_transform([texts[i] for i in np.flatnonzero(train)])
    va = vec.transform([texts[i] for i in np.flatnonzero(val)])
    scores["prompt_tfidf"] = LogisticRegression(C=1, max_iter=2000).fit(
        tr, y[train]).predict_proba(va)[:, 1]
    truth = y[val]
    comparisons = (
        ("chat_pre_vs_base_pre", "chat:pre_response_state", "base:pre_response_state"),
        ("chat_pre_vs_prompt_text", "chat:pre_response_state", "prompt_tfidf"),
        ("base_early_vs_chat_early", "base:early_response_state", "chat:early_response_state"),
    )
    report = {"validation_n": int(val.sum()),
              "auc": {k: round(float(roc_auc_score(truth, v)), 4) for k, v in scores.items()},
              "paired_auc_deltas": {name: paired_auc_delta(truth, scores[a], scores[b])
                                    for name, a, b in comparisons}}
    forced = np.array([all_rows["chat"][x["response_id"]]["mean_forced_logprob"] -
                       all_rows["base"][x["response_id"]]["mean_forced_logprob"]
                       for x in rows])
    report["chat_minus_base_observed_logprob"] = {
        "train": mean_delta(y[train], forced[train]),
        "validation": mean_delta(truth, forced[val]),
        "limit": "Observed natural responses vary in wording and length; this likelihood contrast is descriptive."
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
