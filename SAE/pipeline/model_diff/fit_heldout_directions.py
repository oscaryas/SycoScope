"""Group-held-out prompt-end direction audit for paired Llama-3 states.

The main contrast fixes narrator side and observed verdict: original-side
NTA responses from NTA/NTA versus NTA/YTA conflicts. Centroid directions
are fitted on training conflicts only, then evaluated on validation groups.
Prompt-text and prompt-length baselines check easy story/format cues.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def selected_rows(manifest):
    rows = [x for x in read_rows(manifest) if x["side"] == "original" and
            x["verdict"] == "NTA" and (x["cell"] == "both_nta" or
                                        x["orientation"] == "NTA_YTA")]
    pairs = [x["pair_id"] for x in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("same conflict group selected twice")
    return rows


def state_matrix(rows, result_dir, kind, key):
    folder = Path(result_dir) / (kind + "_states")
    result = []
    for row in rows:
        name = hashlib.sha256(row["response_id"].encode()).hexdigest() + ".npz"
        with np.load(folder / name) as archive:
            result.append(archive[key].astype(np.float32))
    return np.stack(result)


def data_splits(rows):
    y = np.array([x["cell"] == "both_nta" for x in rows], dtype=np.int8)
    train = np.array([x["split"] == "train" for x in rows])
    val = np.array([x["split"] == "validation" for x in rows])
    if (train.sum() + val.sum() != len(rows) or
            len(set(y[train])) != 2 or len(set(y[val])) != 2):
        raise ValueError("expected disjoint train/validation with both cells")
    return y, train, val


def auc_ci(y, scores, seed=13, n=1000):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = np.concatenate((rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)))
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    return [round(float(x), 4) for x in np.quantile(aucs, [0.025, 0.975])]


def centroid_direction(X, y, train, val):
    pos = X[train & (y == 1)].mean(0)
    neg = X[train & (y == 0)].mean(0)
    direction = pos - neg
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        raise ValueError("zero direction")
    center = (pos + neg) / 2
    score = (X[val] - center) @ (direction / norm)
    truth = y[val]
    return {"validation_auc": round(float(roc_auc_score(truth, score)), 4),
            "validation_auc_ci95": auc_ci(truth, score),
            "validation_accuracy": round(float(((score >= 0) == truth).mean()), 4),
            "direction_norm": round(norm, 4)}


def text_baselines(rows, y, train, val):
    texts = [x["prompt"] for x in rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    X_train = vectorizer.fit_transform([texts[i] for i in np.flatnonzero(train)])
    X_val = vectorizer.transform([texts[i] for i in np.flatnonzero(val)])
    lexical = LogisticRegression(C=1.0, max_iter=2000).fit(X_train, y[train])
    lexical_score = lexical.predict_proba(X_val)[:, 1]
    lengths = np.array([[np.log1p(len(x["prompt"])),
                         np.log1p(len(x["prompt"].split()))] for x in rows])
    length = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    length.fit(lengths[train], y[train])
    length_score = length.predict_proba(lengths[val])[:, 1]
    return {
        "prompt_tfidf": {"validation_auc": round(float(roc_auc_score(y[val], lexical_score)), 4),
                         "validation_auc_ci95": auc_ci(y[val], lexical_score)},
        "prompt_length": {"validation_auc": round(float(roc_auc_score(y[val], length_score)), 4),
                          "validation_auc_ci95": auc_ci(y[val], length_score)}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--result-dir", type=Path)
    p.add_argument("--baselines-only", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = selected_rows(a.manifest)
    y, train, val = data_splits(rows)
    report = {"main_contrast": "Original-side NTA, NTA/NTA versus NTA/YTA pairs",
              "train_n": int(train.sum()), "validation_n": int(val.sum()),
              "train_cells": {"both_nta": int(sum(train & (y == 1))),
                              "mixed": int(sum(train & (y == 0)))},
              "validation_cells": {"both_nta": int(sum(val & (y == 1))),
                                   "mixed": int(sum(val & (y == 0)))},
              "baselines": text_baselines(rows, y, train, val)}
    if not a.baselines_only:
        if not a.result_dir:
            raise ValueError("--result-dir required unless --baselines-only")
        for kind in ("base", "chat"):
            report[kind] = {key: centroid_direction(
                state_matrix(rows, a.result_dir, kind, key), y, train, val)
                for key in ("pre_response_state", "early_response_state")}
    report["limit"] = ("Teacher forcing uses the chat template for both models; "
                       "prompt-end directions can encode story text. No causal "
                       "sycophancy claim follows from held-out AUC alone.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print("baselines", {k: v["validation_auc"] for k, v in report["baselines"].items()})
    if not a.baselines_only:
        print("state AUC", {kind: {key: v["validation_auc"] for key, v in report[kind].items()}
                            for kind in ("base", "chat")})


if __name__ == "__main__":
    main()
