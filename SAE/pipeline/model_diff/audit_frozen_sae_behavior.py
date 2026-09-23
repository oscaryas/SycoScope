"""Local frozen-SAE audit with a group-held-out AITA style baseline.

The SAE already trained unsupervised on these response texts. This is a
downstream feature-association check, not an independent SAE generalization
or causal steering test.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def style_vector(row, n_sentences):
    response = row["response"]
    lines = response.splitlines()
    numbered = sum(bool(re.match(r"^\s*\d+[.)]\s", line)) for line in lines)
    bullets = sum(bool(re.match(r"^\s*[-*]\s", line)) for line in lines)
    opening = response[:180].lower()
    empathetic = any(x in opening for x in (
        "understandable", "i understand", "i'm sorry", "you are not alone",
        "the age-old", "it's natural", "your feelings are valid"))
    explicit_verdict = bool(re.search(r"\b(?:NTA|YTA|ESH|NAH)\b", response, re.I))
    return np.array([np.log1p(len(response)), np.log1p(len(row["prompt"])),
                     np.log1p(n_sentences), numbered / max(len(lines), 1),
                     bullets / max(len(lines), 1), float(empathetic),
                     float(explicit_verdict)], dtype=np.float32)


def response_features(manifest, sentence_path, sae_path):
    rows = read_rows(manifest)
    wanted = {x["response_id"]: x for x in rows}
    sentence_df = pd.read_parquet(sentence_path, columns=["response_id", "global_idx", "text"])
    sentence_df = sentence_df[sentence_df["response_id"].isin(wanted)]
    ids = np.load(sae_path / "assignments_topk.npy", mmap_mode="r")
    strength = np.load(sae_path / "assignments_topk_strength.npy", mmap_mode="r")
    if len(ids) != 833300 or len(strength) != len(ids):
        raise ValueError("stored assignment row-count mismatch")
    latent_count = int(json.loads((sae_path / "config.json").read_text())["n_latents"])
    result = {}
    for rid, group in sentence_df.groupby("response_id", sort=False):
        indices = group["global_idx"].to_numpy(dtype=np.int64)
        assigned = ids[indices]
        values = strength[indices]
        activations = np.zeros((len(indices), latent_count), dtype=np.float32)
        for slot in range(assigned.shape[1]):
            valid = assigned[:, slot] >= 0
            activations[np.flatnonzero(valid), assigned[valid, slot]] = values[valid, slot]
        row = wanted[rid]
        result[rid] = {"feature_mean": activations.mean(axis=0),
                       "feature_fraction": (activations > 0).mean(axis=0),
                       "style": style_vector(row, len(indices)),
                       "n_sentences": len(indices)}
    missing = set(wanted) - set(result)
    if missing:
        raise ValueError(f"{len(missing)} manifest response IDs absent from sentence index")
    return rows, result


def collect(rows, features, side, orientation):
    selected = [x for x in rows if x["side"] == side and x["verdict"] == "NTA" and
                (x["cell"] == "both_nta" or x["orientation"] == orientation)]
    X_sae = np.stack([features[x["response_id"]]["feature_mean"] for x in selected])
    X_style = np.stack([features[x["response_id"]]["style"] for x in selected])
    y = np.array([x["cell"] == "both_nta" for x in selected], dtype=np.int8)
    split = np.array([x["split"] for x in selected])
    return selected, X_sae, X_style, y, split


def evaluate(X, y, split):
    train, val = split == "train", split == "validation"
    if len(set(y[train])) != 2 or len(set(y[val])) != 2:
        raise ValueError("both classes required in train and validation")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    model.fit(X[train], y[train])
    scores = model.predict_proba(X[val])[:, 1]
    return {"train_n": int(train.sum()), "validation_n": int(val.sum()),
            "validation_auc": float(roc_auc_score(y[val], scores)),
            "validation_accuracy": float(((scores >= 0.5) == y[val]).mean()),
            "validation_scores": scores, "validation_truth": y[val]}


def evaluate_extra(X, y, split, X_extra, y_extra):
    train = split == "train"
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    model.fit(X[train], y[train])
    scores = model.predict_proba(X_extra)[:, 1]
    return {"validation_auc": float(roc_auc_score(y_extra, scores)),
            "validation_accuracy": float(((scores >= 0.5) == y_extra).mean()),
            "validation_scores": scores, "validation_truth": y_extra}


def feature_differences(X, y, split, labels):
    train, val = split == "train", split == "validation"
    diffs = []
    for j in range(X.shape[1]):
        pos, neg = X[train & (y == 1), j], X[train & (y == 0), j]
        raw = float(pos.mean() - neg.mean())
        pooled = float(np.sqrt((pos.var() + neg.var()) / 2))
        val_raw = float(X[val & (y == 1), j].mean() - X[val & (y == 0), j].mean())
        diffs.append({"latent_id": j, "label": labels.get(j, ""),
                      "train_mean_difference": round(raw, 4),
                      "train_standardized_difference": round(raw / max(pooled, 1e-6), 4),
                      "validation_mean_difference": round(val_raw, 4),
                      "same_sign_validation": bool(raw * val_raw > 0)})
    return sorted(diffs, key=lambda x: abs(x["train_standardized_difference"]), reverse=True)


def bootstrap_auc_intervals(style_eval, sae_eval, combined_eval, n=1000, seed=13):
    truth = style_eval["validation_truth"]
    positive, negative = np.flatnonzero(truth == 1), np.flatnonzero(truth == 0)
    rng = np.random.default_rng(seed)
    samples = {"style_baseline": [], "frozen_sae": [], "style_plus_sae": [],
               "sae_minus_style": []}
    for _ in range(n):
        indices = np.concatenate((rng.choice(positive, len(positive), replace=True),
                                  rng.choice(negative, len(negative), replace=True)))
        auc = {key: roc_auc_score(truth[indices], result["validation_scores"][indices])
               for key, result in (("style_baseline", style_eval),
                                   ("frozen_sae", sae_eval),
                                   ("style_plus_sae", combined_eval))}
        for key, value in auc.items():
            samples[key].append(value)
        samples["sae_minus_style"].append(auc["frozen_sae"] - auc["style_baseline"])
    return {key: [round(float(x), 4) for x in np.quantile(values, [0.025, 0.975])]
            for key, values in samples.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--sentences", type=Path, required=True)
    p.add_argument("--sae", type=Path, required=True)
    p.add_argument("--extra-manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows, features = response_features(a.manifest, a.sentences, a.sae)
    selected, sae, style, y, split = collect(rows, features, "original", "NTA_YTA")
    labels = {x["latent_id"]: x["title"]
              for x in json.loads((a.sae / "labels.json").read_text())}
    style_eval = evaluate(style, y, split)
    sae_eval = evaluate(sae, y, split)
    combined_eval = evaluate(np.concatenate((style, sae), axis=1), y, split)
    extra_report = None
    if a.extra_manifest:
        extra_rows, extra_features = response_features(
            a.extra_manifest, a.sentences, a.sae)
        _, extra_sae, extra_style, extra_y, _ = collect(
            extra_rows, extra_features, "original", "NTA_YTA")
        extra_style_eval = evaluate_extra(style, y, split, extra_style, extra_y)
        extra_sae_eval = evaluate_extra(sae, y, split, extra_sae, extra_y)
        extra_combined_eval = evaluate_extra(
            np.concatenate((style, sae), axis=1), y, split,
            np.concatenate((extra_style, extra_sae), axis=1), extra_y)
        extra_report = {
            "n": len(extra_y), "both_nta_n": int(extra_y.sum()),
            "mixed_NTA_YTA_n": int((extra_y == 0).sum()),
            "auc": {"style_baseline": round(extra_style_eval["validation_auc"], 4),
                    "frozen_sae": round(extra_sae_eval["validation_auc"], 4),
                    "style_plus_sae": round(extra_combined_eval["validation_auc"], 4)},
            "auc_ci95": bootstrap_auc_intervals(
                extra_style_eval, extra_sae_eval, extra_combined_eval)}
    selected_flip, flip_sae, _, flip_y, flip_split = collect(
        rows, features, "flipped", "YTA_NTA")
    original_differences = feature_differences(sae, y, split, labels)
    flip_differences = {x["latent_id"]: x for x in feature_differences(
        flip_sae, flip_y, flip_split, labels)}
    report = {
        "main_contrast": "Original-side NTA responses: chat NTA/NTA versus NTA/YTA conflict pairs",
        "train_responses": int((split == "train").sum()),
        "validation_responses": int((split == "validation").sum()),
        "train_cell_counts": {c: int(sum((split == "train") & (y == v)))
                              for c, v in (("both_nta", 1), ("mixed", 0))},
        "validation_cell_counts": {c: int(sum((split == "validation") & (y == v)))
                                   for c, v in (("both_nta", 1), ("mixed", 0))},
        "validation_auc": {"style_baseline": round(style_eval["validation_auc"], 4),
                           "frozen_sae": round(sae_eval["validation_auc"], 4),
                           "style_plus_sae": round(combined_eval["validation_auc"], 4)},
        "validation_auc_ci95": bootstrap_auc_intervals(
            style_eval, sae_eval, combined_eval),
        "validation_accuracy": {"style_baseline": round(style_eval["validation_accuracy"], 4),
                                "frozen_sae": round(sae_eval["validation_accuracy"], 4),
                                "style_plus_sae": round(combined_eval["validation_accuracy"], 4)},
        "latent_differences": original_differences,
        "flipped_side_control": {
            "train_n": int((flip_split == "train").sum()),
            "validation_n": int((flip_split == "validation").sum()),
            "mixed_YTA_NTA_train_n": int(sum((flip_split == "train") & (flip_y == 0))),
            "mixed_YTA_NTA_validation_n": int(sum((flip_split == "validation") & (flip_y == 0))),
            "original_top5_latent_effects": [
                {"latent_id": x["latent_id"],
                 "train_mean_difference": flip_differences[x["latent_id"]]["train_mean_difference"],
                 "validation_mean_difference": flip_differences[x["latent_id"]]["validation_mean_difference"]}
                for x in original_differences[:5]]},
        "limit": "SAE was trained unsupervised on these exact texts; response-style and story-content confounds remain."
    }
    if extra_report:
        report["unused_conflict_holdout"] = extra_report
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print("n", report["train_responses"], report["validation_responses"])
    print("validation AUC", report["validation_auc"])
    if extra_report:
        print("unused holdout AUC", extra_report["auc"])
    print("top latents", [(x["latent_id"], x["label"], x["train_standardized_difference"],
                           x["validation_mean_difference"]) for x in report["latent_differences"][:5]])


if __name__ == "__main__":
    main()
