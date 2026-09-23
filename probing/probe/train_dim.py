#!/usr/bin/env python3
"""Fit difference-in-means (DIM) directions for system-prompt pairs.

This is the non-parametric counterpart to ``train_probes.py``.  It reuses the
same activation caches, paired degeneracy filtering, and shared prompt-level
train/test split.  For each cell, token position, and layer, the direction is

    unit(mean(train activations | label=1) - mean(train activations | label=0)).

The decision threshold is the midpoint of the two projected training means.
All reported metrics are computed on held-out prompts.  Keeping both
polarities of a prompt in the same split is essential: an example-level split
would leak the shared question text across train and test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.utils import common  # noqa: E402
from probing.probe import get_activations as ga  # noqa: E402
from probing.probe.train_probes import (  # noqa: E402
    load_cell,
    load_or_make_split,
    paired_win_rate,
    safe_auc,
    spec_id,
)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("DIM mean difference has zero or non-finite norm")
    return np.asarray(vector, dtype=np.float64) / norm


def cohens_d(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = np.asarray(scores)[labels == 1]
    negative = np.asarray(scores)[labels == 0]
    denominator = len(positive) + len(negative) - 2
    if denominator <= 0:
        return float("nan")
    pooled = np.sqrt(
        ((len(positive) - 1) * positive.var(ddof=1) + (len(negative) - 1) * negative.var(ddof=1))
        / denominator
    )
    if not np.isfinite(pooled) or pooled == 0.0:
        return float("nan")
    return float((positive.mean() - negative.mean()) / pooled)


def fit_holdout_dim(X, y, prompt_ids, split) -> dict | None:
    """Fit DIM on shared training prompts and score shared held-out prompts."""
    train_set, test_set = set(split["train"]), set(split["test"])
    train_mask = np.array([prompt_id in train_set for prompt_id in prompt_ids], dtype=bool)
    test_mask = np.array([prompt_id in test_set for prompt_id in prompt_ids], dtype=bool)
    if (
        train_mask.sum() == 0
        or test_mask.sum() == 0
        or len(np.unique(y[train_mask])) < 2
        or len(np.unique(y[test_mask])) < 2
    ):
        return None

    X_train, y_train = X[train_mask], y[train_mask]
    direction = unit(X_train[y_train == 1].mean(axis=0) - X_train[y_train == 0].mean(axis=0))
    train_scores = X_train @ direction
    negative_mean = float(train_scores[y_train == 0].mean())
    positive_mean = float(train_scores[y_train == 1].mean())
    if positive_mean <= negative_mean:
        raise AssertionError("DIM direction does not point toward label 1")
    threshold = (positive_mean + negative_mean) / 2.0

    test_scores = X[test_mask] @ direction
    test_labels = y[test_mask]
    predictions = (test_scores >= threshold).astype(int)
    test_ids = [prompt_id for prompt_id, keep in zip(prompt_ids, test_mask) if keep]
    win_rate, n_pairs = paired_win_rate(test_scores, test_labels, test_ids)

    all_scores = X @ direction
    return {
        "direction_raw": direction,
        "threshold": threshold,
        "negative_mean": negative_mean,
        "positive_mean": positive_mean,
        "proj_std": float(np.std(all_scores)),
        "test_accuracy": float(accuracy_score(test_labels, predictions)),
        "test_balanced_accuracy": float(balanced_accuracy_score(test_labels, predictions)),
        "test_auc": safe_auc(test_labels, test_scores),
        "test_cohens_d": cohens_d(test_scores, test_labels),
        "test_paired_win_rate": win_rate,
        "n_test_pairs": n_pairs,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


def load_probe_directions(run_dir: Path, slug: str) -> dict[str, np.ndarray]:
    path = run_dir / "probes" / slug / "probes.npz"
    if not path.exists():
        return {}
    with np.load(path) as arrays:
        return {
            name.removesuffix("__direction_raw"): np.asarray(arrays[name], dtype=np.float64)
            for name in arrays.files
            if name.endswith("__direction_raw")
        }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else float("nan")


def rebuild_summary(run_dir: Path) -> dict:
    rows = []
    for path in sorted((run_dir / "dim").glob("*/metrics.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    split_path = run_dir / "prompt_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else {}
    summary = {
        "schema_version": 1,
        "method": "training-split difference in means; midpoint threshold; held-out prompt evaluation",
        "code_version": common.get_code_version(),
        "split": {
            "test_frac": split.get("test_frac"),
            "seed": split.get("seed"),
            "n_train_prompts": len(split.get("train", [])),
            "n_test_prompts": len(split.get("test", [])),
        },
        "n_rows": len(rows),
        "rows": sorted(rows, key=lambda row: (row["spec"], row["position"], row["layer"])),
    }
    output = run_dir / "dim" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    common.add_cells_arg(parser)
    parser.add_argument("--positions", nargs="+", choices=common.POSITIONS, default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-degenerate", action="store_true")
    args = parser.parse_args()

    run_dir = common.resolve_run_dir(args.run_name, create=False)
    meta_path = run_dir / "activations" / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} missing -- run get_activations.py first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    layers = args.layers or meta["layers"]
    positions = args.positions or list(common.POSITIONS)
    split = load_or_make_split(run_dir, args.test_frac, args.seed)
    if split.get("test_frac") != args.test_frac or split.get("seed") != args.seed:
        raise SystemExit(
            "existing prompt_split.json does not match --test-frac/--seed; "
            "DIM must use the probe split for a valid comparison"
        )

    pairs = {pair["slug"]: pair for pair in common.load_prompt_pairs()}
    requested = args.cells or common.all_slugs(include_neutral=False)
    available = [
        slug
        for slug in requested
        if slug != common.NEUTRAL_SLUG and (run_dir / "activations" / f"{slug}.npz").exists()
    ]
    if not available:
        raise SystemExit("no requested cells with activation caches")

    print(
        f"prompt split: {len(split['train'])} train / {len(split['test'])} test prompts; "
        f"{len(available)} cells"
    )
    output_root = run_dir / "dim"
    for slug in available:
        pair = pairs[slug]
        sid = spec_id(pair["type"], slug)
        probe_directions = load_probe_directions(run_dir, slug)
        rows, arrays = [], {}
        print(f"\n[{slug}] spec={sid} type={pair['type']}", flush=True)
        for position in positions:
            for layer in layers:
                X, y, prompt_ids, n_dropped = load_cell(
                    run_dir, slug, position, layer, drop_degenerate=not args.keep_degenerate
                )
                if len(np.unique(y)) < 2:
                    continue
                result = fit_holdout_dim(X, y, prompt_ids, split)
                if result is None:
                    continue
                key = ga.act_key(position, layer)
                probe_direction = probe_directions.get(key)
                direction_cosine = (
                    cosine(result["direction_raw"], probe_direction)
                    if probe_direction is not None
                    else None
                )
                n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
                row = {
                    "spec": sid,
                    "slug": slug,
                    "cell": pair["cell"],
                    "spec_type": pair["type"],
                    "position": position,
                    "layer": layer,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "n_dropped": n_dropped,
                    "n_train": result["n_train"],
                    "n_test": result["n_test"],
                    "holdout_accuracy": result["test_accuracy"],
                    "holdout_balanced_accuracy": result["test_balanced_accuracy"],
                    "holdout_auc": result["test_auc"],
                    "holdout_cohens_d": result["test_cohens_d"],
                    "holdout_paired_win_rate": result["test_paired_win_rate"],
                    "n_test_pairs": result["n_test_pairs"],
                    "logistic_direction_cosine": direction_cosine,
                }
                rows.append(row)
                for field in (
                    "direction_raw",
                    "threshold",
                    "negative_mean",
                    "positive_mean",
                    "proj_std",
                ):
                    arrays[f"{key}__{field}"] = np.asarray(result[field])
                print(
                    f"  {key}: auc={result['test_auc']:.3f} "
                    f"acc={result['test_accuracy']:.3f} d={result['test_cohens_d']:.3f} "
                    f"paired={result['test_paired_win_rate']:.3f}"
                    + (f" probe-cos={direction_cosine:.3f}" if direction_cosine is not None else ""),
                    flush=True,
                )

        cell_dir = output_root / slug
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "metrics.json").write_text(
            json.dumps({"spec": sid, "slug": slug, "rows": rows}, indent=2), encoding="utf-8"
        )
        if arrays:
            np.savez_compressed(cell_dir / "directions.npz", **arrays)

    summary = rebuild_summary(run_dir)
    common.write_run_info(
        run_dir,
        "train_dim",
        args,
        {"cells": available, "n_rows": summary["n_rows"], "output": str(output_root)},
    )
    print(f"\n{summary['n_rows']} DIM settings -> {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
