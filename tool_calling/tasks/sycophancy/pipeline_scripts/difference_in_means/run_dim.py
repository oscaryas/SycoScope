#!/usr/bin/env python3
"""Compute grouped-CV difference-in-means directions from activation caches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SYCOPHANCY_DIR = Path(__file__).resolve().parents[2]
if str(SYCOPHANCY_DIR) not in sys.path:
    sys.path.insert(0, str(SYCOPHANCY_DIR))

from pipeline_scripts.cache import load_activation_cache  # noqa: E402
from pipeline_scripts.common import json_dump, parse_dataset_spec, t_confidence_interval, t_interval  # noqa: E402
from pipeline_scripts.datasets import prepare_cache  # noqa: E402
from pipeline_scripts.training import (  # noqa: E402
    Bundle, equalize_and_pool, reserve_group_holdout, validate_groups,
)


def load_bundle(spec, model: str, seed: int) -> Bundle:
    cache = load_activation_cache(spec.path, expected_model=model)
    arrays, labels, groups, records = prepare_cache(cache, spec.name, seed)
    return Bundle(spec.name, arrays, labels, groups, records)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + 1e-8)


def _direction(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return _unit(features[labels == 1].mean(0) - features[labels == 0].mean(0))


def _midpoint(projections: np.ndarray, labels: np.ndarray) -> float:
    return float((projections[labels == 1].mean() + projections[labels == 0].mean()) / 2)


def _cohens_d(projections: np.ndarray, labels: np.ndarray) -> float:
    positive, negative = projections[labels == 1], projections[labels == 0]
    denominator = len(positive) + len(negative) - 2
    if denominator <= 0:
        return 0.0
    pooled = np.sqrt(
        ((len(positive) - 1) * positive.var(ddof=1) + (len(negative) - 1) * negative.var(ddof=1))
        / denominator
    )
    return float((positive.mean() - negative.mean()) / (pooled + 1e-8))


def compute_layer(features: np.ndarray, labels: np.ndarray, groups: np.ndarray, folds: int, seed: int) -> dict:
    validate_groups(labels, groups, folds)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_accuracy, fold_balanced, fold_auc, fold_d = [], [], [], []
    for train_idx, test_idx in splitter.split(features, labels, groups):
        direction = _direction(features[train_idx], labels[train_idx])
        train_projection = features[train_idx] @ direction
        threshold = _midpoint(train_projection, labels[train_idx])
        test_projection = features[test_idx] @ direction
        prediction = (test_projection >= threshold).astype(int)
        fold_accuracy.append(float(accuracy_score(labels[test_idx], prediction)))
        fold_balanced.append(float(balanced_accuracy_score(labels[test_idx], prediction)))
        fold_auc.append(float(roc_auc_score(labels[test_idx], test_projection)))
        fold_d.append(_cohens_d(test_projection, labels[test_idx]))
    direction = _direction(features, labels)
    projection = features @ direction
    threshold = _midpoint(projection, labels)
    return {
        "cv_accuracy_mean": float(np.mean(fold_accuracy)),
        "cv_accuracy_ci": list(t_confidence_interval(fold_accuracy)),
        "cv_balanced_accuracy_mean": float(np.mean(fold_balanced)),
        "cv_balanced_accuracy_ci": list(t_confidence_interval(fold_balanced)),
        "cv_auc_mean": float(np.mean(fold_auc)),
        "cv_auc_ci": list(t_confidence_interval(fold_auc)),
        "cv_cohens_d_mean": float(np.mean(fold_d)),
        "cv_cohens_d_ci": list(t_interval(fold_d)),
        "fold_accuracy": fold_accuracy, "fold_balanced_accuracy": fold_balanced,
        "fold_auc": fold_auc, "fold_cohens_d": fold_d,
        "direction": direction, "threshold": threshold,
        "projection_std": float(np.std(projection)),
    }


def evaluate_ood(features, labels, direction, threshold) -> dict:
    projection = features @ direction
    prediction = (projection >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "roc_auc": float(roc_auc_score(labels, projection)),
        "cohens_d": _cohens_d(projection, labels),
        "n": int(len(labels)), "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
    }


def feature_keys(array: np.ndarray, component: str):
    if component == "mha":
        for layer in range(array.shape[0]):
            for head in range(array.shape[1]):
                yield f"{layer}:{head}", array[layer, head]
    else:
        for layer in range(array.shape[0]):
            yield str(layer), array[layer]


def run_one(bundle: Bundle, ood: list[Bundle], args, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    vectors = {}
    for component in args.components:
        component_metrics, component_vectors = {}, {}
        for key, features in feature_keys(bundle.arrays[component], component):
            result = compute_layer(features, bundle.labels, bundle.groups, args.cv_folds, args.seed)
            direction = result.pop("direction")
            result["ood"] = {}
            for target in ood:
                if component == "mha":
                    layer, head = map(int, key.split(":"))
                    target_features = target.arrays[component][layer, head]
                else:
                    target_features = target.arrays[component][int(key)]
                result["ood"][target.name] = evaluate_ood(
                    target_features, target.labels, direction, result["threshold"]
                )
            component_metrics[key] = result
            component_vectors[key] = torch.from_numpy(direction).float() * result["projection_std"]
            print(f"{bundle.name} {component} {key}: CV={result['cv_accuracy_mean']:.3f} d={result['cv_cohens_d_mean']:.3f}")
        metrics[component] = component_metrics
        vectors[component] = component_vectors
        torch.save(component_vectors, output_dir / f"{component}_dim_vectors.pt")
    result = {
        "format_version": 1, "model": args.model, "training_name": bundle.name,
        "train_mode": args.train_mode, "components": args.components,
        "cv_folds": args.cv_folds, "n_train": len(bundle.labels), "metrics": metrics,
    }
    if args.probe_results:
        probe_dir = Path(args.probe_results)
        if not (probe_dir / "metrics.json").exists() and (probe_dir / bundle.name / "metrics.json").exists():
            probe_dir = probe_dir / bundle.name
        probe = json.loads((probe_dir / "metrics.json").read_text(encoding="utf-8"))
        if probe.get("model") != args.model:
            raise ValueError("probe results model does not match DIM model")
        selected = probe["metrics"]["residual"]["best_key"]
        result["probe_selected_residual_layer"] = int(selected)
    json_dump(output_dir / "metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--ood-dataset", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--train-mode", choices=["pooled", "separate"], default="pooled")
    parser.add_argument("--model", required=True)
    parser.add_argument("--components", default="residual")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--steering-holdout-frac", type=float, default=0.2)
    parser.add_argument("--probe-results")
    parser.add_argument("--steering-eval", action="store_true")
    parser.add_argument("--baseline-generation", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--alphas", default="-10,-5,-1,1,5,10")
    parser.add_argument("--allow-unverified-baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.components = [value.strip() for value in args.components.split(",") if value.strip()]
    if args.steering_eval and not args.probe_results:
        parser.error("--steering-eval requires --probe-results")
    if args.steering_eval and not args.baseline_generation:
        parser.error("--steering-eval requires at least one --baseline-generation NAME=PATH")
    alphas = [float(value) for value in args.alphas.split(",") if value.strip()]
    if 0.0 in alphas:
        parser.error("alpha 0 is not generated; pass saved unsteered generations as the baseline")

    training = [load_bundle(parse_dataset_spec(value), args.model, args.seed) for value in args.train_dataset]
    ood = [load_bundle(parse_dataset_spec(value), args.model, args.seed) for value in args.ood_dataset]
    train_bundles, holdouts = [], []
    for bundle in training:
        train, holdout = reserve_group_holdout(bundle, args.steering_holdout_frac, args.seed)
        train_bundles.append(train)
        holdouts.append({"name": holdout.name, "groups": holdout.groups.tolist()})
    output = Path(args.output_dir)
    json_dump(output / "steering_holdout.json", {"fraction": args.steering_holdout_frac, "datasets": holdouts})
    if args.train_mode == "pooled":
        run_one(equalize_and_pool(train_bundles, args.seed), ood, args, output)
        if args.steering_eval:
            from pipeline_scripts.difference_in_means.steering_eval import run_steering_evaluation
            run_steering_evaluation(args, alphas, output)
    else:
        for bundle, holdout in zip(train_bundles, holdouts):
            child = output / bundle.name
            json_dump(child / "steering_holdout.json", {
                "fraction": args.steering_holdout_frac, "datasets": [holdout],
            })
            run_one(bundle, ood, args, child)
            if args.steering_eval:
                from pipeline_scripts.difference_in_means.steering_eval import run_steering_evaluation
                run_steering_evaluation(args, alphas, child)


if __name__ == "__main__":
    main()
