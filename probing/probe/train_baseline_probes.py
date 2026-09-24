#!/usr/bin/env python3
"""Train group-aware linear probes from reusable activation caches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.utils.probes_cache import load_activation_cache  # noqa: E402
from probing.utils.probes_common import json_dump, parse_dataset_spec  # noqa: E402
from probing.utils.probes_datasets import prepare_cache  # noqa: E402
from probing.probe.baseline_training import (  # noqa: E402
    Bundle, LinearProbe, cross_validate_probe, equalize_and_pool, evaluate_probe,
    reserve_group_holdout,
)


def load_bundle(spec, model: str, seed: int) -> Bundle:
    cache = load_activation_cache(spec.path, expected_model=model)
    arrays, labels, groups, records = prepare_cache(cache, spec.name, seed)
    return Bundle(spec.name, arrays, labels, groups, records)


def feature_keys(array: np.ndarray, component: str):
    if component == "mha":
        for layer in range(array.shape[0]):
            for head in range(array.shape[1]):
                yield f"{layer}:{head}", array[layer, head]
    else:
        for layer in range(array.shape[0]):
            yield str(layer), array[layer]


def train_one(bundle: Bundle, ood: list[Bundle], args, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    checkpoints = {}
    for component in args.components:
        if component not in bundle.arrays:
            raise ValueError(f"training cache lacks requested component {component!r}")
        component_metrics = {}
        component_states = {}
        for offset, (key, features) in enumerate(feature_keys(bundle.arrays[component], component)):
            result = cross_validate_probe(features, bundle.labels, bundle.groups, args, seed_offset=offset * 100)
            state = result.pop("state_dict")
            component_states[key] = state
            result["ood"] = {}
            probe = LinearProbe(features.shape[-1])
            probe.load_state_dict(state)
            for ood_bundle in ood:
                if component not in ood_bundle.arrays:
                    raise ValueError(f"OOD cache {ood_bundle.name} lacks {component}")
                if component == "mha":
                    layer, head = map(int, key.split(":"))
                    ood_features = ood_bundle.arrays[component][layer, head]
                else:
                    ood_features = ood_bundle.arrays[component][int(key)]
                result["ood"][ood_bundle.name] = evaluate_probe(probe, ood_features, ood_bundle.labels)
            component_metrics[key] = result
            print(f"{bundle.name} {component} {key}: CV={result['cv_accuracy_mean']:.3f}")
        best_key = min(
            component_metrics,
            key=lambda key: (-component_metrics[key]["cv_accuracy_mean"], tuple(map(int, key.split(":")))),
        )
        all_metrics[component] = {"best_key": best_key, "layers": component_metrics}
        checkpoints[component] = component_states
        torch.save(component_states, output_dir / f"{component}_probe_weights.pth")
    run_info = {
        "format_version": 1, "model": args.model, "training_name": bundle.name,
        "train_mode": args.train_mode, "components": args.components,
        "cv_folds": args.cv_folds, "balance_method": args.balance_method,
        "regularization": args.regularization, "l2_lambda": args.l2_lambda,
        "steering_holdout_fraction": args.steering_holdout_frac,
        "n_train": len(bundle.labels), "metrics": all_metrics,
    }
    json_dump(output_dir / "metrics.json", run_info)
    return run_info


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
    parser.add_argument("--balance-method", choices=["undersample", "upweight"], default="undersample")
    parser.add_argument("--regularization", choices=["none", "l2"], default="none")
    parser.add_argument("--l2-lambda", type=float)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.components = [value.strip() for value in args.components.split(",") if value.strip()]
    if args.regularization == "l2" and (args.l2_lambda is None or args.l2_lambda <= 0):
        parser.error("--regularization=l2 requires a positive --l2-lambda")
    if args.regularization == "none" and args.l2_lambda is not None:
        parser.error("--l2-lambda is valid only with --regularization=l2")

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
        train_one(equalize_and_pool(train_bundles, args.seed), ood, args, output)
    else:
        for bundle in train_bundles:
            train_one(bundle, ood, args, output / bundle.name)


if __name__ == "__main__":
    main()
