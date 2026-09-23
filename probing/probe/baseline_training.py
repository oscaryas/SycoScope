from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from probing.utils.baseline_probes_common import t_confidence_interval


@dataclass
class Bundle:
    name: str
    arrays: dict[str, np.ndarray]
    labels: np.ndarray
    groups: np.ndarray
    records: list[dict]


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, values):
        return self.linear(values).squeeze(-1)


def reserve_group_holdout(bundle: Bundle, fraction: float, seed: int) -> tuple[Bundle, Bundle]:
    if fraction == 0:
        empty = np.asarray([], dtype=int)
        return bundle, subset_bundle(bundle, empty, f"{bundle.name}_holdout")
    if not 0.1 <= fraction <= 0.2:
        raise ValueError("steering holdout fraction must be between 0.10 and 0.20")
    unique_groups, first = np.unique(bundle.groups, return_index=True)
    group_labels = bundle.labels[first]
    train_groups, holdout_groups = train_test_split(
        unique_groups, test_size=fraction, random_state=seed, stratify=group_labels,
    )
    train_idx = np.nonzero(np.isin(bundle.groups, train_groups))[0]
    holdout_idx = np.nonzero(np.isin(bundle.groups, holdout_groups))[0]
    return subset_bundle(bundle, train_idx, bundle.name), subset_bundle(
        bundle, holdout_idx, f"{bundle.name}_holdout"
    )


def subset_bundle(bundle: Bundle, indices: np.ndarray, name: str | None = None) -> Bundle:
    arrays = {}
    for component, array in bundle.arrays.items():
        axis = 2 if component == "mha" else 1
        arrays[component] = np.take(array, indices, axis=axis)
    return Bundle(
        name or bundle.name, arrays, bundle.labels[indices], bundle.groups[indices],
        [bundle.records[int(i)] for i in indices],
    )


def balanced_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    classes = np.unique(labels)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError(f"binary labels 0/1 required, found {classes.tolist()}")
    rng = np.random.default_rng(seed)
    by_class = [np.nonzero(labels == value)[0] for value in (0, 1)]
    n = min(map(len, by_class))
    selected = np.concatenate([rng.choice(indices, n, replace=False) for indices in by_class])
    rng.shuffle(selected)
    return selected


def equalize_and_pool(bundles: list[Bundle], seed: int) -> Bundle:
    balanced = [subset_bundle(bundle, balanced_indices(bundle.labels, seed + i)) for i, bundle in enumerate(bundles)]
    n = min(len(bundle.labels) for bundle in balanced)
    rng = np.random.default_rng(seed)
    equalized = []
    for bundle in balanced:
        indices = np.arange(len(bundle.labels))
        if len(indices) > n:
            indices = rng.choice(indices, n, replace=False)
        equalized.append(subset_bundle(bundle, np.asarray(indices)))
    components = set(equalized[0].arrays)
    if any(set(bundle.arrays) != components for bundle in equalized):
        raise ValueError("all pooled caches must contain the same components")
    arrays = {}
    for component in components:
        axis = 2 if component == "mha" else 1
        arrays[component] = np.concatenate([bundle.arrays[component] for bundle in equalized], axis=axis)
    return Bundle(
        "pooled", arrays, np.concatenate([b.labels for b in equalized]),
        np.concatenate([b.groups for b in equalized]), sum((b.records for b in equalized), []),
    )


def fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    regularization: str,
    l2_lambda: float | None,
    balance_method: str,
    seed: int,
) -> LinearProbe:
    torch.manual_seed(seed)
    x = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    probe = LinearProbe(x.shape[1])
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    if balance_method == "upweight":
        positive = float((y == 1).sum())
        negative = float((y == 0).sum())
        pos_weight = torch.tensor(negative / positive) if positive and negative else torch.tensor(1.0)
    else:
        pos_weight = torch.tensor(1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        permutation = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            index = permutation[start:start + batch_size]
            loss = criterion(probe(x[index]), y[index])
            if regularization == "l2":
                loss = loss + float(l2_lambda) * probe.linear.weight.pow(2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    probe.eval()
    return probe


def scores(probe: LinearProbe, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return probe(torch.as_tensor(np.asarray(features), dtype=torch.float32)).numpy()


def validate_groups(labels: np.ndarray, groups: np.ndarray, folds: int) -> None:
    group_to_label = {}
    for label, group in zip(labels, groups):
        previous = group_to_label.setdefault(str(group), int(label))
        if previous != int(label):
            raise ValueError(f"source group {group!r} contains conflicting labels")
    counts = {value: sum(label == value for label in group_to_label.values()) for value in (0, 1)}
    if min(counts.values()) < folds:
        raise ValueError(f"{folds}-fold grouped CV requires at least {folds} groups per class; found {counts}")


def cross_validate_probe(features, labels, groups, args, seed_offset=0) -> dict:
    validate_groups(labels, groups, args.cv_folds)
    splitter = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    fold_accuracy, fold_balanced, fold_auc = [], [], []
    out_of_fold = np.full(len(labels), np.nan, dtype=float)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(features, labels, groups)):
        if args.balance_method == "undersample":
            keep = balanced_indices(labels[train_idx], args.seed + fold)
            train_idx = train_idx[keep]
        probe = fit_probe(
            features[train_idx], labels[train_idx], args.epochs, args.batch_size,
            args.learning_rate, args.regularization, args.l2_lambda,
            args.balance_method, args.seed + seed_offset + fold,
        )
        logits = scores(probe, features[test_idx])
        predictions = (logits > 0).astype(int)
        out_of_fold[test_idx] = logits
        fold_accuracy.append(float(accuracy_score(labels[test_idx], predictions)))
        fold_balanced.append(float(balanced_accuracy_score(labels[test_idx], predictions)))
        fold_auc.append(float(roc_auc_score(labels[test_idx], logits)))
    mean = float(np.mean(fold_accuracy))
    ci = t_confidence_interval(fold_accuracy)
    final_idx = np.arange(len(labels))
    if args.balance_method == "undersample":
        final_idx = balanced_indices(labels, args.seed)
    final_probe = fit_probe(
        features[final_idx], labels[final_idx], args.epochs, args.batch_size,
        args.learning_rate, args.regularization, args.l2_lambda,
        args.balance_method, args.seed + seed_offset + 1000,
    )
    direction = final_probe.linear.weight.detach().numpy()[0]
    unit = direction / (np.linalg.norm(direction) + 1e-8)
    projection_std = float(np.std(features @ unit))
    return {
        "cv_accuracy_mean": mean, "cv_accuracy_ci": list(ci),
        "fold_accuracy": fold_accuracy, "fold_balanced_accuracy": fold_balanced,
        "fold_auc": fold_auc, "oof_logits": out_of_fold.tolist(),
        "state_dict": final_probe.state_dict(), "projection_std": projection_std,
    }


def evaluate_probe(probe: LinearProbe, features: np.ndarray, labels: np.ndarray) -> dict:
    logits = scores(probe, features)
    predictions = (logits > 0).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, logits)),
        "n": int(len(labels)), "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
    }
