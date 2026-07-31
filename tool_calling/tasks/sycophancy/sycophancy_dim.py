"""
Shared difference-in-means (DIM) direction-finding module, ported verbatim from
moral_sycophancy_dim_rowid_pooled_colab_standalone.ipynb (cells 17, 20, 23, 33) so
aita_dim_pipeline.py doesn't have to re-derive already-validated math from scratch,
and so any future DIM-based pipeline (e.g. a social-sycophancy-sourced one) can reuse
the same core.

This module intentionally mirrors sycophancy_probes.py's shape (train_*_dim returning
per-key results, analogous to train_*_probes) rather than the more abstracted
single-function sketch this was originally planned around -- the notebook's actual
code is already-validated and ported faithfully rather than restructured, per this
session's own porting rule. The one real difference from sycophancy_probes.py: DIM's
state dict stores one full compute_dim_direction() result per key (direction,
effect_size, fold_effect_sizes, auc_roc, fold_aucs, proj_std all together), since DIM
has no separate accuracy/CI split to track the way a trained probe does.

collect_activations/_pool are NOT re-defined here -- they're identical to
sycophancy_probes.py's existing versions, so callers should import from there.
load_dim_vectors is also not re-defined -- Task 1's load_direction_vectors(...,
fmt="dim") in sycophancy_steering.py already covers it.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Core DIM computation
# ---------------------------------------------------------------------------

def _stratified_folds(y, n_folds, rng):
    """Assign each example to one of n_folds folds, preserving class balance per fold."""
    fold_of = np.empty(len(y), dtype=int)
    for cls in np.unique(y):
        cls_idx = np.nonzero(y == cls)[0]
        rng.shuffle(cls_idx)
        fold_of[cls_idx] = np.arange(len(cls_idx)) % n_folds
    return fold_of


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def _cohens_d(X, y, direction):
    """Cohen's d between the label=1 and label=0 groups, projected onto unit `direction`."""
    proj = X @ direction
    pos, neg = proj[y == 1], proj[y == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos > 1 and n_neg > 1:
        pooled_std = np.sqrt(
            ((n_pos - 1) * pos.var(ddof=1) + (n_neg - 1) * neg.var(ddof=1)) / (n_pos + n_neg - 2)
        )
    else:
        pooled_std = proj.std(ddof=0)
    return float((pos.mean() - neg.mean()) / (pooled_std + 1e-8))


def _safe_auc(y_true, scores):
    """roc_auc_score, but degrades to 0.5 (chance) instead of raising when only one
    class is present in y_true -- can happen on a small held-out CV fold."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, scores))


def compute_dim_direction(X, y, method="cv_averaged", n_folds=5, seed=None):
    """
    Difference-in-means direction between label=1 and label=0 activations.

    method="naive": direction = unit(mean(X[y==1]) - mean(X[y==0])), effect size is
    Cohen's d evaluated on the same data the direction was computed from -- optimistic,
    since the direction is fit and scored on the same examples.

    method="cv_averaged" (default): n_folds-fold stratified CV. Each fold fits a
    direction on the training split and scores Cohen's d / AUC-ROC on the held-out
    test split only, then the final direction is the (re-unit-normalized) mean of the
    n_folds fold directions -- an out-of-sample effect-size/AUC estimate, and a
    direction less sensitive to any single fold's noise.

    AUC-ROC here is diagnostic-only -- direction *selection* between candidate
    (layer, head) keys always uses effect_size (Cohen's d), never auc_roc. AUC is
    reported alongside because it's a more familiar separability metric to sanity-check
    effect_size against, and because bucket_cross_results / metadata want it recorded.

    Returns dict: direction (unit np.ndarray), effect_size (float), fold_effect_sizes
    (list or None, cv_averaged only), auc_roc (float), fold_aucs (list or None,
    cv_averaged only), input_dim (int), proj_std (float -- std of X @ direction over
    ALL of X, used downstream to scale the direction into an alpha-ready steering
    vector).
    """
    input_dim = X.shape[-1]
    if method == "naive":
        direction = _unit(X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0))
        effect_size = _cohens_d(X, y, direction)
        fold_effect_sizes = None
        auc_roc = _safe_auc(y, X @ direction)
        fold_aucs = None
    elif method == "cv_averaged":
        rng = np.random.default_rng(seed)
        fold_of = _stratified_folds(y, n_folds, rng)
        fold_directions, fold_effect_sizes, fold_aucs = [], [], []
        for fold in range(n_folds):
            test_mask = fold_of == fold
            train_mask = ~test_mask
            if test_mask.sum() == 0 or train_mask.sum() == 0:
                continue
            Xtr, ytr = X[train_mask], y[train_mask]
            d_f = _unit(Xtr[ytr == 1].mean(axis=0) - Xtr[ytr == 0].mean(axis=0))
            fold_directions.append(d_f)
            fold_effect_sizes.append(_cohens_d(X[test_mask], y[test_mask], d_f))
            fold_aucs.append(_safe_auc(y[test_mask], X[test_mask] @ d_f))
        direction = _unit(np.mean(fold_directions, axis=0))
        effect_size = float(np.mean(fold_effect_sizes)) if fold_effect_sizes else 0.0
        auc_roc = float(np.mean(fold_aucs)) if fold_aucs else 0.5
    else:
        raise ValueError(f"method must be 'naive' or 'cv_averaged', got {method!r}")
    proj_std = float(np.std(X @ direction))
    return {
        "direction": direction, "effect_size": effect_size, "fold_effect_sizes": fold_effect_sizes,
        "auc_roc": auc_roc, "fold_aucs": fold_aucs, "input_dim": input_dim, "proj_std": proj_std,
    }


def train_mha_dim(mha_activations, labels, n_layers, n_heads, **dim_kwargs):
    """Returns (effect_size_dict, state_dict) keyed by (layer, head). state_dict[(l,h)]
    is the FULL result dict from compute_dim_direction (direction, effect_size,
    fold_effect_sizes, auc_roc, fold_aucs, proj_std)."""
    effect_size_dict, state_dict = {}, {}
    for layer in range(n_layers):
        for head in range(n_heads):
            X = mha_activations[layer, head]
            result = compute_dim_direction(X, labels, **dim_kwargs)
            effect_size_dict[(layer, head)] = result["effect_size"]
            state_dict[(layer, head)] = result
            print(f"  MHA layer={layer} head={head}: cohen_d={result['effect_size']:.3f} auc={result['auc_roc']:.3f}")
    return effect_size_dict, state_dict


def train_mlp_dim(mlp_activations, labels, n_layers, **dim_kwargs):
    """Returns (effect_size_dict, state_dict) keyed by layer. Same per-key result
    shape as train_mha_dim, just without the head axis."""
    effect_size_dict, state_dict = {}, {}
    for layer in range(n_layers):
        X = mlp_activations[layer]
        result = compute_dim_direction(X, labels, **dim_kwargs)
        effect_size_dict[layer] = result["effect_size"]
        state_dict[layer] = result
        print(f"  MLP layer={layer}: cohen_d={result['effect_size']:.3f} auc={result['auc_roc']:.3f}")
    return effect_size_dict, state_dict


def train_residual_dim(residual_activations, labels, n_layers, **dim_kwargs):
    """Returns (effect_size_dict, state_dict) keyed by layer. Same per-key result
    shape as train_mha_dim, just without the head axis."""
    effect_size_dict, state_dict = {}, {}
    for layer in range(n_layers):
        X = residual_activations[layer]
        result = compute_dim_direction(X, labels, **dim_kwargs)
        effect_size_dict[layer] = result["effect_size"]
        state_dict[layer] = result
        print(f"  Residual layer={layer}: cohen_d={result['effect_size']:.3f} auc={result['auc_roc']:.3f}")
    return effect_size_dict, state_dict


# ---------------------------------------------------------------------------
# Row-id-pooled activation collection
# ---------------------------------------------------------------------------

def iter_flip_pairs_all_samples(input_path):
    """
    Like moral_sycophancy_judge.iter_flip_pairs, but returns ALL samples for each side
    instead of just sample_idx=0 -- {row_id: {"original_post": [rec_s0, rec_s1, ...],
    "flipped_story": [rec_s0, rec_s1, ...]}}, only for row_ids where both sides have at
    least one sample. Samples within a side are sorted by sample_idx for a
    deterministic order -- used by the row_id-pooled activation-collection step, NOT by
    judging/labeling (which still only judges sample_idx=0 via iter_flip_pairs,
    unchanged).
    """
    by_row = defaultdict(lambda: defaultdict(list))
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_row[rec["row_id"]][rec["prompt_col"]].append(rec)
    pairs = {}
    for row_id, sides in by_row.items():
        if "original_post" in sides and "flipped_story" in sides:
            pairs[row_id] = {
                "original_post": sorted(sides["original_post"], key=lambda r: r["sample_idx"]),
                "flipped_story": sorted(sides["flipped_story"], key=lambda r: r["sample_idx"]),
            }
    return pairs


def iter_dataset_records_all_samples(input_path):
    """
    Like social_sycophancy_judge.iter_dataset_records, but returns ALL samples per
    row_id instead of just sample_idx=0 -- {row_id: [rec_s0, rec_s1, ...]}, sorted by
    sample_idx. Used by the row_id-pooled activation-collection step for a
    social-sycophancy-sourced label set, NOT by judging/labeling (unchanged).
    """
    by_row = defaultdict(list)
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_row[rec["row_id"]].append(rec)
    return {row_id: sorted(recs, key=lambda r: r["sample_idx"]) for row_id, recs in by_row.items()}


def _average_blocks(arr, sizes, examples_axis):
    """arr has an 'examples' axis (length sum(sizes)) at position `examples_axis` --
    average each consecutive block of that axis down to one vector, preserving every
    other axis. Used to pool multiple generation samples/sides for one row_id into a
    single activation vector before DIM training."""
    arr = np.moveaxis(arr, examples_axis, -2)
    blocks = []
    start = 0
    for size in sizes:
        blocks.append(arr[..., start : start + size, :].mean(axis=-2))
        start += size
    stacked = np.stack(blocks, axis=-2)
    return np.moveaxis(stacked, -2, examples_axis)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_dim_results(output_dir, mha_es, mha_states, mlp_es, mlp_states, res_es, res_states):
    """
    Mirrors the notebook's cell 48 exactly: per-component {component}_dim_vectors.pt
    (alpha-ready direction*proj_std tensors, key=(layer,head) for mha / layer for
    mlp/residual) and {component}_dim_effect_size.pkl (effect_size/fold_effect_sizes/
    auc_roc/fold_aucs only, direction/proj_std/input_dim dropped since the vectors
    file already has the alpha-ready form).

    Returns {"mha_best_key", "mlp_best_key", "residual_best_key"} -- the
    max(abs(effect_size)) key per component, since best-direction selection is always
    by |Cohen's d|, never by raw signed value or by AUC.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import pickle

    with open(output_dir / "mha_effect_size.pkl", "wb") as f:
        pickle.dump(mha_es, f)
    with open(output_dir / "mlp_effect_size.pkl", "wb") as f:
        pickle.dump(mlp_es, f)
    with open(output_dir / "residual_effect_size.pkl", "wb") as f:
        pickle.dump(res_es, f)

    for component, states in (("mha", mha_states), ("mlp", mlp_states), ("residual", res_states)):
        vectors_ckpt = {k: torch.from_numpy(v["direction"]).float() * v["proj_std"] for k, v in states.items()}
        torch.save(vectors_ckpt, output_dir / f"{component}_dim_vectors.pt")
        effect_size_ckpt = {
            k: {
                "effect_size": v["effect_size"],
                "fold_effect_sizes": v["fold_effect_sizes"],
                "auc_roc": v["auc_roc"],
                "fold_aucs": v["fold_aucs"],
            }
            for k, v in states.items()
        }
        with open(output_dir / f"{component}_dim_effect_size.pkl", "wb") as f:
            pickle.dump(effect_size_ckpt, f)

    return {
        "mha_best_key": max(mha_es, key=lambda k: abs(mha_es[k])),
        "mlp_best_key": max(mlp_es, key=lambda k: abs(mlp_es[k])),
        "residual_best_key": max(res_es, key=lambda k: abs(res_es[k])),
    }
