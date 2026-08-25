#!/usr/bin/env python3
"""
Fit linear probes predicting which side of a contrastive system-prompt pair was
in context. Stage 3 of the prompt_probes pipeline. CPU only -- reads the cached
.npz, so no GPU and no model weights.

Estimator follows Natarajan et al. (2026) section 3.6: sklearn
LogisticRegression with L2 and lambda = 1 (C = 1.0) on activations standardized
to zero mean and unit variance. The scaler is fit on the TRAINING fold only;
fitting it on all data leaks the test distribution into the features. Note the
resulting weight lives in scaled space, so comparing it against any raw-space
direction (anything in tool_calling/.../results/probing/*, or a steering
vector) requires mapping back via w / scale -- `direction_raw` is stored for
exactly that.

This deliberately does NOT use sycophancy_probes.train_probe /
train_residual_probes / save_probe_results:

  * train_probe's _stratified_folds splits at random over examples. Under the
    paired design, prompt p's sycophantic response can land in train while its
    non-sycophantic response lands in test. Both are responses to the same user
    turn, sharing topic, vocabulary and the entire prompt in context, so a
    4096-dim probe fit on ~320 examples has ample capacity to encode "this is
    the linguistics-professor item, and the one I saw was label 1". That
    inflates accuracy in the flattering direction. grouped_folds assigns whole
    prompt_ids instead.
  * the estimator is now sklearn rather than torch nn.Linear + BCE, so the
    save_probe_results layout (torch state dicts keyed by layer across an
    mha/mlp/residual trio) no longer fits.

Consequence worth stating in any writeup: these numbers are NOT directly
comparable to the existing results/probing/* artifacts, which used the repo's
torch probe on raw activations.

Usage:
    python train_probes.py --run-name main
    python train_probes.py --run-name main --cells pv_explicit          # one cell
    python train_probes.py --run-name main --shuffle-labels             # leakage control
    python train_probes.py --run-name main --rebuild-summary            # re-aggregate only
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
import get_activations as ga  # noqa: E402

# sycophancy_probes does a bare `from sycophancy_model_registry import ...`, so
# its directory has to be importable, not just REPO_ROOT.
if str(common.SYCOPHANCY_DIR) not in sys.path:
    sys.path.insert(0, str(common.SYCOPHANCY_DIR))

from sycophancy_probes import bootstrap_ci, wilson_ci  # noqa: E402

SPEC_PREFIX = {"baseline": "universal", "taxonomy": "cell", "control": "control"}

# Degeneracy reasons that justify dropping a row (and its pair partner).
# "truncated" and "refusal" are deliberately absent: neither corrupts the
# activations, and dropping them would condition the sample on length or on
# refusal behaviour rather than remove those effects. Both stay flagged.
DROP_REASONS = frozenset({"empty", "too_short", "repetitive"})


def spec_id(pair_type: str, slug: str) -> str:
    """Stable probe identifier. The baseline pair is the 'universal' probe."""
    if pair_type == "baseline":
        return "universal"
    return f"{SPEC_PREFIX.get(pair_type, pair_type)}__{slug}"


# ---------------------------------------------------------------------------
# Grouped splitting
# ---------------------------------------------------------------------------


def grouped_folds(prompt_ids: list[str], n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Assign whole prompt_ids to folds; returns a per-row fold index.

    Class balance needs no separate stratification here: in the paired design
    each prompt_id carries exactly one label-1 and one label-0 row per cell, so
    assigning a group assigns one of each.
    """
    uniq = sorted(set(prompt_ids))
    order = rng.permutation(len(uniq))
    fold_of_group = {uniq[g]: int(i % n_folds) for i, g in enumerate(order)}
    return np.array([fold_of_group[p] for p in prompt_ids], dtype=int)


def group_split(prompt_ids: list[str], test_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """One prompt-level holdout, shared by every cell so transfer comparisons
    are all measured on the same unseen prompts."""
    uniq = sorted(set(prompt_ids))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test = sorted(uniq[i] for i in order[:n_test])
    train = sorted(uniq[i] for i in order[n_test:])
    return train, test


def load_or_make_split(run_dir: Path, test_frac: float, seed: int) -> dict:
    """The split is created once from user_prompts.jsonl and then reused, so it
    is independent of which cells happen to have been run."""
    path = run_dir / "prompt_split.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    prompts = common.read_jsonl(run_dir / "user_prompts.jsonl")
    train, test = group_split([p["prompt_id"] for p in prompts], test_frac, seed)
    split = {"test_frac": test_frac, "seed": seed, "train": train, "test": test}
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_probe(X: np.ndarray, y: np.ndarray, seed: int, C: float, max_iter: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    # L2 is LogisticRegression's default. Passing penalty="l2" explicitly is
    # deprecated in sklearn 1.8 and removed in 1.10; omitting it is verified
    # bit-identical (coef max abs diff 0.0) and keeps the paper's lambda=1
    # as C=1.0.
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    clf.fit(scaler.transform(X), y)
    return scaler, clf


def score(scaler, clf, X: np.ndarray) -> np.ndarray:
    """Signed distance from the boundary. Continuous scores, not thresholded
    predictions, because AUC and the paired comparison both need a ranking."""
    return clf.decision_function(scaler.transform(X))


def safe_auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    """None when a split has only one class present, where AUC is undefined."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def paired_win_rate(scores: np.ndarray, y: np.ndarray, prompt_ids: list[str]) -> tuple[float, int]:
    """Fraction of prompts whose label-1 response outscores its label-0 response.

    Preferred over accuracy for transfer because it needs neither a threshold
    nor an intercept: a probe from another cell may have a badly miscalibrated
    bias, shifting every score the same way, which destroys accuracy while
    leaving the ranking intact. It is also tighter than AUC on the same data --
    AUC averages over all positive/negative cross-pairs including
    content-mismatched ones, whereas this compares only responses to the *same*
    user turn. Ties count as half.
    """
    by: dict[str, dict[int, float]] = {}
    for s, label, pid in zip(scores, y, prompt_ids):
        by.setdefault(pid, {})[int(label)] = float(s)
    wins = [
        1.0 if d[1] > d[0] else (0.5 if d[1] == d[0] else 0.0)
        for d in by.values()
        if 1 in d and 0 in d
    ]
    return (float(np.mean(wins)) if wins else float("nan")), len(wins)


def fit_cv(X, y, prompt_ids, n_folds, seed, C, max_iter) -> dict:
    """Grouped k-fold CV. Returns the same key set as sycophancy_probes.train_probe
    (so existing plotting code stays drop-in), plus group counts."""
    rng = np.random.default_rng(seed)
    n_groups = len(set(prompt_ids))
    n_folds = min(n_folds, n_groups)
    fold_of = grouped_folds(prompt_ids, n_folds, rng)

    fold_accs, train_accs, fold_aucs = [], [], []
    total_correct = total_test = 0
    for fold in range(n_folds):
        test_mask = fold_of == fold
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        if len(np.unique(y[train_mask])) < 2:
            continue
        scaler, clf = fit_probe(X[train_mask], y[train_mask], seed, C, max_iter)
        s_test = score(scaler, clf, X[test_mask])
        preds = (s_test > 0).astype(int)
        n_correct = int((preds == y[test_mask]).sum())
        total_correct += n_correct
        total_test += int(test_mask.sum())
        fold_accs.append(n_correct / int(test_mask.sum()))
        train_preds = (score(scaler, clf, X[train_mask]) > 0).astype(int)
        train_accs.append(float((train_preds == y[train_mask]).mean()))
        auc = safe_auc(y[test_mask], s_test)
        if auc is not None:
            fold_aucs.append(auc)

    ci_lower, ci_upper = wilson_ci(total_correct, total_test) if total_test else (0.0, 0.0)
    if fold_aucs:
        auc_roc = float(np.mean(fold_aucs))
        auc_ci_lower, auc_ci_upper = bootstrap_ci(np.array(fold_aucs))
    else:
        auc_roc = auc_ci_lower = auc_ci_upper = float("nan")

    return {
        "accuracy": total_correct / total_test if total_test else float("nan"),
        "fold_accuracies": fold_accs,
        "train_accuracy": float(np.mean(train_accs)) if train_accs else float("nan"),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_test": total_test,
        "auc_roc": auc_roc,
        "fold_aucs": fold_aucs,
        "auc_ci_lower": auc_ci_lower,
        "auc_ci_upper": auc_ci_upper,
        "n_folds": n_folds,
        "n_groups": n_groups,
    }


def fit_holdout(X, y, prompt_ids, split, seed, C, max_iter) -> dict | None:
    """One probe on the shared train prompts, evaluated on the shared test
    prompts. This is the probe used for cross-cell transfer and geometry."""
    train_set, test_set = set(split["train"]), set(split["test"])
    tr = np.array([p in train_set for p in prompt_ids], dtype=bool)
    te = np.array([p in test_set for p in prompt_ids], dtype=bool)
    if tr.sum() == 0 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
        return None

    scaler, clf = fit_probe(X[tr], y[tr], seed, C, max_iter)
    s_test = score(scaler, clf, X[te])
    test_ids = [p for p, m in zip(prompt_ids, te) if m]
    win_rate, n_pairs = paired_win_rate(s_test, y[te], test_ids)

    coef = clf.coef_[0].astype(np.float64)
    # Map the scaled-space weight back to raw activation space, so cosines
    # against any raw-space direction are meaningful.
    raw = coef / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    direction_raw = raw / (np.linalg.norm(raw) + 1e-12)

    # Sign convention: every stored direction points toward label 1. Guaranteed
    # by the fit, but asserted because compute_dim_direction and later
    # projection arithmetic can flip it. For CONTROL pairs, label 1 is the
    # legitimate behaviour, not sycophancy.
    proj = X @ direction_raw
    assert proj[y == 1].mean() > proj[y == 0].mean(), "direction does not point toward label 1"

    return {
        "coef": coef,
        "intercept": float(clf.intercept_[0]),
        "mean": scaler.mean_.astype(np.float64),
        "scale": scaler.scale_.astype(np.float64),
        "direction_raw": direction_raw,
        "proj_std": float(np.std(proj)),
        "test_accuracy": float(((s_test > 0).astype(int) == y[te]).mean()),
        "test_auc": safe_auc(y[te], s_test),
        "test_paired_win_rate": win_rate,
        "n_test_pairs": n_pairs,
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
    }


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def load_cell(run_dir: Path, slug: str, position: str, layer: int, drop_degenerate: bool):
    """Returns (X, y, prompt_ids, n_dropped) with pairing preserved.

    Dropping a degenerate row also drops its pair partner, so exact 1:1 pairing
    and exact class balance survive filtering. Without that, a silent imbalance
    plus a class-weighted loss produces an accuracy nobody can interpret.
    """
    index = ga.load_index(run_dir, slug)
    X = ga.load_acts(run_dir, slug, position, layer)
    assert len(index) == X.shape[0], f"{slug}: index has {len(index)} rows, array has {X.shape[0]}"

    keep = np.ones(len(index), dtype=bool)
    if drop_degenerate:
        bad_prompts = {r["prompt_id"] for r in index if r["degenerate"] in DROP_REASONS}
        keep = np.array([r["prompt_id"] not in bad_prompts for r in index], dtype=bool)
    n_dropped = int((~keep).sum())
    if len(index) and not keep.any():
        reasons = sorted({r["degenerate"] for r in index if r["degenerate"]})
        print(
            f"  WARNING [{slug} {position}_L{layer:02d}]: all {len(index)} rows dropped as "
            f"degenerate (reasons: {reasons}). Nothing left to fit. Inspect the generations, or "
            "pass --keep-degenerate if the flags are spurious (e.g. a small --max-new-tokens in a "
            "smoke run makes every response look 'truncated')."
        )

    rows = [r for r, k in zip(index, keep) if k]
    y = np.array([r["label"] for r in rows], dtype=int)
    return X[keep], y, [r["prompt_id"] for r in rows], n_dropped


def rebuild_summary(run_dir: Path) -> dict:
    """Aggregate whatever per-cell metrics.json files exist, so the summary is
    correct after running a single cell."""
    rows = []
    for path in sorted((run_dir / "probes").glob("*/metrics.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    split_path = run_dir / "prompt_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else {}
    summary = {
        "schema_version": 1,
        "code_version": common.get_code_version(),
        "split": {
            "test_frac": split.get("test_frac"),
            "seed": split.get("seed"),
            "n_train_prompts": len(split.get("train", [])),
            "n_test_prompts": len(split.get("test", [])),
        },
        "n_rows": len(rows),
        "rows": sorted(rows, key=lambda r: (r["spec"], r["position"], r["layer"])),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if rows:
        import pandas as pd

        pd.DataFrame(summary["rows"]).to_csv(run_dir / "summary.csv", index=False)
    return summary


def print_gates(summary: dict, shuffled: bool) -> None:
    rows = summary["rows"]
    if not rows:
        return

    def mean_auc(position):
        vals = [r["cv_auc"] for r in rows if r["position"] == position and r["cv_auc"] == r["cv_auc"]]
        return float(np.mean(vals)) if vals else float("nan")

    ceiling = mean_auc("last_prompt")
    print("\n--- gates ---")
    if shuffled:
        aucs = [r["cv_auc"] for r in rows if r["cv_auc"] == r["cv_auc"]]
        n_groups = min((r["n_groups"] for r in rows), default=0)
        mean, mx = float(np.mean(aucs)), float(np.max(aucs))
        print(
            f"label-shuffle control over {len(aucs)} settings: mean CV AUC {mean:.3f}, "
            f"max {mx:.3f} (want mean ~0.5)"
        )
        # Gate on the mean, not the max. A real leak elevates every setting
        # systematically; the max over many noisy per-fold estimates is near 1.0
        # by construction whenever folds are small, which would make the gate
        # fire on sample size rather than on leakage.
        if n_groups < 20:
            print(
                f"  SKIPPED: only {n_groups} prompt groups. This control needs enough groups for "
                "per-fold AUC to be meaningful; re-run it on the full data."
            )
        elif mean > 0.6:
            print("  FAIL: shuffled labels are still separable -- grouped_folds or the index is leaking")
        else:
            print("  PASS")
        return
    print(f"ceiling gate  (last_prompt mean CV AUC): {ceiling:.3f} (want >= 0.95)")
    if ceiling < 0.95:
        print(
            "  WARNING: last_prompt should be near-trivial -- the two classes differ there by\n"
            "  literally different system-prompt tokens in context. A low value means the system\n"
            "  prompt is not reaching the model or the extraction is misaligned. Fix before\n"
            "  interpreting anything else."
        )
    print(f"intent  (first5   mean CV AUC): {mean_auc('first5'):.3f}")
    print(f"behaviour (response mean CV AUC): {mean_auc('response'):.3f}")
    print(
        "\nfirst5 -> response is the content-contamination measurement. If they match, check\n"
        "the first5_text field in the activation index: five tokens are only a clean intent\n"
        "measurement if they are too sparse to carry the label."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", type=str, required=True)
    common.add_cells_arg(parser)
    parser.add_argument("--positions", type=str, nargs="+", default=None, choices=common.POSITIONS)
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Default: all layers in meta.json.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--C", type=float, default=1.0, help="Inverse L2 strength; 1.0 is the paper's lambda=1.")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-degenerate", action="store_true", help="Do not drop degenerate rows (or their pair partners).")
    parser.add_argument(
        "--shuffle-labels",
        action="store_true",
        help="Permute labels within prompt groups. Every setting must land at chance; this is "
        "the primary check that grouped_folds and the index are not leaking. Writes to "
        "probes_shuffled/ so it cannot overwrite real results.",
    )
    parser.add_argument("--rebuild-summary", action="store_true", help="Re-aggregate summary.json and exit.")
    args = parser.parse_args()

    run_dir = common.resolve_run_dir(args.run_name, create=False)
    if not run_dir.exists():
        raise SystemExit(f"no such run: {run_dir}")

    if args.rebuild_summary:
        summary = rebuild_summary(run_dir)
        print(f"summary.json rebuilt from {summary['n_rows']} rows")
        print_gates(summary, shuffled=False)
        return

    meta_path = run_dir / "activations" / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} missing -- run get_activations.py first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    layers = args.layers or meta["layers"]
    positions = args.positions or list(common.POSITIONS)

    split = load_or_make_split(run_dir, args.test_frac, args.seed)
    print(f"prompt split: {len(split['train'])} train / {len(split['test'])} test prompts (shared by all cells)")

    pairs = {p["slug"]: p for p in common.load_prompt_pairs()}
    requested = args.cells or common.all_slugs(include_neutral=False)
    # The neutral cell has no labels and no pair, so no probe is fitted for it;
    # it is the control set that analyze_probes.py centres scores against.
    available = [
        s for s in requested
        if s != common.NEUTRAL_SLUG and (run_dir / "activations" / f"{s}.npz").exists()
    ]
    if not available:
        raise SystemExit("no cells with activations to train on")

    out_root = run_dir / ("probes_shuffled" if args.shuffle_labels else "probes")
    rng = np.random.default_rng(args.seed)

    for slug in available:
        pair = pairs[slug]
        sid = spec_id(pair["type"], slug)
        cell_dir = out_root / slug
        cell_dir.mkdir(parents=True, exist_ok=True)
        rows, arrays = [], {}
        print(f"\n[{slug}] spec={sid} type={pair['type']}")

        for position in positions:
            for layer in layers:
                X, y, prompt_ids, n_dropped = load_cell(
                    run_dir, slug, position, layer, drop_degenerate=not args.keep_degenerate
                )
                if len(np.unique(y)) < 2:
                    print(f"  {position}_L{layer:02d}: only one class present, skipping")
                    continue
                if args.shuffle_labels:
                    # Permute *within* each prompt group, so the group structure
                    # is preserved and only the label is destroyed.
                    y = y.copy()
                    by: dict[str, list[int]] = {}
                    for i, pid in enumerate(prompt_ids):
                        by.setdefault(pid, []).append(i)
                    for idxs in by.values():
                        y[idxs] = rng.permutation(y[idxs])

                n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
                assert n_pos == n_neg, (
                    f"{slug} {position}_L{layer:02d}: {n_pos} positive vs {n_neg} negative. "
                    "The paired design should be exactly balanced; a mismatch means pair "
                    "partners were dropped inconsistently."
                )

                cv = fit_cv(X, y, prompt_ids, args.n_folds, args.seed, args.C, args.max_iter)
                ho = fit_holdout(X, y, prompt_ids, split, args.seed, args.C, args.max_iter)

                row = {
                    "spec": sid,
                    "slug": slug,
                    "cell": pair["cell"],
                    "spec_type": pair["type"],
                    "position": position,
                    "layer": layer,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "n_groups": cv["n_groups"],
                    "n_dropped": n_dropped,
                    "cv_accuracy": cv["accuracy"],
                    "cv_ci": [cv["ci_lower"], cv["ci_upper"]],
                    "cv_auc": cv["auc_roc"],
                    "cv_auc_ci": [cv["auc_ci_lower"], cv["auc_ci_upper"]],
                    "train_accuracy": cv["train_accuracy"],
                    "holdout_accuracy": ho["test_accuracy"] if ho else None,
                    "holdout_auc": ho["test_auc"] if ho else None,
                    "holdout_paired_win_rate": ho["test_paired_win_rate"] if ho else None,
                }
                rows.append(row)
                if ho:
                    key = ga.act_key(position, layer)
                    for field in ("coef", "intercept", "mean", "scale", "direction_raw", "proj_std"):
                        arrays[f"{key}__{field}"] = np.asarray(ho[field])
                print(
                    f"  {position}_L{layer:02d}: cv_auc {cv['auc_roc']:.3f} "
                    f"acc {cv['accuracy']:.3f} | holdout auc "
                    f"{(ho['test_auc'] if ho and ho['test_auc'] is not None else float('nan')):.3f} "
                    f"paired {(ho['test_paired_win_rate'] if ho else float('nan')):.3f}"
                    + (f" | dropped {n_dropped}" if n_dropped else "")
                )

        (cell_dir / "metrics.json").write_text(
            json.dumps({"spec": sid, "slug": slug, "rows": rows}, indent=2), encoding="utf-8"
        )
        if arrays:
            np.savez(cell_dir / "probes.npz", **arrays)

    if args.shuffle_labels:
        rows = []
        for path in sorted(out_root.glob("*/metrics.json")):
            rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
        print_gates({"rows": rows}, shuffled=True)
        return

    summary = rebuild_summary(run_dir)
    common.write_run_info(run_dir, "train_probes", args, {"cells": available, "n_rows": summary["n_rows"]})
    print(f"\n{summary['n_rows']} probe settings -> {run_dir / 'summary.json'}")
    print_gates(summary, shuffled=False)


if __name__ == "__main__":
    main()
