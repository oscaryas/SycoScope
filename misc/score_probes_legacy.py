#!/usr/bin/env python3
"""
LEGACY holding pen: the cross-cell analyses from the retired
analyze_probes/analyze_probes.py that only work on the OLD prompt_probes run
layout, which the new cache_activations.py / train_probes.py pipeline does not
produce:

  results/prompt_probes/<run>/probes/<cell>/probes.npz        (coef/mean/scale per act_key)
  results/prompt_probes/<run>/activations/<cell>.npz          (pos_L## keys, probes_core.load_acts)
  results/prompt_probes/<run>/activations/<cell>_index.jsonl
  results/prompt_probes/<run>/prompt_split.json               (shared train/test prompt split)
  results/prompt_probes/<run>/eval_*/activations.npz          (old eval_* caches)

Moved here rather than faked on the new format:
  * transfer_matrix: cell i's probe on cell j's held-out prompts needs the
    shared prompt_split; new probes are refit on ALL rows of their cache.
  * length_analysis: needs per-cell n_response_tokens in the old index.
  * reliability_ceiling / score_correlations / discover_eval_bases in their
    run-dir form, and the old main() (ANOVA over eval_elephant/summary.json).
  * load_probes / apply_probe for the old .npz probes (still used by
    misc/summarize_syconbench.py and misc/cluster_syconbench_stability.py).

Format-agnostic helpers (ANOVA, clustering, plots) are imported from
scorer.score_probes; the new-format equivalents live there.

Usage (old run dirs only):
    python -m misc.score_probes_legacy --run-name main
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import common  # noqa: E402
from analyze_probes import probes_core as ga  # noqa: E402
from analyze_probes.probes_core import load_cell, paired_win_rate, safe_auc  # noqa: E402
from scorer.score_probes import (  # noqa: E402
    anova, anova_plot, as_matrix, cluster_from_matrix, cluster_sweep, correlation_dendrogram, cosine, heatmap,
)

ELEPHANT_BASIS = "elephant"
LEGACY_ANOVA_FACTORS = {"prompt_pair": "spec", "layer": "layer", "position": "position"}


def load_probes(run_dir: Path, slug: str) -> dict:
    """{act_key: {coef, intercept, mean, scale, direction_raw, proj_std}}"""
    path = run_dir / "probes" / slug / "probes.npz"
    if not path.exists():
        return {}
    with np.load(path) as z:
        out: dict[str, dict] = {}
        for name in z.files:
            key, field = name.split("__")
            out.setdefault(key, {})[field] = z[name]
    return out


def apply_probe(probe: dict, X: np.ndarray) -> np.ndarray:
    """Replicate sklearn's decision_function on standardized features."""
    scale = np.where(probe["scale"] == 0, 1.0, probe["scale"])
    return ((X - probe["mean"]) / scale) @ probe["coef"] + float(probe["intercept"])


def transfer_matrix(run_dir, slugs, probes, position, layer, split, drop_degenerate):
    """rows = probe cell, cols = evaluation cell, on the shared holdout prompts."""
    key = ga.act_key(position, layer)
    test_set = set(split["test"])
    evals = {}
    for slug in slugs:
        X, y, pids, _ = load_cell(run_dir, slug, position, layer, drop_degenerate)
        if not pids:
            continue
        mask = np.array([p in test_set for p in pids], dtype=bool)
        if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
            continue
        evals[slug] = (X[mask], y[mask], [p for p, m in zip(pids, mask) if m])

    win = {i: {} for i in slugs}
    auc = {i: {} for i in slugs}
    for i in slugs:
        probe = probes.get(i, {}).get(key)
        if probe is None:
            continue
        for j, (Xj, yj, pj) in evals.items():
            s = apply_probe(probe, Xj)
            win[i][j] = paired_win_rate(s, yj, pj)[0]
            auc[i][j] = safe_auc(yj, s)
    return {"paired_win_rate": win, "auc": auc, "eval_cells": sorted(evals)}


def discover_eval_bases(run_dir: Path) -> list[str]:
    """Every eval_* directory with cached activations (eval_sypr, eval_moral,
    eval_elephant, ...), in a stable order. Each is a distinct real-world
    dataset -- correlate on each separately before ever pooling (V1 plan step
    3: "pooled correlations can reflect differences between datasets rather
    than agreement within them")."""
    return sorted(
        p.name for p in run_dir.glob("eval_*")
        if (p / "activations.npz").exists()
    )


def load_basis_activations(run_dir, basis_slug, key) -> np.ndarray | None:
    """(n_rows, hidden_dim) for one (eval_* basis, position, layer), or None
    if the eval directory or that (position, layer) isn't cached."""
    if basis_slug == ELEPHANT_BASIS:
        basis_slug = "eval_elephant"
    npz = run_dir / basis_slug / "activations.npz"
    if not npz.exists():
        return None
    with np.load(npz) as z:
        if key not in z.files:
            return None
        return z[key].astype(np.float32)


def score_correlations(run_dir, slugs, probes, position, layer, basis_slug):
    """Pearson correlation between probe scores over one common sample set.

    An `eval_*` basis (eval_sypr, eval_moral, eval_elephant, ...; "elephant"
    is a back-compat alias for eval_elephant) uses real-world OOD activations
    -- what the paper does in section 5.3, correlating probe outputs "across
    all evaluation samples". That is the version to report: correlations
    measured in-distribution are computed where every probe saturates, so
    they describe behaviour on data that cannot discriminate between them.

    Any per-cell slug (e.g. `neutral`) is also accepted as an in-distribution
    basis. `neutral` is the cleanest of those -- identical inputs for every
    probe, and no probe trained on it -- and remains the reference for
    control-adjusted scores (section 5.5).
    """
    key = ga.act_key(position, layer)
    resolved = ELEPHANT_BASIS if basis_slug == ELEPHANT_BASIS else basis_slug
    if resolved.startswith("eval_") or resolved == ELEPHANT_BASIS:
        X = load_basis_activations(run_dir, resolved, key)
        if X is None:
            return None
    else:
        npz = run_dir / "activations" / f"{basis_slug}.npz"
        if not npz.exists():
            return None
        X = ga.load_acts(run_dir, basis_slug, position, layer)

    names, scores = [], []
    for slug in slugs:
        probe = probes.get(slug, {}).get(key)
        if probe is None:
            continue
        names.append(slug)
        scores.append(apply_probe(probe, X))
    if len(names) < 2:
        return None

    S = np.vstack(scores)
    corr = np.corrcoef(S)
    medians = {n: float(np.median(s)) for n, s in zip(names, scores)}
    return {
        "basis": basis_slug,
        "n_samples": int(X.shape[0]),
        "cells": names,
        "correlation": corr.tolist(),
        "control_median": medians,
    }


def length_analysis(run_dir, slugs, probes, position, layer, drop_degenerate) -> dict:
    """Is the probe reading response length rather than content?

    Responses are generated uncapped, so length varies and varies *by class* --
    smoke runs show the sycophantic side running roughly twice as long on some
    cells. With `response` mean-pooling the number of vectors averaged then
    differs systematically between classes, and mean-pooling over different
    lengths carries class information on its own: long sequences regress toward
    a generic-text mean while short ones stay near their distinctive opening.

    `first5` is immune by construction (always five tokens), so a large
    correlation on `response` alongside a small one on `first5` localises the
    problem to the pooling rather than to the representation.

    Read `score_length_r_within`, not `score_length_r`. The raw correlation is
    confounded by class: when the probe separates well and lengths differ by
    class, score and length are both driven by the label, so |r| is large even
    where reading length is impossible.

    `last_prompt` is a free null control for exactly that. Response length is
    causally unavailable there -- no response token has been generated yet --
    so its within-class r must sit near zero. Observed on the smoke run:
    ctrl_politeness raw r = +0.826 (length gap +300 tokens) but within-class
    r = +0.077. A large within-class value at last_prompt would mean this
    metric is broken, not that the probe reads length.
    """
    key = ga.act_key(position, layer)
    out = {}
    for slug in slugs:
        probe = probes.get(slug, {}).get(key)
        if probe is None:
            continue
        index = ga.load_index(run_dir, slug)
        X, y, pids, _ = load_cell(run_dir, slug, position, layer, drop_degenerate)
        if not pids:
            continue
        kept = {p: True for p in pids}
        lens = np.array(
            [r["n_response_tokens"] for r in index if r["prompt_id"] in kept], dtype=float
        )
        if len(lens) != len(y):  # defensive: index/array drift
            continue
        scores = apply_probe(probe, X)
        r = float(np.corrcoef(scores, lens)[0, 1]) if np.std(lens) > 0 else float("nan")

        # The overall correlation is confounded by class and cannot answer the
        # question on its own: when a probe separates well AND length differs by
        # class, score and length are both driven by the label, so |r| is large
        # even where the probe demonstrably cannot be reading length (it shows
        # up at last_prompt, before any response token exists).
        #
        # The within-class correlation removes that shared cause. It asks: among
        # responses that share a label, do longer ones score higher? That is the
        # actual "probe reads length" signal.
        within = []
        for label in (0, 1):
            m = y == label
            if m.sum() >= 3 and np.std(lens[m]) > 0 and np.std(scores[m]) > 0:
                within.append(float(np.corrcoef(scores[m], lens[m])[0, 1]))
        r_within = float(np.mean(within)) if within else float("nan")

        out[slug] = {
            "score_length_r": r,
            "score_length_r_within": r_within,
            "n_within_classes": len(within),
            "mean_len_pos": float(lens[y == 1].mean()) if (y == 1).any() else None,
            "mean_len_neg": float(lens[y == 0].mean()) if (y == 0).any() else None,
        }
        if out[slug]["mean_len_pos"] is not None and out[slug]["mean_len_neg"] is not None:
            out[slug]["mean_len_gap"] = out[slug]["mean_len_pos"] - out[slug]["mean_len_neg"]
    return out


def reliability_ceiling(run_dir, slug, position, layer, split, n_splits, seed, C, max_iter, drop_degenerate):
    """Split-half cosine between two probe directions fit on disjoint prompt halves.

    This is how well the direction is estimated at all. Cross-cell cosines are
    uninterpretable without it. Halves are split by prompt_id, never by row, so
    a prompt's two polarities stay together.
    """
    from analyze_probes.probes_core import fit_probe

    X, y, pids, _ = load_cell(run_dir, slug, position, layer, drop_degenerate)
    if not pids:
        return None
    train_set = set(split["train"])
    mask = np.array([p in train_set for p in pids], dtype=bool)
    X, y, pids = X[mask], y[mask], [p for p, m in zip(pids, mask) if m]
    uniq = sorted(set(pids))
    if len(uniq) < 4:
        return None

    rng = np.random.default_rng(seed)
    cosines = []
    for _ in range(n_splits):
        order = rng.permutation(len(uniq))
        half = len(uniq) // 2
        a = {uniq[i] for i in order[:half]}
        b = {uniq[i] for i in order[half : 2 * half]}
        dirs = []
        for group in (a, b):
            m = np.array([p in group for p in pids], dtype=bool)
            if len(np.unique(y[m])) < 2:
                dirs = []
                break
            scaler, clf = fit_probe(X[m], y[m], seed, C, max_iter)
            raw = clf.coef_[0] / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
            dirs.append(raw / (np.linalg.norm(raw) + 1e-12))
        if len(dirs) == 2:
            cosines.append(cosine(dirs[0], dirs[1]))
    if not cosines:
        return None
    return {"mean": float(np.mean(cosines)), "min": float(np.min(cosines)), "max": float(np.max(cosines)), "n": len(cosines)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--positions", type=str, nargs="+", default=None, choices=common.POSITIONS)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--n-clusters", type=int, default=5, help="Paper found 5 clusters over 23 prompts.")
    parser.add_argument(
        "--cluster-only", action="store_true",
        help="Only compute score correlations, clustering, and their heatmaps from cached "
        "evaluation activations; skip ANOVA, transfer, length analysis, and probe geometry.",
    )
    parser.add_argument(
        "--cluster-basis",
        type=str,
        nargs="+",
        default=None,
        help="Sample set(s) for score-correlation clustering: any of 'elephant' (alias for "
        "eval_elephant), 'eval_sypr', 'eval_moral', or an in-distribution cell slug such as "
        "'neutral'. Default: every eval_* directory with cached activations under this run, "
        "each correlated separately (V1 plan step 3 -- never pool datasets by default).",
    )
    parser.add_argument("--ceiling-splits", type=int, default=5, help="0 to skip reliability ceilings.")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-degenerate", action="store_true")
    args = parser.parse_args()

    run_dir = common.resolve_run_dir(args.run_name, create=False)
    if not run_dir.exists():
        raise SystemExit(f"no such run: {run_dir}")
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit("summary.json missing -- run train_probes.py first")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split = json.loads((run_dir / "prompt_split.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "activations" / "meta.json").read_text(encoding="utf-8"))
    out_dir = run_dir / "analysis"
    plot_dir = run_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    drop_degenerate = not args.keep_degenerate

    slugs = sorted({r["slug"] for r in summary["rows"]}, key=lambda s: common.all_slugs().index(s))
    probes = {s: load_probes(run_dir, s) for s in slugs}
    spec_of = {r["slug"]: r["spec"] for r in summary["rows"]}
    positions = args.positions or sorted({r["position"] for r in summary["rows"]}, key=common.POSITIONS.index)
    layers = args.layers or sorted({r["layer"] for r in summary["rows"]})

    print(f"{len(slugs)} cells, positions {positions}, layers {layers}")

    # ---- 1. ANOVA -------------------------------------------------------
    # On OOD AUC only, as the paper does (section 4.1: "evaluated each on the
    # validation dataset"). In-distribution AUC saturates near 1.000, so it has
    # no variance to decompose and the residual absorbs everything.
    ood_path = run_dir / "eval_elephant" / "summary.json"
    if args.cluster_only:
        print("Score-correlation clustering only; other analyses skipped.")
    elif not ood_path.exists():
        print()
        print(f"--- ANOVA skipped: {ood_path} not found; run the old eval_social_sycophancy.py first ---")
    else:
        ood = json.loads(ood_path.read_text(encoding="utf-8"))
        anova_rows = [
            dict(r, spec=r["slug"]) for r in ood["rows"]
            if r["split"] == "eval" and r.get("dataset", "all") == "all"
            and r["slug"] not in common.NULL_SLUGS
        ]
        decomposition = anova(anova_rows, "auc", LEGACY_ANOVA_FACTORS)
        (out_dir / "anova.json").write_text(
            json.dumps({"source": "ELEPHANT eval-half AUC", **decomposition}, indent=2), encoding="utf-8"
        )
        print()
        print("--- ANOVA on ELEPHANT eval-half AUC (paper: prompt 70.6%, layer 2.7%, token selection 0.6%) ---")
        if "error" in decomposition:
            print(f"  skipped: {decomposition['error']}")
        else:
            for label in ("prompt_pair", "layer", "position", "residual"):
                e = decomposition.get(label)
                if not e:
                    continue
                pv = e.get("p")
                ptxt = "" if pv is None else f"  p={pv:.3g}"
                print(f"  {label:<12} {e['pct_variance']:>6.2f}%  (df={e['df']}){ptxt}")
            anova_plot(plot_dir / "anova_auc.png", decomposition, "AUC variance explained (ELEPHANT)")

    # ---- 2/3/4 per (position, layer) ------------------------------------
    for position in positions:
        for layer in layers:
            tag = f"{position}_L{layer:02d}"
            print(f"\n=== {tag} ===")

            tm = (
                {"paired_win_rate": {}, "eval_cells": []}
                if args.cluster_only else
                transfer_matrix(run_dir, slugs, probes, position, layer, split, drop_degenerate)
            )
            rows_present = [s for s in slugs if tm["paired_win_rate"].get(s)]
            cols = tm["eval_cells"]
            if rows_present and cols:
                (out_dir / f"transfer_{tag}.json").write_text(
                    json.dumps({"specs": {s: spec_of.get(s) for s in slugs}, **tm}, indent=2), encoding="utf-8"
                )
                M = as_matrix(tm["paired_win_rate"], rows_present, cols)
                heatmap(
                    plot_dir / f"transfer_paired_{tag}.png",
                    M,
                    rows_present,
                    cols,
                    f"Transfer: paired win rate ({tag})",
                    0.0,
                    1.0,
                    "RdBu_r",
                    "rows = probe, cols = evaluated cell; 0.5 = chance",
                )
                uni = tm["paired_win_rate"].get("general_baseline", {})
                if uni:
                    diag = {s: tm["paired_win_rate"].get(s, {}).get(s) for s in cols}
                    print("  universal -> cell (vs that cell's own probe):")
                    for s in cols:
                        own = diag.get(s)
                        print(
                            f"    {s:<26} universal {uni.get(s, float('nan')):.3f}"
                            + (f"   own {own:.3f}" if own is not None else "")
                        )

            la = {} if args.cluster_only else length_analysis(
                run_dir, slugs, probes, position, layer, drop_degenerate
            )
            if la:
                (out_dir / f"length_{tag}.json").write_text(json.dumps(la, indent=2), encoding="utf-8")

                def _within(e):
                    v = e.get("score_length_r_within")
                    return 0.0 if v is None or v != v else v

                worst = max(la.items(), key=lambda kv: abs(_within(kv[1])))
                print(f"  score-vs-length: max within-class |r| {abs(_within(worst[1])):.3f} ({worst[0]})")
                for slug, e in sorted(la.items(), key=lambda kv: -abs(_within(kv[1])))[:3]:
                    gap = e.get("mean_len_gap")
                    print(
                        f"    {slug:<26} within r={_within(e):+.3f}  (raw r={e['score_length_r']:+.3f})"
                        + (f"  len gap {gap:+.0f} tok" if gap is not None else "")
                    )
                # Gate on the within-class value: the raw correlation is large
                # whenever the probe separates well and lengths differ by class,
                # which says nothing about whether length is being read.
                if abs(_within(worst[1])) > 0.5 and position == "response":
                    print(
                        "    WARNING: within-class, response-position scores still track length.\n"
                        "    Compare against first5 (length-immune by construction); if first5 holds\n"
                        "    up, the pooling is the problem, not the representation."
                    )

            bases = args.cluster_basis or discover_eval_bases(run_dir) or [common.NEUTRAL_SLUG]
            any_basis_found = False
            for basis in bases:
                sc = score_correlations(run_dir, slugs, probes, position, layer, basis)
                if sc is None:
                    print(f"  [{basis}] skipped: cached activations unavailable for {tag}")
                    continue
                any_basis_found = True
                corr = np.array(sc["correlation"], dtype=float)
                # Signed is the primary view (V1 plan step 3): opposite-scoring
                # probes must NOT land in the same cluster. Unsigned (1-|r|) is
                # reported alongside only as a "same axis, either direction" view.
                sc["clustering_signed"] = cluster_from_matrix(sc["cells"], corr, args.n_clusters, signed=True)
                sc["clustering_unsigned"] = cluster_from_matrix(sc["cells"], corr, args.n_clusters, signed=False)
                sc["clustering"] = sc["clustering_signed"]  # back-compat default
                sc["cluster_sweep_signed"] = cluster_sweep(sc["cells"], corr, range(2, min(10, len(sc["cells"]))), signed=True)
                (out_dir / f"score_correlation_{tag}_{basis}.json").write_text(json.dumps(sc, indent=2), encoding="utf-8")
                heatmap(
                    plot_dir / f"score_correlation_{tag}_{basis}.png",
                    corr,
                    sc["cells"],
                    sc["cells"],
                    f"Probe score correlation on {sc['basis']} ({tag})",
                    -1.0,
                    1.0,
                    "RdBu_r",
                    f"Pearson r over {sc['n_samples']} common samples",
                )
                correlation_dendrogram(
                    plot_dir / f"dendrogram_{tag}_{basis}.png",
                    corr, sc["cells"],
                    f"Probe score clustering on {sc['basis']} ({tag}, n={sc['n_samples']})",
                )
                print(f"  [{basis}] score-correlation clusters, SIGNED (n={sc['n_samples']}):")
                for cname, members in sorted(sc["clustering_signed"]["clusters"].items()):
                    rng_ = sc["clustering_signed"]["internal_similarity_range"][cname]
                    print(f"    {cname}: {members}  internal r {rng_}")
                best_k = max(
                    sc["cluster_sweep_signed"].items(),
                    key=lambda kv: kv[1]["silhouette"] if kv[1]["silhouette"] is not None else -2,
                    default=(None, None),
                )
                if best_k[0] is not None:
                    print(f"    best-silhouette k (signed): {best_k[0]} (silhouette {best_k[1]['silhouette']:.3f})")
                print(f"  [{basis}] score-correlation clusters, UNSIGNED 1-|r| (same axis, either direction -- supplementary):")
                for cname, members in sorted(sc["clustering_unsigned"]["clusters"].items()):
                    rng_ = sc["clustering_unsigned"]["internal_similarity_range"][cname]
                    print(f"    {cname}: {members}  internal r {rng_}")
            if not any_basis_found:
                print(
                    f"  score correlation skipped: none of {bases} have cached activations. "
                    "Run an eval_*.py script (or generate the 'neutral' cell) for a common basis."
                )

            key = ga.act_key(position, layer)
            dirs = {s: probes[s][key]["direction_raw"] for s in slugs if key in probes.get(s, {})}
            if not args.cluster_only and len(dirs) >= 2:
                names = list(dirs)
                cos = np.array([[cosine(dirs[a], dirs[b]) for b in names] for a in names])
                geometry = {"cells": names, "cosine": cos.tolist()}
                if args.ceiling_splits:
                    geometry["reliability_ceiling"] = {
                        s: reliability_ceiling(
                            run_dir, s, position, layer, split, args.ceiling_splits,
                            args.seed, args.C, args.max_iter, drop_degenerate,
                        )
                        for s in names
                    }
                geometry["clustering"] = cluster_from_matrix(names, cos, args.n_clusters)
                (out_dir / f"geometry_{tag}.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
                heatmap(
                    plot_dir / f"cosine_{tag}.png",
                    cos,
                    names,
                    names,
                    f"Probe direction cosine ({tag})",
                    -1.0,
                    1.0,
                    "RdBu_r",
                    "compare each off-diagonal against the per-cell reliability ceiling",
                )
                ceilings = geometry.get("reliability_ceiling") or {}
                if any(ceilings.values()):
                    print("  reliability ceilings (split-half cosine of the same cell's direction):")
                    for s in names:
                        c = ceilings.get(s)
                        if c:
                            print(f"    {s:<26} {c['mean']:.3f}")
                    print(
                        "    An off-diagonal cosine near a cell's ceiling means the same direction;\n"
                        "    a raw cosine cannot be read without this."
                    )

    print(f"\nDone. -> {out_dir} and {plot_dir}")


if __name__ == "__main__":
    main()
