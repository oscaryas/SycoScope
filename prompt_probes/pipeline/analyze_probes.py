#!/usr/bin/env python3
"""
Cross-probe analysis. Stage 4 of the prompt_probes pipeline. Reads the saved
probes and cached activations; no GPU, no model weights. Kept separate from
train_probes.py because this is the part that gets re-run repeatedly while
training runs once.

Four analyses, all following Natarajan et al. (2026):

1. ANOVA variance decomposition (their section 4.1). How much of the AUC
   variance is attributable to the prompt pair vs the layer vs the token
   position. They found system prompt 70.6%, layer 2.7%, token selection 0.6%
   on Gemma-2-9B (71.0% / 5.8% / 0.3% on Llama-3.3-70B). If that holds here,
   the 3x3 position/layer grid is ~3% of the effect and the prompt axis is
   where any further budget should go. Position may well matter more here than
   it did for them, because we generate on-policy and position controls how
   much response content the probe can see.

2. Transfer matrix. Cell i's probe scored on cell j's held-out-prompt rows.
   Primary statistic is the paired win rate (per prompt, does the
   sycophantic-condition response outscore the non-sycophantic one), with AUC
   alongside for comparability with their reported AUC deltas. Both are
   threshold-free, which is what makes a transferred direction scorable at all:
   a transferred intercept is meaningless across cells, but the ranking is not.
   The `universal` row is the direct answer to "does one probe generalize".

3. Score-correlation clustering (their section 5.3). Pearson correlation
   between probe *scores* over a common evaluation set, then hierarchical
   clustering. This is behavioural similarity. They found 16 linguistically
   distinct prompts collapsing onto ~5 clusters (internal r up to .97), and
   took the control probes forming their own cluster as evidence the deception
   clusters meant something -- our 5 controls play that role.

4. Cosine matrix between probe directions, with split-half reliability
   ceilings. Not redundant with (3): a high score correlation can arise from
   two different directions both loading on a shared component, and only the
   geometry distinguishes that. The ceilings are not optional either -- a
   cross-cosine of 0.4 against a per-cell ceiling of 0.45 means "the same
   direction", not "different directions".

Scores are reported both raw and control-adjusted (median score on the
`neutral` cell subtracted), because probe logit scales are not comparable
across probes.

Usage:
    python analyze_probes.py --run-name main
    python analyze_probes.py --run-name main --positions first5 --layers 16
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
from train_probes import load_cell, paired_win_rate, safe_auc  # noqa: E402


ELEPHANT_BASIS = "elephant"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Same convention as rq1_gate1_geometry.cosine."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------------------------------------------------------------------------
# 1. ANOVA
# ---------------------------------------------------------------------------


def anova(rows: list[dict], value_key: str = "auc") -> dict:
    """Main-effect variance decomposition over a fully crossed design.

    One observation per (spec, position, layer) cell means interactions cannot
    be separated from error, so they land in the residual -- which is exactly
    how the paper reports it ("Residual (Unexplained)").
    """
    usable = [r for r in rows if r.get(value_key) is not None and r[value_key] == r[value_key]]
    if len(usable) < 4:
        return {"error": f"only {len(usable)} usable rows"}

    y = np.array([r[value_key] for r in usable], dtype=float)
    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())
    if ss_total == 0:
        return {"error": "no variance in " + value_key}

    factors = {"prompt_pair": "spec", "layer": "layer", "position": "position"}
    out, ss_explained, df_used = {}, 0.0, 0
    for label, key in factors.items():
        levels = {}
        for r, val in zip(usable, y):
            levels.setdefault(r[key], []).append(val)
        # Unbalanced-safe: weight each level by its own count.
        ss = float(sum(len(v) * (np.mean(v) - grand) ** 2 for v in levels.values()))
        out[label] = {
            "ss": ss,
            "pct_variance": round(100.0 * ss / ss_total, 2),
            "df": len(levels) - 1,
            "n_levels": len(levels),
        }
        ss_explained += ss
        df_used += len(levels) - 1

    ss_resid = max(ss_total - ss_explained, 0.0)
    df_resid = max(len(usable) - 1 - df_used, 1)
    ms_resid = ss_resid / df_resid
    for label, entry in out.items():
        if entry["df"] > 0 and ms_resid > 0:
            f = (entry["ss"] / entry["df"]) / ms_resid
            entry["F"] = round(float(f), 3)
            try:
                from scipy.stats import f as f_dist

                entry["p"] = float(f_dist.sf(f, entry["df"], df_resid))
            except Exception:
                entry["p"] = None

    out["residual"] = {"ss": ss_resid, "pct_variance": round(100.0 * ss_resid / ss_total, 2), "df": df_resid}
    out["_meta"] = {"value_key": value_key, "n_observations": len(usable), "ss_total": ss_total}
    return out


# ---------------------------------------------------------------------------
# Probe loading / scoring
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 2/3/4
# ---------------------------------------------------------------------------


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


def score_correlations(run_dir, slugs, probes, position, layer, basis_slug):
    """Pearson correlation between probe scores over one common sample set.

    basis_slug == "elephant" uses the cached OOD activations, which is what the
    paper does -- section 5.3 correlates probe outputs "across all evaluation
    samples". That is the version to report: correlations measured
    in-distribution are computed where every probe saturates, so they describe
    behaviour on data that cannot discriminate between them.

    Any per-cell slug (e.g. `neutral`) is also accepted as an in-distribution
    basis. `neutral` is the cleanest of those -- identical inputs for every
    probe, and no probe trained on it -- and remains the reference for
    control-adjusted scores (section 5.5).
    """
    key = ga.act_key(position, layer)
    if basis_slug == ELEPHANT_BASIS:
        npz = run_dir / "eval_elephant" / "activations.npz"
        if not npz.exists():
            return None
        with np.load(npz) as z:
            if key not in z.files:
                return None
            X = z[key].astype(np.float32)
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


def cluster_from_matrix(names: list[str], matrix: np.ndarray, n_clusters: int) -> dict:
    """Agglomerative clustering on a precomputed distance = 1 - |similarity|."""
    from sklearn.cluster import AgglomerativeClustering

    dist = 1.0 - np.abs(np.nan_to_num(matrix, nan=0.0))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    k = min(n_clusters, len(names))
    labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(dist)
    clusters: dict[str, list[str]] = {}
    for name, lab in zip(names, labels):
        clusters.setdefault(f"cluster_{int(lab)}", []).append(name)
    internal = {}
    for cname, members in clusters.items():
        idx = [names.index(m) for m in members]
        vals = [matrix[a][b] for a in idx for b in idx if a < b]
        internal[cname] = [round(float(min(vals)), 3), round(float(max(vals)), 3)] if vals else None
    return {"clusters": clusters, "internal_similarity_range": internal}


def reliability_ceiling(run_dir, slug, position, layer, split, n_splits, seed, C, max_iter, drop_degenerate):
    """Split-half cosine between two probe directions fit on disjoint prompt halves.

    This is how well the direction is estimated at all. Cross-cell cosines are
    uninterpretable without it. Halves are split by prompt_id, never by row, so
    a prompt's two polarities stay together.
    """
    from train_probes import fit_probe

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


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def heatmap(path, matrix, row_names, col_names, title, vmin, vmax, cmap, center_note=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1 + 0.55 * len(col_names), 1 + 0.5 * len(row_names)))
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(col_names)), col_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(row_names)), row_names, fontsize=7)
    ax.set_title(title + ("\n" + center_note if center_note else ""), fontsize=9)
    for a in range(len(row_names)):
        for b in range(len(col_names)):
            v = matrix[a][b]
            if v is not None and v == v:
                ax.text(b, a, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def anova_plot(path, decomposition, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["prompt_pair", "layer", "position", "residual"]
    vals = [decomposition[k]["pct_variance"] for k in labels if k in decomposition]
    labels = [k for k in labels if k in decomposition]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(labels, vals, color=["#3b6ea5", "#7ba7d7", "#b8cfe6", "#cccccc"][: len(labels)])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_ylabel("variance explained (%)")
    ax.set_ylim(0, max(vals) * 1.2 + 5)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def as_matrix(nested: dict, rows: list[str], cols: list[str]) -> np.ndarray:
    return np.array([[nested.get(r, {}).get(c, np.nan) for c in cols] for r in rows], dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--positions", type=str, nargs="+", default=None, choices=common.POSITIONS)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--n-clusters", type=int, default=5, help="Paper found 5 clusters over 23 prompts.")
    parser.add_argument(
        "--cluster-basis",
        default=ELEPHANT_BASIS,
        help="Sample set for the score-correlation clustering: 'elephant' (the paper's "
        "'all evaluation samples') or a cell slug such as 'neutral'.",
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
    if not ood_path.exists():
        print()
        print(f"--- ANOVA skipped: {ood_path} not found; run eval_elephant.py first ---")
    else:
        ood = json.loads(ood_path.read_text(encoding="utf-8"))
        anova_rows = [dict(r, spec=r["slug"]) for r in ood["rows"] if r["split"] == "eval"]
        decomposition = anova(anova_rows, "auc")
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

            tm = transfer_matrix(run_dir, slugs, probes, position, layer, split, drop_degenerate)
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

            la = length_analysis(run_dir, slugs, probes, position, layer, drop_degenerate)
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

            sc = score_correlations(run_dir, slugs, probes, position, layer, args.cluster_basis)
            if sc is None and args.cluster_basis == ELEPHANT_BASIS:
                sc = score_correlations(run_dir, slugs, probes, position, layer, common.NEUTRAL_SLUG)
                if sc is not None:
                    print("  NOTE: no ELEPHANT activations; clustered on the in-distribution")
                    print("  'neutral' cell instead, where probes saturate. The paper correlates")
                    print("  over evaluation samples -- run eval_elephant.py for that version.")
            if sc:
                corr = np.array(sc["correlation"], dtype=float)
                sc["clustering"] = cluster_from_matrix(sc["cells"], corr, args.n_clusters)
                (out_dir / f"score_correlation_{tag}.json").write_text(json.dumps(sc, indent=2), encoding="utf-8")
                heatmap(
                    plot_dir / f"score_correlation_{tag}.png",
                    corr,
                    sc["cells"],
                    sc["cells"],
                    f"Probe score correlation on {sc['basis']} ({tag})",
                    -1.0,
                    1.0,
                    "RdBu_r",
                    f"Pearson r over {sc['n_samples']} common samples",
                )
                print(f"  score-correlation clusters (basis={sc['basis']}, n={sc['n_samples']}):")
                for cname, members in sorted(sc["clustering"]["clusters"].items()):
                    rng_ = sc["clustering"]["internal_similarity_range"][cname]
                    print(f"    {cname}: {members}  internal r {rng_}")
            else:
                print(
                    f"  score correlation skipped: no '{common.NEUTRAL_SLUG}' activations. "
                    "Generate that cell to get a common evaluation basis and control-adjusted scores."
                )

            key = ga.act_key(position, layer)
            dirs = {s: probes[s][key]["direction_raw"] for s in slugs if key in probes.get(s, {})}
            if len(dirs) >= 2:
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
