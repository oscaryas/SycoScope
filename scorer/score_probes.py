#!/usr/bin/env python3
"""
Probe loading/scoring for the new pickled-weights format, plus the cross-probe
analyses that still make sense on it. CPU only, no model weights.

Probe format: the pickle written by analyze_probes/train_probes.py (or any
cache_train_*.py) via probes_core.pickle_weights:

    {"meta": {...}, "layers": {layer: {"best_C": C,
                                       "by_C": {C: {"model", "scaler", "direction_raw", ...}}}}}

Scoring uses the pickled StandardScaler + LogisticRegression exactly:
decision_function(scaler.transform(X)), i.e. the signed distance from the
boundary. `best_C` (highest grouped-CV accuracy) is used unless a C is given.

Analyses (following Natarajan et al. 2026), over N probes on ONE common
activation cache (probes_core.save_cache format, e.g. an eval_* cache):

1. Score-correlation clustering (their section 5.3). Pearson correlation
   between probe *scores* over the common sample set, then hierarchical
   clustering (signed primary, unsigned supplementary, silhouette sweep).
2. Cosine matrix between probe directions (direction_raw), with split-half
   reliability ceilings refit on each probe's own training cache
   (meta["cache"], when it still exists). A cross-cosine of 0.4 against a
   per-probe ceiling of 0.45 means "the same direction".
3. Optional ANOVA variance decomposition (their section 4.1) of eval-half AUC
   over probe x layer x position, read from scorer/eval_*.py result JSONs.

The old multi-cell analyses (transfer matrix over a shared prompt_split,
per-cell response-length analysis, run-dir eval_* basis discovery) need the
retired probes/<cell>/probes.npz + activations/<cell>.npz layout and live in
misc/score_probes_legacy.py.

Usage:
    python -m scorer.score_probes --probe a=weights_a.pkl b=weights_b.pkl c.pkl \\
        --cache results/probes/scores/eval_moral/cache.npz
    python -m scorer.score_probes --anova results/probes/scores/eval_social_sycophancy.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyze_probes import probes_core as core  # noqa: E402

SCORES_DIR = REPO_ROOT / "results" / "probes" / "scores"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Same convention as rq1_gate1_geometry.cosine."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------------------------------------------------------------------------
# Probe loading / scoring (new pickled format)
# ---------------------------------------------------------------------------


def load_probes(path: Path) -> dict:
    """The pickled payload {"meta", "layers": {layer: {"best_C", "by_C"}}}."""
    payload = core.load_weights(Path(path))
    if not isinstance(payload, dict) or "layers" not in payload:
        raise ValueError(f"{path}: not a probes_core.pickle_weights payload (no 'layers')")
    payload["layers"] = {int(layer): entry for layer, entry in payload["layers"].items()}
    return payload


def parse_probe_spec(spec: str) -> tuple[str, Path]:
    """`name=path` or bare `path` (name = file stem)."""
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name, Path(path)
    return Path(spec).stem, Path(spec)


def load_probe_set(specs: list[str]) -> dict[str, dict]:
    """{name: payload} from --probe arguments; names must be unique."""
    out = {}
    for spec in specs:
        name, path = parse_probe_spec(spec)
        if name in out:
            raise SystemExit(f"duplicate probe name {name!r}; use name=path")
        out[name] = load_probes(path)
    return out


def probe_layers(payload: dict) -> list[int]:
    return sorted(payload["layers"])


def select_probe(payload: dict, layer: int, C: float | None = None) -> dict | None:
    """The by_C entry ({"model", "scaler", "direction_raw", ...}) for this
    layer at `C`, or at the layer's best_C when C is None. None if absent."""
    entry = payload["layers"].get(int(layer))
    if entry is None:
        return None
    by_C = {float(c): v for c, v in entry["by_C"].items()}
    c = float(entry["best_C"]) if C is None else float(C)
    if c not in by_C:
        raise KeyError(f"C={c:g} not trained for layer {layer} (have {sorted(by_C)})")
    return dict(by_C[c], C=c)


def apply_probe(probe: dict, X: np.ndarray) -> np.ndarray:
    """Signed distance from the boundary via the pickled scaler + model."""
    return core.score(probe["scaler"], probe["model"], np.asarray(X, dtype=np.float32))


def probe_model(payload: dict) -> str | None:
    """Model the probe was trained on, from its meta or its training cache's
    .meta.json sidecar (cache_activations.py writes one)."""
    meta = payload.get("meta") or {}
    if meta.get("model"):
        return meta["model"]
    cache = meta.get("cache")
    if cache:
        sidecar = Path(cache).with_name(Path(cache).name + ".meta.json")
        if sidecar.exists():
            return json.loads(sidecar.read_text(encoding="utf-8")).get("model")
    return None


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------


def anova(rows: list[dict], value_key: str = "auc", factors: dict | None = None) -> dict:
    """Main-effect variance decomposition over a fully crossed design.

    One observation per (probe, position, layer) cell means interactions cannot
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

    factors = factors or {"probe": "slug", "layer": "layer", "position": "position"}
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


def anova_rows_from_results(paths: list[Path], null_slugs=()) -> list[dict]:
    """Eval-half, dataset-pooled AUC rows from scorer/eval_*.py result JSONs."""
    rows = []
    for path in paths:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        rows += [
            r for r in result["rows"]
            if r["split"] == "eval" and r.get("dataset", "all") == "all" and r["slug"] not in null_slugs
        ]
    return rows


# ---------------------------------------------------------------------------
# Score correlation / clustering / geometry
# ---------------------------------------------------------------------------


def score_correlations(probes: dict, X: np.ndarray, layer: int, basis: str, C: float | None = None):
    """Pearson correlation between probe scores over one common sample set.

    Use a real-world OOD cache (an eval_* cache) as the basis -- what the paper
    does in section 5.3. Correlations measured in-distribution are computed
    where every probe saturates, so they describe behaviour on data that
    cannot discriminate between them.
    """
    names, scores = [], []
    for name, payload in probes.items():
        probe = select_probe(payload, layer, C)
        if probe is None:
            continue
        names.append(name)
        scores.append(apply_probe(probe, X))
    if len(names) < 2:
        return None
    corr = np.corrcoef(np.vstack(scores))
    return {
        "basis": basis,
        "layer": int(layer),
        "n_samples": int(X.shape[0]),
        "cells": names,
        "correlation": corr.tolist(),
        "control_median": {n: float(np.median(s)) for n, s in zip(names, scores)},
    }


def cluster_from_matrix(names: list[str], matrix: np.ndarray, n_clusters: int, signed: bool = False) -> dict:
    """Agglomerative clustering on a precomputed distance.

    signed=False (default): distance = 1 - |similarity|. Groups probes with
    strongly *opposite* scores into the same cluster (same axis, either
    direction). Useful as a supplementary "same axis regardless of sign" view,
    but must be labeled as such -- see V1_ANALYSIS_PLAN.md step 3.

    signed=True: distance = 1 - similarity (range [0, 2]). This is the primary
    view for score-correlation clustering: two probes that fire in opposite
    directions on the same examples (e.g. the calibrated-hedging inversion,
    see FINDINGS.md section 4) land in *different* clusters, which is the
    behaviourally correct read -- they disagree on most examples.
    """
    from sklearn.cluster import AgglomerativeClustering

    m = np.nan_to_num(matrix, nan=0.0)
    dist = (1.0 - m) if signed else (1.0 - np.abs(m))
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
    return {"clusters": clusters, "internal_similarity_range": internal, "signed": signed}


def cluster_sweep(names: list[str], matrix: np.ndarray, k_range: range, signed: bool = False) -> dict:
    """Cluster count left free to explore (V1 plan step 3), not fixed at 5.

    Reports, for each k, the clustering and its silhouette score (on the same
    precomputed distance), so a stable k can be picked from where silhouette
    peaks rather than assumed in advance.
    """
    from sklearn.metrics import silhouette_score

    m = np.nan_to_num(matrix, nan=0.0)
    dist = (1.0 - m) if signed else (1.0 - np.abs(m))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    out = {}
    for k in k_range:
        if k < 2 or k >= len(names):
            continue
        res = cluster_from_matrix(names, matrix, k, signed=signed)
        labels = [
            next(int(lab.split("_")[1]) for lab, members in res["clusters"].items() if name in members)
            for name in names
        ]
        try:
            sil = float(silhouette_score(dist, labels, metric="precomputed"))
        except ValueError:
            sil = None
        out[str(k)] = {"silhouette": sil, **res}
    return out


def reliability_ceiling(payload: dict, layer: int, n_splits: int, seed: int, C: float | None = None):
    """Split-half cosine between two directions fit on disjoint group halves of
    the probe's own training cache (payload["meta"]["cache"]).

    This is how well the direction is estimated at all; cross-probe cosines are
    uninterpretable without it. Halves are split by CV group (prompt/row id),
    never by row, so a prompt's paired responses stay together. None when the
    training cache is not recorded or no longer exists.
    """
    meta = payload.get("meta") or {}
    cache_path = meta.get("cache")
    if not cache_path or not Path(cache_path).exists():
        return None
    cache = core.load_cache(Path(cache_path))
    if int(layer) not in cache["acts"]:
        return None
    X, y, groups = cache["acts"][int(layer)], cache["y"], cache["groups"]
    c = float(payload["layers"][int(layer)]["best_C"]) if C is None else float(C)
    max_iter = int(meta.get("max_iter", 2000))
    uniq = sorted(set(groups.tolist()))
    if len(uniq) < 4:
        return None

    rng = np.random.default_rng(seed)
    cosines = []
    for _ in range(n_splits):
        order = rng.permutation(len(uniq))
        half = len(uniq) // 2
        halves = ({uniq[i] for i in order[:half]}, {uniq[i] for i in order[half : 2 * half]})
        dirs = []
        for group in halves:
            m = np.array([g in group for g in groups], dtype=bool)
            if len(np.unique(y[m])) < 2:
                dirs = []
                break
            scaler, clf = core.fit_probe(X[m], y[m], seed, c, max_iter)
            raw = clf.coef_[0] / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
            dirs.append(raw / (np.linalg.norm(raw) + 1e-12))
        if len(dirs) == 2:
            cosines.append(cosine(dirs[0], dirs[1]))
    if not cosines:
        return None
    return {"mean": float(np.mean(cosines)), "min": float(np.min(cosines)), "max": float(np.max(cosines)),
            "n": len(cosines)}


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


def correlation_dendrogram(path, matrix, names, title):
    """Average-linkage tree using the same signed distance as primary clustering."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    dist = 1.0 - np.asarray(matrix, dtype=float)
    dist = np.maximum((dist + dist.T) / 2.0, 0.0)
    np.fill_diagonal(dist, 0.0)
    tree = linkage(squareform(dist, checks=True), method="average")
    fig, ax = plt.subplots(figsize=(11, 7))
    dendrogram(
        tree, labels=names, orientation="right", leaf_font_size=9,
        color_threshold=0, above_threshold_color="#3b6ea5", ax=ax,
    )
    ax.set_xlabel("Average-linkage distance (1 - signed Pearson r)")
    ax.set_title(title + "\nTree shown without a fixed cluster cut", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def anova_plot(path, decomposition, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [k for k in decomposition if not k.startswith("_")]  # factors in order, then residual
    vals = [decomposition[k]["pct_variance"] for k in labels]
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_anova(paths: list[Path], out_dir: Path) -> None:
    from utils import common

    decomposition = anova(anova_rows_from_results(paths, common.NULL_SLUGS), "auc")
    (out_dir / "anova.json").write_text(
        json.dumps({"source": [str(p) for p in paths], **decomposition}, indent=2), encoding="utf-8"
    )
    print("\n--- ANOVA on eval-half AUC (paper: prompt 70.6%, layer 2.7%, token selection 0.6%) ---")
    if "error" in decomposition:
        print(f"  skipped: {decomposition['error']}")
        return
    for label in ("probe", "layer", "position", "residual"):
        e = decomposition.get(label)
        if not e:
            continue
        pv = e.get("p")
        print(f"  {label:<12} {e['pct_variance']:>6.2f}%  (df={e['df']})" + ("" if pv is None else f"  p={pv:.3g}"))
    anova_plot(out_dir / "anova_auc.png", decomposition, "AUC variance explained")


def run_cross_probe(probes: dict, cache_path: Path, out_dir: Path, args) -> None:
    cache = core.load_cache(cache_path)
    basis = cache_path.stem
    layers = sorted(set(cache["layers"]) & {layer for p in probes.values() for layer in probe_layers(p)})
    if not layers:
        raise SystemExit(f"no probe layer is cached in {cache_path} (cache has {cache['layers']})")
    for layer in layers:
        tag = f"L{layer:02d}"
        print(f"\n=== {basis} {tag} ===")
        sc = score_correlations(probes, cache["acts"][layer], layer, basis, args.C)
        if sc is None:
            print("  score correlation skipped: fewer than two probes at this layer")
        else:
            corr = np.array(sc["correlation"], dtype=float)
            # Signed is the primary view (V1 plan step 3): opposite-scoring
            # probes must NOT land in the same cluster.
            sc["clustering_signed"] = cluster_from_matrix(sc["cells"], corr, args.n_clusters, signed=True)
            sc["clustering_unsigned"] = cluster_from_matrix(sc["cells"], corr, args.n_clusters, signed=False)
            sc["cluster_sweep_signed"] = cluster_sweep(sc["cells"], corr, range(2, min(10, len(sc["cells"]))),
                                                       signed=True)
            (out_dir / f"score_correlation_{tag}.json").write_text(json.dumps(sc, indent=2), encoding="utf-8")
            heatmap(out_dir / f"score_correlation_{tag}.png", corr, sc["cells"], sc["cells"],
                    f"Probe score correlation on {basis} ({tag})", -1.0, 1.0, "RdBu_r",
                    f"Pearson r over {sc['n_samples']} common samples")
            if len(sc["cells"]) >= 3:
                correlation_dendrogram(out_dir / f"dendrogram_{tag}.png", corr, sc["cells"],
                                       f"Probe score clustering on {basis} ({tag}, n={sc['n_samples']})")
            print(f"  score-correlation clusters, SIGNED (n={sc['n_samples']}):")
            for cname, members in sorted(sc["clustering_signed"]["clusters"].items()):
                print(f"    {cname}: {members}  internal r {sc['clustering_signed']['internal_similarity_range'][cname]}")

        dirs = {}
        for name, payload in probes.items():
            probe = select_probe(payload, layer, args.C)
            if probe is not None:
                dirs[name] = np.asarray(probe["direction_raw"], dtype=np.float64)
        if len(dirs) >= 2:
            names = list(dirs)
            cos = np.array([[cosine(dirs[a], dirs[b]) for b in names] for a in names])
            geometry = {"cells": names, "cosine": cos.tolist(),
                        "clustering": cluster_from_matrix(names, cos, args.n_clusters)}
            if args.ceiling_splits:
                geometry["reliability_ceiling"] = {
                    n: reliability_ceiling(probes[n], layer, args.ceiling_splits, args.seed, args.C) for n in names
                }
            (out_dir / f"geometry_{tag}.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
            heatmap(out_dir / f"cosine_{tag}.png", cos, names, names, f"Probe direction cosine ({tag})",
                    -1.0, 1.0, "RdBu_r", "compare each off-diagonal against the per-probe reliability ceiling")
            ceilings = {k: v for k, v in (geometry.get("reliability_ceiling") or {}).items() if v}
            for n, c in ceilings.items():
                print(f"    ceiling {n:<26} {c['mean']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", nargs="+", default=[], help="Pickled probes: path or name=path (2+).")
    parser.add_argument("--cache", type=Path, default=None, help="Common-basis cache (.npz, probes_core.save_cache).")
    parser.add_argument("--anova", type=Path, nargs="+", default=None, help="scorer/eval_*.py result JSON(s).")
    parser.add_argument("--C", type=float, default=None, help="Default: each layer's best_C.")
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--ceiling-splits", type=int, default=5, help="0 to skip reliability ceilings.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help=f"Output dir (default under {SCORES_DIR}).")
    args = parser.parse_args()
    if not args.anova and not (args.probe and args.cache):
        parser.error("give --probe ... --cache, and/or --anova")

    out_dir = args.output or SCORES_DIR / "score_probes" / (args.cache.stem if args.cache else "anova")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.anova:
        run_anova(args.anova, out_dir)
    if args.probe and args.cache:
        run_cross_probe(load_probe_set(args.probe), args.cache, out_dir, args)
    print(f"\nDone. -> {out_dir}")


if __name__ == "__main__":
    main()
