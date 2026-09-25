#!/usr/bin/env python3
"""Exploratory, question-bootstrap stability of frozen SYCON probe scores.

CPU-only; no API calls, model inference, label-based selection, or probe fitting.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from utils import common
from analyze_probes.analyze_probes import apply_probe, load_probes, heatmap
from analyze_probes.eval_syconbench import SETTINGS


def correlation(values):
    if np.any(np.std(values, axis=0) < 1e-10):
        raise ValueError("constant probe score; cannot interpret its correlation")
    corr = np.clip(np.corrcoef(values.T), -1, 1)
    if not np.isfinite(corr).all():
        raise ValueError("nonfinite score correlation")
    np.fill_diagonal(corr, 1)
    return corr


def partition(corr, unsigned=False):
    dist = np.maximum(0, 1 - (np.abs(corr) if unsigned else corr))
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    tree = linkage(squareform(dist, checks=True), method="average")
    ks = list(range(2, min(10, len(corr))))
    cuts = cut_tree(tree, n_clusters=ks)
    scores = [float(silhouette_score(dist, cuts[:, i], metric="precomputed"))
              for i in range(len(ks))]
    best = int(np.argmax(scores))
    return {"best_k": ks[best], "silhouette": scores[best],
            "labels": cuts[:, best].tolist(), "fixed_k3_labels": cuts[:, ks.index(3)].tolist(),
            "silhouette_by_k": dict(zip(map(str, ks), scores))}


def residualize(values, turns, lengths):
    # Refit in each bootstrap draw. These are descriptive nuisance adjustments,
    # not causal controls: response length may itself carry behavioral signal.
    design = np.column_stack([np.ones(len(turns)),
                              *[(turns == t).astype(float) for t in range(2, 6)],
                              np.log1p(lengths)])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def group_draw(groups, rng):
    unique = np.unique(groups)
    return np.concatenate([np.flatnonzero(groups == g)
                           for g in rng.choice(unique, len(unique), replace=True)])


def geometry(corr):
    eig = np.maximum(np.linalg.eigvalsh(corr)[::-1], 0)
    share = eig / eig.sum()
    return {"pc1_share": float(share[0]), "pc1_3_share": float(share[:3].sum()),
            "participation_rank": float(1 / np.sum(share ** 2))}


def members(labels, names):
    return [[name for name, label in zip(names, labels) if label == k]
            for k in sorted(set(labels))]


def bootstrap(values, groups, turns, lengths, adjusted, n_boot, seed):
    transform = lambda v, t, n: residualize(v, t, n) if adjusted else v
    corr = correlation(transform(values, turns, lengths))
    base = partition(corr)
    rng = np.random.default_rng(seed)
    counts, ari, same, fixed = Counter(), [], np.zeros_like(corr), np.zeros_like(corr)
    for _ in range(n_boot):
        ix = group_draw(groups, rng)
        b = partition(correlation(transform(values[ix], turns[ix], lengths[ix])))
        lab, lab3 = np.array(b["labels"]), np.array(b["fixed_k3_labels"])
        counts[b["best_k"]] += 1
        ari.append(adjusted_rand_score(base["labels"], lab))
        same += lab[:, None] == lab[None, :]
        fixed += lab3[:, None] == lab3[None, :]
    return {**base, **geometry(corr), "correlation": corr.tolist(),
            "bootstrap_k_counts": dict(sorted(counts.items())),
            "bootstrap_ari_median": float(np.median(ari)),
            "bootstrap_ari_95pct_range": np.quantile(ari, [.025, .975]).tolist(),
            "coclustering_selected_k": (same / n_boot).tolist(),
            "coclustering_fixed_k3": (fixed / n_boot).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    if args.n_boot < 1:
        parser.error("--n-boot must be positive")
    run = common.resolve_run_dir(args.run_name, create=False)
    out = run / "analysis/syconbench/clustering_stability"
    out.mkdir(parents=True, exist_ok=True)
    index = common.read_jsonl(run / "eval_syconbench/activations_index.jsonl")
    names = [s for s in common.all_slugs(include_neutral=False)
             if (run / "probes" / s / "probes.npz").exists()]
    probes = {name: load_probes(run, name) for name in names}
    groups = np.array([r["group_id"] for r in index])
    turns = np.array([r["turn"] for r in index])
    lengths = np.array([r["n_response_tokens"] for r in index])
    datasets = np.array([r["dataset"] for r in index])
    results = {}
    with np.load(run / "eval_syconbench/activations.npz") as arrays:
        for key in sorted(k for k in arrays.files if k.startswith("response_")):
            X = arrays[key].astype(np.float32)
            scores = np.column_stack([apply_probe(probes[name][key], X) for name in names])
            for setting in SETTINGS:
                mask = datasets == f"SYCON:{setting}"
                v, g, t, n = scores[mask], groups[mask], turns[mask], lengths[mask]
                tag = f"{setting}_{key}"
                raw = bootstrap(v, g, t, n, False, args.n_boot, args.seed)
                adj = bootstrap(v, g, t, n, True, args.n_boot, args.seed)
                keep = [i for i, name in enumerate(names) if name != "ctrl_calibrated_hedging"]
                ablated = partition(correlation(v[:, keep]))
                means = np.array([v[g == q].mean(axis=0) for q in np.unique(g)])
                mean_partition = partition(correlation(means))
                per_turn = {}
                for turn in range(1, 6):
                    p = partition(correlation(v[t == turn]))
                    per_turn[str(turn)] = {**p, "ari_vs_all_turns": float(adjusted_rand_score(raw["labels"], p["labels"]))}
                for result in (raw, adj):
                    result["clusters"] = members(result["labels"], names)
                results[tag] = {
                    "n_turns": len(v), "n_questions": len(np.unique(g)), "raw": raw,
                    "turn_and_log_length_adjusted": adj,
                    "raw_adjusted_ari": float(adjusted_rand_score(raw["labels"], adj["labels"])),
                    "unsigned_same_axis": partition(correlation(v), unsigned=True),
                    "without_calibrated_hedging": {**ablated, "clusters": members(ablated["labels"], [names[i] for i in keep]),
                        "ari_vs_raw_restricted": float(adjusted_rand_score(np.array(raw["labels"])[keep], ablated["labels"]))},
                    "question_mean": {**mean_partition, "ari_vs_all_turns": float(adjusted_rand_score(raw["labels"], mean_partition["labels"]))},
                    "per_turn": per_turn,
                }
                print(f"{tag}: k={raw['best_k']}, adjusted k={adj['best_k']}, "
                      f"bootstrap ARI={raw['bootstrap_ari_median']:.3f}", flush=True)
                if key in ("response_L12", "response_L16", "response_L20"):
                    heatmap(out / f"{tag}_bootstrap_selected_k.png", np.array(raw["coclustering_selected_k"]), names, names,
                            f"SYCON {setting}, {key}: question-bootstrap co-clustering", 0, 1, "viridis",
                            f"{args.n_boot} question resamples; k reselected by silhouette in each draw")
                    heatmap(out / f"{tag}_bootstrap_k3.png", np.array(raw["coclustering_fixed_k3"]), names, names,
                            f"SYCON {setting}, {key}: question-bootstrap co-clustering", 0, 1, "viridis",
                            f"{args.n_boot} question resamples; fixed k=3 sensitivity, not estimated concept count")
                (out / "results.json").write_text(json.dumps({"n_boot": args.n_boot, "seed": args.seed,
                    "cells": names, "configs": results}, indent=2) + "\n")
    rows = ["# SYCON-Bench cluster stability: numerical summary", "",
            "Full-response-average scores; signed Pearson distance 1−r; average linkage; silhouette sweep k=2…9.", "",
            "Question bootstrap retains all turns of each sampled question. ARI compares each reselected bootstrap partition to the original partition; 1 is identical up to cluster numbering. The bootstrap range is descriptive, not a performance CI.", "",
            "| Setting / layer | Best k | Sizes | Silhouette | Bootstrap same k | Median ARI | Adjusted k | Raw–adjusted ARI | PC1 | Participation rank |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, r in results.items():
        a, b = r["raw"], r["turn_and_log_length_adjusted"]
        rate = a["bootstrap_k_counts"].get(a["best_k"], 0) / args.n_boot
        rows.append(f"| {tag} | {a['best_k']} | {' + '.join(map(str, map(len, a['clusters'])))} | {a['silhouette']:.3f} | {rate:.0%} | {a['bootstrap_ari_median']:.3f} | {b['best_k']} | {r['raw_adjusted_ari']:.3f} | {a['pc1_share']:.0%} | {a['participation_rank']:.2f} |")
    rows += ["", "## Interpretation limits", "",
             "All available questions are used descriptively, without judge-label selection. Layers reuse questions and are not independent replications. Score correlation describes probe behavior on this dataset, not orthogonality or causal circuits. Best k is conditional on searching 2–9: this procedure cannot establish that any discrete clusters exist.", "",
             "The adjusted view removes linear effects of turn indicators and log(1 + response tokens), refitted inside each bootstrap sample. It does not eliminate all wording, topic, style, or system-prompt confounds, and may remove genuine behavior-associated signal. Question means average probe scores across turns, not token activations across a conversation.", "",
             "Unsigned distance 1−|r| is only a same-axis sensitivity check. Participation rank is (sum eigenvalues)²/sum(eigenvalues²) of the score-correlation matrix, not an estimate of the number of sycophancy concepts. Fixed k=3 co-clustering deliberately forces three groups and is not evidence for three concepts.", "",
             f"Reproduce: `.venv/bin/python misc/cluster_syconbench_stability.py --run-name {args.run_name} --n-boot {args.n_boot} --seed {args.seed}`", "",
             "Full memberships, correlation matrices, bootstrap consensus, per-turn, unsigned, ablation, and question-mean sensitivities are in [results.json](results.json)."]
    (out / "SUMMARY.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
