#!/usr/bin/env python3
"""Make SYCON setting-specific clusters, fixed middle-layer views and a report.

Requires eval_matrix.py --target syconbench first. No fitting or GPU inference.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common
import get_activations as ga
from analyze_probes import apply_probe, load_probes, cluster_sweep, heatmap, correlation_dendrogram
from eval_common import selection_split, split_key
from eval_syconbench import LABEL_FIELDS, SETTINGS
from train_probes import safe_auc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    run_dir = common.resolve_run_dir(args.run_name, create=False)
    basis = run_dir / "eval_syconbench"
    out = run_dir / "analysis/syconbench"
    out.mkdir(parents=True, exist_ok=True)
    index = common.read_jsonl(basis / "activations_index.jsonl")
    meta = json.loads((basis / "meta.json").read_text())
    matrix = json.loads((run_dir / "analysis/matrix_syconbench.json").read_text())
    split = meta["evaluation_selection"]
    selection = selection_split(index, split["frac"], split["seed"])
    is_sel = np.array([split_key(r) in selection for r in index])
    turns = np.array([r["turn"] for r in index])
    datasets = np.array([r["dataset"] for r in index])
    names = [s for s in common.all_slugs(include_neutral=False)
             if (run_dir / "probes" / s / "probes.npz").exists()]
    probes = {s: load_probes(run_dir, s) for s in names}
    scores, fixed, cluster_results = {}, {}, {}
    with np.load(basis / "activations.npz") as arrays:
        for key in arrays.files:
            X = arrays[key].astype(np.float32)
            scores[key] = np.column_stack([apply_probe(probes[s][key], X) for s in names])
            if not key.startswith("response_"):
                continue
            for setting in SETTINGS:
                mask = datasets == f"SYCON:{setting}"
                values = scores[key][mask]
                if len(values) < 3 or np.any(values.std(axis=0) < 1e-12):
                    continue
                corr = np.corrcoef(values.T)
                sweep = cluster_sweep(names, corr, range(2, min(10, len(names))), signed=True)
                tag = f"{setting}_{key}"
                cluster_results[tag] = {"n_turns": int(mask.sum()), "cells": names,
                                        "correlation": corr.tolist(), "cluster_sweep_signed": sweep}
                heatmap(out / f"{tag}_correlation.png", corr, names, names,
                        f"SYCON {setting}, {key}", -1, 1, "RdBu_r", "All turns; exploratory signed Pearson r")
                correlation_dendrogram(out / f"{tag}_dendrogram.png", corr, names,
                                       f"SYCON {setting}, {key}")
    by_turn = {}
    for field in LABEL_FIELDS:
        y = np.array([-1 if r[field] is None else r[field] for r in index])
        sel_mask, test_mask = is_sel & (y >= 0), (~is_sel) & (y >= 0)
        fixed[field] = {}
        for layer in (12, 16, 20):
            key = ga.act_key("response", layer)
            entries = {}
            for j, name in enumerate(names):
                entries[name] = {"selection_auc": safe_auc(y[sel_mask], scores[key][sel_mask, j]),
                                 "report_auc": safe_auc(y[test_mask], scores[key][test_mask, j])}
            candidates = {s: e for s, e in entries.items() if e["selection_auc"] is not None}
            winner = max(candidates, key=lambda s: candidates[s]["selection_auc"]) if candidates else None
            fixed[field][key] = {"selection_winner": winner, "probes": entries}
        result = matrix["fields"].get(field)
        if not result:
            continue
        winner = result["best_single_probe"]
        config = result["probes"][winner]
        values = scores[ga.act_key(config["position"], config["layer"])][:, names.index(winner)]
        by_turn[field] = {"selected_probe": winner, "turn_number_auc": safe_auc(y[test_mask], turns[test_mask]),
                          "per_turn": {}}
        for turn in range(1, 6):
            mask = test_mask & (turns == turn)
            by_turn[field]["per_turn"][str(turn)] = {
                "n": int(mask.sum()), "n_positive": int((y[mask] == 1).sum()),
                "auc": safe_auc(y[mask], values[mask])}
    detail = {"fixed_response_views": fixed, "per_turn": by_turn, "clusters": cluster_results}
    (out / "details.json").write_text(json.dumps(detail, indent=2))
    lines = ["# SYCON-Bench-derived OOD results", "",
             "Frozen 20-probe Llama-3.1-8B suite; fresh base-scenario generations, actual five-turn histories, Claude Sonnet 5 labels.", "",
             "Configurations and the winning probe are chosen on selection questions; AUCs below use disjoint report questions. CIs bootstrap questions, retaining all related turns.", "",
             "| Target | Selected probe | Position/layer | Report AUC (95% CI) | Turns / questions |", "|---|---|---|---|---|"]
    for field in LABEL_FIELDS:
        result = matrix["fields"].get(field)
        if not result:
            lines.append(f"| {field} | Not estimable | — | — | — |")
            continue
        winner = result["best_single_probe"]
        e = result["probes"][winner]
        ci = f"{e['ci_lo']:.3f}–{e['ci_hi']:.3f}" if e['ci_lo'] is not None else "unavailable"
        lines.append(f"| {field} | {winner} | {e['position']} L{e['layer']} | {e['auc']:.3f} ({ci}) | {result['n_eval']} / {result['n_eval_groups']} |")
    lines.extend(["", "## Fixed response-average middle-layer views", "",
                  "Each cell uses the probe selected on selection questions at that fixed layer. These are sensitivity views, not additional untouched replications.", "",
                  "| Target | L12 | L16 | L20 |", "|---|---|---|---|"])
    for field in LABEL_FIELDS:
        cells = []
        for layer in (12, 16, 20):
            r = fixed[field][ga.act_key("response", layer)]
            winner = r["selection_winner"]
            auc = r["probes"][winner]["report_auc"] if winner else None
            cells.append(f"{auc:.3f} ({winner})" if auc is not None else "not estimable")
        lines.append(f"| {field} | " + " | ".join(cells) + " |")
    stability_note = ("The subsequent [cluster analysis](CLUSTER_ANALYSIS.md) adds question-bootstrap stability and nuisance/sensitivity checks."
                      if (out / "CLUSTER_ANALYSIS.md").exists()
                      else "Cluster stability has not been bootstrapped.")
    lines.extend(["", "## Interpretation and artifacts", "",
                  "Current-turn failure includes initial failure. First-failure targets include only turns 2–5 after an entirely successful prior history. Debate failure means not maintaining an assigned stance; it is not necessarily factual error or unjustified agreement.", "",
                  "Response-average activations cover only the current answer. They measure detection/decodability, not prediction before that answer. See per-turn AUCs and the turn-number baseline in details.json; changing failure prevalence over turns can inflate pooled AUC.", "",
                  "The separate setting-specific correlation heatmaps/dendrograms use all available turns and are exploratory. Layers reuse the same questions. Cluster counts are not independent concepts or circuits. " + stability_note, "",
                  "Source reproduction and judge caveats are documented in prompt_probes/SYCONBENCH_EVAL.md. Main matrix: analysis/matrix_syconbench.json. Detailed middle-layer, per-turn, and clustering results: analysis/syconbench/details.json."])
    path = out / "RESULTS.md"
    path.write_text("\n".join(lines) + "\n")
    print(path)


if __name__ == "__main__":
    main()
