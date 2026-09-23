#!/usr/bin/env python3
"""
Two standalone local PNGs for google/gemma-4-12B-it, no Artifact/HTML:

1. Behavioral sycophancy rate per category (all 9 datasets generated for
   gemma this session/prior sessions) -- reuses the exact label definitions
   already verified against this session's sycophancy_rates.html artifact
   (OEQ/SS/AITA-YTA: "validation" metric; AITA-NTA-FLIP: both-NTA;
   sypr/truthfulqa/are_you_sure_*: each dataset's own summary.json rate;
   dissociating_sycophancy: CAVED rate).
2. The mixture residual probe's held-out/OOD transfer-eval accuracy per
   target (from probing/data/google__gemma-4-12B-it/gemma_mixture_probe_transfer/
   results.json), with the in-distribution training-mixture CV accuracy
   (83.1%, layer 31) as a reference line, plus AUC-ROC as a second series
   where defined.

Usage:
    python -m probing.analyze_probes.plot_gemma_summary_charts
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
GEMMA_DIR = REPO_ROOT / "probing" / "data" / "google__gemma-4-12B-it"
TRANSFER_DIR = GEMMA_DIR / "gemma_mixture_probe_transfer"
PLOTS_DIR = GEMMA_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette, house style for this task's plots.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
GRAY = "#999999"


def load_json(path):
    return json.loads(path.read_text())


def plot_behavioral_rates():
    oeq = [json.loads(l) for l in open(GEMMA_DIR / "oeq" / "judged.jsonl")]
    ss = [json.loads(l) for l in open(GEMMA_DIR / "ss" / "judged.jsonl")]
    aita_yta = [json.loads(l) for l in open(GEMMA_DIR / "aita_yta" / "judged.jsonl")]
    aita_flip = [json.loads(l) for l in open(GEMMA_DIR / "aita_nta_flip" / "judged.jsonl")]
    sypr = load_json(GEMMA_DIR / "sypr_merged" / "summary.json")
    tqa = load_json(GEMMA_DIR / "truthfulqa" / "summary.json")
    ays_ff = load_json(GEMMA_DIR / "are_you_sure_freeform" / "summary.json")
    ays_mc = load_json(GEMMA_DIR / "are_you_sure_mc" / "summary.json")
    dis_path = GEMMA_DIR / "dissociating_sycophancy" / "summary.json"

    def validation_rate(rows):
        valid = [r for r in rows if r["validation"] is not None]
        return sum(r["validation"] == 1 for r in valid), len(valid)

    both_nta = sum(1 for r in aita_flip if r["original_post_verdict"] == "NTA" and r["flipped_story_verdict"] == "NTA")

    bars = [
        ("OEQ\n(validation)", *validation_rate(oeq), BLUE),
        ("SS\n(validation)", *validation_rate(ss), BLUE),
        ("AITA-YTA\n(validation)", *validation_rate(aita_yta), BLUE),
        ("AITA-NTA-FLIP\n(both-NTA)", both_nta, len(aita_flip), ORANGE),
        ("SyPR\n(sycophantic praise)", sypr["n_sycophantic_praise"], sypr["n_judged"], GREEN),
        ("TruthfulQA\n(false-imitative)", tqa["n_false_imitative"], tqa["n_judged"], VERMILLION),
        ("Are-You-Sure\nfreeform (caved)", ays_ff["n_sycophantic_caved"], ays_ff["n_judged"], GRAY),
        ("Are-You-Sure\nMC (caved)", ays_mc["n_sycophantic_caved"], ays_mc["n_judged"], GRAY),
    ]
    if dis_path.exists():
        dis = load_json(dis_path)
        bars.append(("Dissociating-\nSycophancy (caved)", dis["n_pos"], dis["n_judged"], "#CC79A7"))

    labels = [b[0] for b in bars]
    rates = [b[1] / b[2] for b in bars]
    ns = [b[2] for b in bars]
    colors = [b[3] for b in bars]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(labels))
    bars_obj = ax.bar(x, rates, color=colors, width=0.65)
    for i, (rate, n) in enumerate(zip(rates, ns)):
        ax.text(i, rate + 0.015, f"{rate:.1%}\n(n={n})", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Sycophancy rate")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("google/gemma-4-12B-it: sycophancy rate by dataset", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / "gemma_sycophancy_rates_by_category.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_ood_transfer():
    results_path = TRANSFER_DIR / "results.json"
    if not results_path.exists():
        print(f"SKIP: {results_path} not found")
        return
    data = load_json(results_path)
    in_dist_acc = 0.831  # layer-31 5-fold CV accuracy on the training mixture itself

    order = ["sypr", "social", "are_you_sure", "truthfulqa", "moral"]
    labels, accs, cis, aucs, ns, npos = [], [], [], [], [], []
    for name in order:
        t = data["targets"][name]["scores"]["__all__"]
        labels.append(name)
        accs.append(t["accuracy"])
        cis.append((t["accuracy"] - t["ci"][0], t["ci"][1] - t["accuracy"]))
        aucs.append(t["auc_roc"])
        ns.append(data["targets"][name]["n_total"])
        npos.append(data["targets"][name]["n_pos"])

    fig, ax = plt.subplots(figsize=(9, 6))
    x = range(len(labels))
    yerr = list(zip(*cis))
    colors = [BLUE, GREEN, GRAY, VERMILLION, ORANGE]
    ax.bar(x, accs, yerr=yerr, capsize=4, color=colors, width=0.55, label="Accuracy (held-out)")
    ax.axhline(in_dist_acc, color="black", linestyle="--", linewidth=1.2, label=f"In-distribution CV accuracy ({in_dist_acc:.1%}, layer 31)")
    ax.axhline(0.5, color=GRAY, linestyle=":", linewidth=1, label="Chance (50%)")

    for i, (acc, auc, n, np_) in enumerate(zip(accs, aucs, ns, npos)):
        auc_str = f"AUC={auc:.2f}" if auc is not None else "AUC=N/A"
        ax.text(i, 0.03, f"n={n}\n(n_pos={np_})\n{auc_str}", ha="center", va="bottom", fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, ec="none"))

    ax.set_xticks(list(x))
    ax.set_xticklabels(["SyPR\n(13 pos, tiny n)", "Social\n(OEQ+SS)", "Are-You-Sure\n(0 pos, degenerate)", "TruthfulQA\n(true zero-shot)", "Moral\n(AITA-NTA-FLIP)"], fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("gemma-4-12B-it mixture residual probe (layer 31): held-out transfer accuracy", fontsize=12.5)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = TRANSFER_DIR / "ood_transfer_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    plot_behavioral_rates()
    plot_ood_transfer()
