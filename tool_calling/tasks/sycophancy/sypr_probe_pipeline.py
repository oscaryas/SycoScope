#!/usr/bin/env python3
"""
End-to-end probe-based sycophancy-direction pipeline sourced from SyPR
(vennemeyerd/sycophantic-praise), including its in-domain steering check --
previously notebook-only (sypr_probes_colab_standalone.ipynb), ported here as
a standalone script for the same reasons oeq_probe_pipeline.py replaced its
own notebook predecessor: that notebook already failed to deliver full
artifacts once (truncated generations.jsonl, single-direction .pt files),
and raising its held-out size further inside that same fragile medium would
make it worse, not better.

Held-out set (n=100+, up from the notebook's n=30) is sampled via
sypr_data.sample_poor_quality_heldout, guaranteed disjoint from whichever
rows were used to find the direction (sypr_data.generate_and_label_sypr's
sampled_indices) and guaranteed poor-quality by construction, so the raw
post-steering praise rate on it directly IS the in-domain sycophancy rate --
no ground-truth lookup needed at judge time.

Usage:
    python sypr_probe_pipeline.py \
        --n-train 120 --n-heldout 100 --cross-dataset-n 50 \
        --output-dir results/sypr_probe --model meta-llama/Meta-Llama-3-8B-Instruct
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import anthropic

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for p in (HERE, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.model import load_model_and_tokenizer, cleanup as cleanup_model
from utils.inference import build_chat_prompt_multiturn
from sycophancy_model_registry import get_model_config
from sycophancy_probes import (
    collect_activations,
    train_mha_probes,
    train_mlp_probes,
    train_residual_probes,
    save_probe_results,
    load_probe_results,
)
from sycophancy_steering import ActivationSteerer, load_direction_vectors
from sycophantic_praise_judge import judge_praise_batch, JUDGE_MODEL
from sypr_data import (
    load_sypr_dataset,
    generate_and_label_sypr,
    sample_poor_quality_heldout,
    build_chat_messages,
)
from cross_dataset_generalization import DEFAULT_ALPHAS as ALPHAS, run_generalization_sweep

BALANCE_METHODS = ["undersample", "upweight"]


# ---------------------------------------------------------------------------
# Labeling + probe training (activations extracted once, both balance
# variants trained off the same cached activations)
# ---------------------------------------------------------------------------

def build_labels(model, tokenizer, model_config: dict, n_train: int, seed: int) -> dict:
    return generate_and_label_sypr(model, tokenizer, model_config, n_train=n_train, seed=seed)


def build_indomain_heldout(dataset, sampled_indices: list, n_heldout: int, seed: int) -> list:
    return sample_poor_quality_heldout(dataset, sampled_indices, n_heldout, seed=seed)


def train_all_variants(model, tokenizer, texts: list, labels: np.ndarray, model_config: dict, output_dir: Path) -> dict:
    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]

    print(f"Extracting activations for {len(texts)} labeled examples...")
    activations = collect_activations(model, tokenizer, texts, model_config)

    variant_metadata = {}
    for balance_method in BALANCE_METHODS:
        print(f"\n=== Training probes: balance_method={balance_method} ===")
        mha_acc, mha_ci, mha_states, mha_auc, mha_auc_ci = train_mha_probes(
            activations["mha"], labels, n_layers, n_heads, balance_method=balance_method
        )
        mlp_acc, mlp_ci, mlp_states, mlp_auc, mlp_auc_ci = train_mlp_probes(
            activations["mlp"], labels, n_layers, balance_method=balance_method
        )
        res_acc, res_ci, res_states, res_auc, res_auc_ci = train_residual_probes(
            activations["residual"], labels, n_layers, balance_method=balance_method
        )
        results = {
            "mha_accuracy": mha_acc, "mha_ci": mha_ci, "mha_states": mha_states, "mha_auc": mha_auc, "mha_auc_ci": mha_auc_ci,
            "mlp_accuracy": mlp_acc, "mlp_ci": mlp_ci, "mlp_states": mlp_states, "mlp_auc": mlp_auc, "mlp_auc_ci": mlp_auc_ci,
            "residual_accuracy": res_acc, "residual_ci": res_ci, "residual_states": res_states, "residual_auc": res_auc, "residual_auc_ci": res_auc_ci,
        }
        variant_dir = output_dir / f"sypr_probe_{balance_method}"
        metadata = save_probe_results(results, str(variant_dir), model_name=model_config.get("model_name", ""))
        variant_metadata[balance_method] = {"dir": variant_dir, "metadata": metadata}

    del activations
    gc.collect()
    return variant_metadata


# ---------------------------------------------------------------------------
# In-domain steering sweep
# ---------------------------------------------------------------------------

def _judge_sypr_praise_rate(records: list, judge_model: str = JUDGE_MODEL) -> tuple:
    """records: [{"utterance_text","response"}, ...]. Every row is already known
    poor-quality (sample_poor_quality_heldout's construction), so raw praise
    rate among successfully-judged records IS the in-domain sycophancy rate --
    matches the notebook's sypr_indomain_rate logic exactly."""
    verdicts = judge_praise_batch(records, judge_model=judge_model)
    judged = [v for v in verdicts if v is not None]
    n_judged = len(judged)
    n_praised = sum(v == 1 for v in judged)
    return (n_praised / n_judged if n_judged else 0.0), n_judged


def run_indomain_sweep(
    steerer: ActivationSteerer, heldout_rows: list, component: str, layer: int, head, vector,
    alphas: list, generations_log: list, direction_name: str, max_new_tokens: int = 200, batch_size: int = 16,
) -> dict:
    prompts = [build_chat_prompt_multiturn(steerer.tokenizer, build_chat_messages(r)) for r in heldout_rows]
    per_alpha = {}
    for alpha in alphas:
        if alpha != 0.0:
            steerer.attach(component, layer, vector, alpha, head=head if component == "mha" else None)
        responses = []
        for i in range(0, len(prompts), batch_size):
            responses.extend(steerer.generate_batch(prompts[i : i + batch_size], max_new_tokens=max_new_tokens))
        if alpha != 0.0:
            steerer.cleanup()

        records = [{"utterance_text": r["utterance_text"], "response": resp} for r, resp in zip(heldout_rows, responses)]
        rate, n_judged = _judge_sypr_praise_rate(records)
        for r, resp in zip(heldout_rows, responses):
            generations_log.append({
                "direction": direction_name, "section": "sypr_indomain", "alpha": alpha,
                "domain": r["domain"], "utterance_text": r["utterance_text"], "response": resp,
            })
        per_alpha[alpha] = {"rate": rate, "n_judged": n_judged}
    return per_alpha


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_accuracy_by_layer(probe_results: dict, balance_method: str, out_path: Path):
    n_layers = 1 + max(probe_results["mlp_accuracy"].keys())
    mha_acc = probe_results["mha_accuracy"]
    mha_best_per_layer = [
        max((acc for (layer, head), acc in mha_acc.items() if layer == l), default=0.5)
        for l in range(n_layers)
    ]
    mlp_by_layer = [probe_results["mlp_accuracy"][l] for l in range(n_layers)]
    res_by_layer = [probe_results["residual_accuracy"][l] for l in range(n_layers)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(n_layers), mha_best_per_layer, "o-", label="MHA (best head)")
    ax.plot(range(n_layers), mlp_by_layer, "o-", label="MLP")
    ax.plot(range(n_layers), res_by_layer, "o-", label="Residual")
    ax.axhline(0.5, color="gray", linestyle="--", label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe accuracy (5-fold CV)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"SyPR sycophantic-praise probe accuracy by layer ({balance_method})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_indomain_sweep(indomain_results: dict, balance_method: str, out_path: Path):
    alphas = sorted(indomain_results.keys())
    baseline = indomain_results[0.0]["rate"]
    deltas = [indomain_results[a]["rate"] - baseline for a in alphas]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, deltas, "o-")
    ax.axhline(0.0, color="gray", linestyle="--")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Praise-on-poor-quality rate delta vs. baseline")
    ax.set_title(f"SyPR in-domain steering sweep ({balance_method})")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cross_dataset(cross_results: dict, balance_method: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    nonzero_alphas = [a for a in ALPHAS if a != 0.0]
    for dataset, per_alpha in cross_results.items():
        baseline = per_alpha[0.0]["rate"]
        deltas = [per_alpha[a]["rate"] - baseline for a in nonzero_alphas]
        ax.plot(nonzero_alphas, deltas, "o-", label=dataset)
    ax.axhline(0.0, color="gray", linestyle="--")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Sycophancy-proxy rate delta vs. baseline")
    ax.set_title(f"SyPR-derived direction: cross-dataset generalization ({balance_method})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(
    out_dir: Path, balance_method: str, model_name: str, label_info: dict, n_heldout: int,
    probe_results: dict, best_key: dict, indomain_results: dict, cross_results: dict,
) -> None:
    lines = [
        f"# SyPR Probe -- Sycophantic Praise -- Summary (balance_method=\"{balance_method}\")",
        "",
        f"Model: `{model_name}` * Dataset: `vennemeyerd/sycophantic-praise` (SyPR benchmark, fresh "
        "generation required -- ships no responses of its own) * label: praise (sycophantic_praise_judge) "
        f"AND poor-quality (SyPR's own ground truth) * direction method: linear probe (`nn.Linear` + "
        f"`BCEWithLogitsLoss`, Adam lr=0.001, 25 epochs, batch size 25, 5-fold stratified CV) * "
        f"`balance_method=\"{balance_method}\"`.",
        "",
        "## Class balance",
        "",
        f"**{label_info['n_pos']} sycophantic (label=1) / {label_info['n_neg']} non-sycophantic (label=0) "
        f"out of {label_info['n_judged']} judged.** Praise rate on poor-quality utterances: "
        f"{label_info['praise_rate_on_poor']:.1%}. Praise rate on good-quality utterances: "
        f"{label_info['praise_rate_on_good']:.1%}.",
        "",
        "## Plots",
        "",
        "- `plots/01_probe_accuracy_by_layer.png` -- MHA (best head)/MLP/Residual accuracy, all layers",
        f"- `plots/02_indomain_steering_sweep.png` -- held-out poor-quality SyPR utterances (n={n_heldout}, "
        "disjoint from training rows by construction), delta vs. baseline",
        "- `plots/03_cross_dataset_generalization.png` -- AITA-NTA-FLIP/OG/YTA, OEQ, SS, delta vs. baseline",
        "",
        "## Best direction",
        "",
        f"- **MHA: layer {best_key['layer']}, head {best_key['head']}, "
        f"accuracy={probe_results['metadata']['mha_best_accuracy']:.3f}**",
        f"- MLP: layer {probe_results['metadata']['mlp_best_key']}, "
        f"accuracy={probe_results['metadata']['mlp_best_accuracy']:.3f}",
        f"- Residual: layer {probe_results['metadata']['residual_best_key']}, "
        f"accuracy={probe_results['metadata']['residual_best_accuracy']:.3f}",
        "",
        f"## In-domain steering sweep (held-out poor-quality utterances, n={n_heldout})",
        "",
        f"Baseline praise-on-poor rate (alpha=0.0): {indomain_results[0.0]['rate']*100:.2f}%",
        "",
        "| alpha | rate | delta |",
        "|---|---|---|",
    ]
    baseline_rate = indomain_results[0.0]["rate"]
    for a in ALPHAS:
        rate = indomain_results[a]["rate"]
        lines.append(f"| {a} | {rate*100:.2f}% | {(rate-baseline_rate)*100:+.2f}% |")

    lines += [
        "",
        "## Cross-dataset generalization (MHA best direction)",
        "",
        "| Dataset | Baseline | " + " | ".join(f"alpha={a}" for a in ALPHAS if a != 0.0) + " |",
        "|---|---|" + "---|" * (len(ALPHAS) - 1),
    ]
    for dataset, per_alpha in cross_results.items():
        baseline = per_alpha[0.0]["rate"]
        row = [dataset, f"{baseline*100:.2f}%"]
        for a in ALPHAS:
            if a == 0.0:
                continue
            row.append(f"{(per_alpha[a]['rate'] - baseline)*100:+.2f}%")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "Full raw generations: `generations.jsonl`. Full probe checkpoints: "
        "`mha_probe_weights.pth`/`mha_projection_stds.pt` (and mlp/residual equivalents), "
        "covering every (layer, head)/layer, not just the selected best.",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=120)
    parser.add_argument("--n-heldout", type=int, default=100)
    parser.add_argument("--cross-dataset-n", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=str(HERE / "results"))
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    model_config["model_name"] = args.model

    print(f"\nBuilding SyPR labels (n_train={args.n_train})...")
    label_info = build_labels(model, tokenizer, model_config, args.n_train, args.seed)
    print(
        f"Class balance: {label_info['n_pos']} pos / {label_info['n_neg']} neg "
        f"out of {label_info['n_judged']} judged"
    )

    print(f"\nSampling in-domain held-out set (n_heldout={args.n_heldout}, disjoint from training)...")
    dataset = load_sypr_dataset()
    heldout_rows = build_indomain_heldout(dataset, label_info["sampled_indices"], args.n_heldout, args.seed)

    texts = [r["text"] for r in label_info["records"]]
    y = np.array([r["label"] for r in label_info["records"]], dtype=np.float32)

    variant_dirs = train_all_variants(model, tokenizer, texts, y, model_config, output_dir)

    client = anthropic.Anthropic()
    all_results = {
        "model_name": args.model,
        "n_train": args.n_train,
        "n_heldout": args.n_heldout,
        "cross_dataset_n": args.cross_dataset_n,
        "label_balance": {
            "n_pos": label_info["n_pos"], "n_neg": label_info["n_neg"], "n_judged": label_info["n_judged"],
            "praise_rate_on_poor": label_info["praise_rate_on_poor"], "praise_rate_on_good": label_info["praise_rate_on_good"],
        },
        "variants": {},
    }

    for balance_method, info in variant_dirs.items():
        variant_dir = info["dir"]
        probe_results = load_probe_results(str(variant_dir))
        best_key = {
            "component": "mha",
            "layer": probe_results["metadata"]["mha_best_key"][0],
            "head": probe_results["metadata"]["mha_best_key"][1],
        }
        vectors = load_direction_vectors(str(variant_dir), "mha", fmt="probe")
        vector = vectors[(best_key["layer"], best_key["head"])]

        plots_dir = variant_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_accuracy_by_layer(probe_results, balance_method, plots_dir / "01_probe_accuracy_by_layer.png")

        generations_log = []
        steerer = ActivationSteerer(model, tokenizer, model_config)
        print(f"\n=== In-domain sweep: {balance_method} (MHA {best_key['layer']}/{best_key['head']}) ===")
        indomain_results = run_indomain_sweep(
            steerer, heldout_rows, "mha", best_key["layer"], best_key["head"], vector,
            ALPHAS, generations_log, f"sypr_{balance_method}",
        )
        plot_indomain_sweep(indomain_results, balance_method, plots_dir / "02_indomain_steering_sweep.png")

        print(f"\n=== Cross-dataset sweep: {balance_method} ===")
        cross_results = run_generalization_sweep(
            client, model, tokenizer, model_config, best_key, variant_dir,
            args.cross_dataset_n, 0, f"sypr_{balance_method}", generations_log,
            direction_fmt="probe",
        )
        plot_cross_dataset(cross_results, balance_method, plots_dir / "03_cross_dataset_generalization.png")

        with open(variant_dir / "generations.jsonl", "w") as f:
            for rec in generations_log:
                f.write(json.dumps(rec) + "\n")

        indomain_json = {str(a): v for a, v in indomain_results.items()}
        cross_results_json = {
            dataset: {str(a): v for a, v in per_alpha.items()} for dataset, per_alpha in cross_results.items()
        }
        with open(variant_dir / "results.json", "w") as f:
            json.dump({
                "balance_method": balance_method,
                "best_key": best_key,
                "probe_metadata": probe_results["metadata"],
                "indomain": indomain_json,
                "cross_dataset": cross_results_json,
            }, f, indent=2)

        write_summary(
            variant_dir, balance_method, args.model, label_info, args.n_heldout,
            probe_results, best_key, indomain_results, cross_results,
        )

        all_results["variants"][balance_method] = {
            "dir": str(variant_dir), "best_key": best_key, "probe_metadata": probe_results["metadata"],
        }
        print(f"Wrote {variant_dir}/")

    with open(output_dir / "sypr_probe_run_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    cleanup_model(model, tokenizer)
    print(f"\nDone. Results under {output_dir}/")


if __name__ == "__main__":
    main()
