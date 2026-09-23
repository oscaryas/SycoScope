#!/usr/bin/env python3
"""
gemma-4-12B-it analogue of truthfulqa_probe_transfer_pipeline.py -- zero-shot
genuine held-out transfer check for the gemma sycophancy-mixture residual
probe (layer 31 by default, trained via mixture_residual_probe_pipeline.py
on gemma_sycophancy_mixture/mixture.jsonl). No retraining -- the frozen
layer-31 nn.Linear probe is scored directly against fresh residual
activations extracted from each target. sypr/social/are_you_sure/moral are
scored on the SAME categories the probe was trained on, but on rows
EXCLUDED from the training mixture (the exact 410 rows/pairs each category
contributed are removed -- see the TARGETS comment for how). truthfulqa is
never in the mixture at all, the strongest generalization test here.

Targets (all google/gemma-4-12B-it, this session's own generations, no
llama31 data anywhere):
    sypr         .../ood_pools/sypr_heldout.jsonl (2,012 rows, mixture's 410
                 sypr rows excluded, ~1% pos remaining -- caution: only 13
                 positive examples left, so this cell's accuracy/AUC is noisy)
    social       .../ood_pools/social_heldout.jsonl (1,983 rows, oeq+ss pooled,
                 mixture's 410 social rows excluded, ~85% pos)
    are_you_sure .../ood_pools/are_you_sure_heldout.jsonl (1,812 rows, freeform+mc
                 pooled, mixture's 410 are_you_sure rows excluded -- caution:
                 are_you_sure only had 205 positive rows total and ALL of them
                 went into the training mixture, so this target has ZERO
                 positive examples; accuracy/AUC on it can only ever reflect
                 "always predict negative" and is not a meaningful generalization
                 signal, kept only for the negative-class calibration check)
    truthfulqa   .../truthfulqa/checkpoint.jsonl (3,456 rows, full -- NEVER in
                 the training mixture at all, the strongest OOD test here)
    moral        .../aita_nta_flip/judged.jsonl, mixture's 410 row_ids excluded
                 via MORAL_EXCLUDED_ROW_IDS_PATH -- special-cased:
                 unlike the other targets' single forward pass over row["text"],
                 moral needs TWO forward passes per pair (original post,
                 flipped story) averaged per layer, matching moral_avg_
                 residual_probe_pipeline.py's fix for the truncation bug.
                 Unlike the llama31 version of this script, gemma's judged.jsonl
                 prompt fields (original_post_prompt/flipped_story_prompt) are
                 ALREADY full chat-templated strings (built at generation time),
                 so this version does NOT re-wrap them with build_chat_prompt --
                 it just concatenates prompt+response directly, same convention
                 as build_gemma_sycophancy_mixture.py's load_moral().

SYCON-Bench is intentionally NOT wired up here (no conversations.jsonl exists
yet for gemma) -- add a "syconbench" target once that's generated, mirroring
the llama31 script's TARGET_MAX_LENGTH/TARGET_POOLING overrides for it.

Usage:
    python -m probing.probe.gemma_mixture_probe_transfer_pipeline \
        --probe-dir probing/data/google__gemma-4-12B-it/gemma_sycophancy_mixture_residual --layer 31 \
        --targets sypr,social,are_you_sure,truthfulqa,moral \
        --model google/gemma-4-12B-it
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.metrics import roc_auc_score

from utils.model import load_model_and_tokenizer, cleanup as cleanup_model
from utils.model_registry import get_model_config
from probing.probe.baseline_probes import LinearProbe, _probe_accuracy, _probe_scores, _fold_auc, wilson_ci
from probing.probe.mixture_residual_probe_pipeline import collect_residual_only

GEMMA_DIR = REPO_ROOT / "probing" / "data" / "google__gemma-4-12B-it"
DEFAULT_MORAL_JUDGED = GEMMA_DIR / "aita_nta_flip" / "judged.jsonl"

OOD_DIR = GEMMA_DIR / "ood_pools"
MORAL_EXCLUDED_ROW_IDS_PATH = OOD_DIR / "moral_excluded_row_ids.json"

TARGETS = {
    # sypr/social/are_you_sure/moral targets are held OUT of the training mixture --
    # the *_heldout.jsonl files (and moral's row_id exclusion list) were built by
    # exactly replaying build_gemma_sycophancy_mixture.py's category-pool
    # construction + random.Random(0) sampling order/indices, so what's excluded
    # here is provably identical to what mixture.jsonl actually drew (a first pass
    # using exact-text-match exclusion undercounted for sypr, which has some
    # duplicate response text across rows -- 601 excluded instead of the true 410).
    "sypr": OOD_DIR / "sypr_heldout.jsonl",
    "social": OOD_DIR / "social_heldout.jsonl",
    "are_you_sure": OOD_DIR / "are_you_sure_heldout.jsonl",
    "truthfulqa": GEMMA_DIR / "truthfulqa" / "checkpoint.jsonl",  # never in the mixture at all
    "moral": DEFAULT_MORAL_JUDGED,  # filtered against MORAL_EXCLUDED_ROW_IDS_PATH in load_moral_pairs
}

TARGET_MAX_LENGTH = {}
TARGET_POOLING = {}


def load_target(path: Path) -> tuple:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    texts = [r["text"] for r in rows]
    labels = np.array([r["label"] for r in rows], dtype=np.float32)
    # mixture rows have "category" (sypr/social/moral/are_you_sure); truthfulqa
    # rows instead have "template" (plain vs. pushback_incorrect); are_you_sure_
    # full-pool rows have neither but do have "domain" (truthful_qa/trivia_qa for
    # freeform, or an MC domain for mc rows). Falling back through all three lets
    # score_by_group break out per-template/per-domain accuracy-AUC for free,
    # without a separate filtered target or extraction pass.
    categories = [r.get("category") or r.get("template") or r.get("domain") for r in rows]
    return texts, labels, categories


def load_moral_pairs(path: Path) -> tuple:
    """Held-out (mixture-excluded) moral judged pool, same label definition as
    build_gemma_sycophancy_mixture.py's load_moral(): positive = both NTA,
    negative = both YTA or either side OTHER/unclear, Mixed pairs (one NTA
    one YTA) excluded entirely. Also drops the exact 410 row_ids that went
    into the training mixture (MORAL_EXCLUDED_ROW_IDS_PATH, built by
    replaying the mixture's sampling RNG -- see TARGETS comment) so this is
    a genuine held-out eval, not partly in-sample. Returns (rows, labels) --
    caller builds og/flip texts by concatenating each side's already-
    templated prompt with its response and extracts activations for each
    side separately."""
    excluded_ids = set(json.loads(MORAL_EXCLUDED_ROW_IDS_PATH.read_text())) if MORAL_EXCLUDED_ROW_IDS_PATH.exists() else set()
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    selected, labels = [], []
    for r in rows:
        if r["row_id"] in excluded_ids:
            continue
        og, flip = r["original_post_verdict"], r["flipped_story_verdict"]
        if og == "NTA" and flip == "NTA":
            label = 1
        elif (og == "NTA" and flip == "YTA") or (og == "YTA" and flip == "NTA"):
            continue  # Mixed -- excluded
        else:
            label = 0  # both YTA, or either side OTHER
        selected.append(r)
        labels.append(label)
    return selected, np.array(labels, dtype=np.float32)


def bootstrap_auc_ci(y: np.ndarray, scores: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05, seed: int = 42):
    """Percentile bootstrap CI for a single-shot AUC-ROC point estimate --
    resamples (label, score) pairs together (not scores alone, since AUC is
    a function of the pairing) and recomputes roc_auc_score each draw.
    Returns None if every draw collapses to a single class (tiny/imbalanced
    groups) -- same "no signal" convention as _fold_auc returning None."""
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boots.append(roc_auc_score(y_b, scores[idx]))
    if not boots:
        return None
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (lo, hi)


def subsample_indices(labels: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    """Stratified subsample: up to max_per_class examples of each label
    value (all of them if fewer are available), shuffled seed-reproducibly.
    Keeps every positive when a class is already scarce (e.g. sypr's 13) --
    it can only shrink a class, never pad it out."""
    rng = np.random.default_rng(seed)
    idx = []
    for v in sorted(set(labels.tolist())):
        cls_idx = np.nonzero(labels == v)[0]
        if len(cls_idx) > max_per_class:
            cls_idx = rng.choice(cls_idx, size=max_per_class, replace=False)
        idx.extend(cls_idx.tolist())
    idx = np.array(sorted(idx))
    rng.shuffle(idx)
    return idx


def score_by_group(probe, X: np.ndarray, labels: np.ndarray, groups: list) -> dict:
    """Accuracy/CI/AUC(+CI) for the full set plus each distinct group value
    (e.g. mixture's "category" field) -- None group values are skipped as
    their own bucket (non-mixture targets have no category field at all)."""
    out = {}
    for name, idx in [("__all__", list(range(len(labels))))] + [
        (g, [i for i, gg in enumerate(groups) if gg == g]) for g in sorted(set(groups)) if g is not None
    ]:
        Xg, yg = X[idx], labels[idx]
        n_correct, n_total, accuracy = _probe_accuracy(probe, Xg, yg)
        ci = wilson_ci(n_correct, n_total)
        scores_g = _probe_scores(probe, Xg)
        auc = _fold_auc(yg, scores_g)
        auc_ci = bootstrap_auc_ci(yg, scores_g) if auc is not None else None
        out[name] = {
            "n": n_total, "n_pos": int((yg == 1).sum()),
            "accuracy": accuracy, "ci": ci,
            "auc_roc": auc, "auc_roc_ci": auc_ci,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe-dir", type=str, default=str(GEMMA_DIR / "gemma_sycophancy_mixture_residual"))
    parser.add_argument("--layer", type=int, default=31, help="Layer of the saved probe to score with (default: 31, this mixture's trained best layer).")
    parser.add_argument("--targets", type=str, default="sypr,social,are_you_sure,truthfulqa,moral",
                         help="Comma-separated subset of: " + ",".join(TARGETS.keys()))
    parser.add_argument("--model", type=str, default="google/gemma-4-12B-it")
    parser.add_argument("--output-dir", type=str, default=str(GEMMA_DIR / "gemma_mixture_probe_transfer"))
    parser.add_argument("--pooling-override", type=str, default=None,
                         help="Overrides TARGET_POOLING for every selected target. Omit to use 'mean' for all (no targets have a non-default entry currently).")
    parser.add_argument("--max-per-class", type=int, default=None,
                         help="Stratified-subsample each target to at most this many examples per label BEFORE "
                              "extraction (never pads a scarce class up, e.g. sypr's 13 positives stay 13). "
                              "Omit to run the full held-out pool for every target.")
    parser.add_argument("--seed", type=int, default=0, help="Subsample seed (--max-per-class only).")
    args = parser.parse_args()

    selected = args.targets.split(",")
    for name in selected:
        if name not in TARGETS:
            print(f"ERROR: unknown target {name!r}, must be one of {list(TARGETS)}", file=sys.stderr)
            sys.exit(1)

    weights_path = Path(args.probe_dir) / "residual_probe_weights.pth"
    weights_ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if args.layer not in weights_ckpt:
        print(f"ERROR: layer {args.layer} not in {weights_path} (available: {sorted(weights_ckpt)})", file=sys.stderr)
        sys.exit(1)
    state_dict = weights_ckpt[args.layer]
    input_dim = state_dict["linear.weight"].shape[1]
    probe = LinearProbe(input_dim)
    probe.load_state_dict(state_dict)
    probe.eval()
    print(f"Loaded frozen probe from {weights_path} @ layer {args.layer} (input_dim={input_dim})")

    print(f"\nLoading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    model_config["model_name"] = args.model
    n_layers = model_config["n_layers"]

    all_results = {
        "probe_source": str(weights_path), "probe_layer": args.layer,
        "model": args.model, "targets": {},
    }
    for name in selected:
        path = TARGETS[name]
        print(f"\n{'='*70}\nTarget: {name}  ({path})\n{'='*70}")

        if name == "moral":
            rows, labels = load_moral_pairs(path)
            n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
            print(f"Loaded {len(rows)} pairs ({n_pos} positive / {n_neg} negative, Mixed excluded)")

            if args.max_per_class is not None:
                keep = subsample_indices(labels, args.max_per_class, args.seed)
                rows = [rows[i] for i in keep]
                labels = labels[keep]
                n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
                print(f"Subsampled to {len(rows)} pairs ({n_pos} positive / {n_neg} negative, max_per_class={args.max_per_class})")

            og_texts = [r["original_post_prompt"] + r["original_post_response"] for r in rows]
            flip_texts = [r["flipped_story_prompt"] + r["flipped_story_response"] for r in rows]
            print(f"Extracting residual activations for {len(rows)} pairs, original-post side...")
            og_activations = collect_residual_only(model, tokenizer, og_texts, model_config)
            print(f"Extracting residual activations for {len(rows)} pairs, flipped-story side...")
            flip_activations = collect_residual_only(model, tokenizer, flip_texts, model_config)
            residual_activations = (og_activations + flip_activations) / 2.0
            X = residual_activations[args.layer]
            categories = [None] * len(rows)
            n_total = len(rows)
        else:
            texts, labels, categories = load_target(path)
            n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
            print(f"Loaded {len(texts)} rows ({n_pos} positive / {n_neg} negative)")

            if args.max_per_class is not None:
                keep = subsample_indices(labels, args.max_per_class, args.seed)
                texts = [texts[i] for i in keep]
                labels = labels[keep]
                categories = [categories[i] for i in keep]
                n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
                print(f"Subsampled to {len(texts)} rows ({n_pos} positive / {n_neg} negative, max_per_class={args.max_per_class})")

            max_length = TARGET_MAX_LENGTH.get(name, 1024)
            target_pooling = args.pooling_override or TARGET_POOLING.get(name, "mean")
            print(f"Extracting residual-stream activations for {len(texts)} examples ({n_layers} layers, max_length={max_length}, pooling={target_pooling!r})...")
            residual_activations = collect_residual_only(model, tokenizer, texts, model_config, max_length=max_length, pooling=target_pooling)
            X = residual_activations[args.layer]  # (n_examples, hidden_dim)
            n_total = len(texts)

        finite_mask = np.isfinite(X).all(axis=1)
        n_dropped_nan = int((~finite_mask).sum())
        if n_dropped_nan:
            print(f"WARNING: dropping {n_dropped_nan} row(s) with non-finite residual activations "
                  f"(likely empty/degenerate response span for the pooling window) before scoring")
            X = X[finite_mask]
            labels = labels[finite_mask]
            categories = [c for c, keep in zip(categories, finite_mask) if keep]

        scores = score_by_group(probe, X, labels, categories)
        all_results["targets"][name] = {
            "n_total": n_total, "n_pos": n_pos, "n_neg": n_neg,
            "n_dropped_nan": n_dropped_nan, "scores": scores,
        }

        overall = scores["__all__"]
        print(f"[{name}] overall: accuracy={overall['accuracy']:.3f} CI=[{overall['ci'][0]:.3f},{overall['ci'][1]:.3f}] "
              f"AUC={overall['auc_roc']} AUC_CI={overall['auc_roc_ci']}")
        for cat, r in scores.items():
            if cat == "__all__":
                continue
            print(f"    {cat}: accuracy={r['accuracy']:.3f} CI=[{r['ci'][0]:.3f},{r['ci'][1]:.3f}] "
                  f"(n={r['n']}, n_pos={r['n_pos']}) AUC={r['auc_roc']} AUC_CI={r['auc_roc_ci']}")

        del residual_activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cleanup_model(model, tokenizer)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDone. Results written to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
