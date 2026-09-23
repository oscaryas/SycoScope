#!/usr/bin/env python3
"""
Zero-shot transfer check: does the best-layer TruthfulQA residual probe
(layer 13 by default, trained via `category_residual_probe_pipeline.py
--categories truthfulqa`) generalize to other sycophancy datasets it was
never trained on? No retraining -- the frozen layer-13 nn.Linear probe is
scored directly against fresh residual activations extracted from each
target dataset, mirroring oeq_probe_aita_transfer_pipeline.py's score_probe
pattern (that script's LinearProbe reconstruction + Wilson-CI scoring,
generalized here to arbitrary target datasets instead of a single
OEQ->AITA check).

Targets:
    mixture      results/generations/sycophancy_mixture/mixture.jsonl (2,840
                 rows, 4-way balanced sypr/social/moral/are_you_sure mixture --
                 per-category accuracy breakdown reported since it carries a
                 "category" field)
    sypr         results/generations/sypr_praise_llama31_full/checkpoint.jsonl (full)
    social       results/generations/social_sycophancy_pooled/pooled.jsonl (full)
    truthfulqa   results/generations/truthfulqa_sycophancyeval/checkpoint.jsonl (full)
    are_you_sure results/generations/are_you_sure_pooled/pooled.jsonl (full, uncapped --
                 same file build_sycophancy_mixture.py draws its are_you_sure rows
                 from, natural ~50/50 balance so no calibration skew expected)
    moral        AITA-NTA-FLIP_moral_sycophancy_judged.jsonl, full judged pool
                 (no capping, unlike moral_avg_residual_probe_pipeline.py's own
                 370/370 training split) -- special-cased: unlike the other
                 targets' single forward pass over row["text"], moral needs
                 TWO forward passes per pair (original post, flipped story)
                 averaged per layer, same fix as moral_avg_residual_probe_
                 pipeline.py, because its label depends on both sides' verdict
                 and there's no pre-built single "text" field for it.

Direction formats (--direction-format):
    probe (default)  frozen nn.Linear logistic probe from --probe-dir's
                     residual_probe_weights.pth, decision rule logit > 0.
    dim              frozen difference-in-means direction from --dim-dir's
                     residual_dim_vectors.pt (saved alpha-ready as
                     unit_direction * proj_std by mixture_dim_pipeline.py's
                     save_dim_vectors) plus that run's per-layer decision
                     threshold from dim_classification_results.json. The
                     vector is re-unit-normalized before scoring because the
                     stored threshold is a midpoint of TRAIN class-mean
                     projections onto the UNIT direction -- comparing
                     projections of the proj_std-scaled vector against it
                     would misplace the boundary by a factor of proj_std.
                     Orientation is taken from the sign of the layer's
                     train_effect_size: at layers where the class-1 train
                     mean projects BELOW class-0 (negative d), the midpoint
                     rule inverts, so the score is sign-flipped to keep
                     "score > 0 == predicted sycophantic". --layer defaults
                     to the run's recorded best_layer for dim (vs. 13 for
                     probe), and --output-dir defaults to
                     cross_domain_transfer/mixture_dim_transfer.

Usage:
    python truthfulqa_probe_transfer_pipeline.py \
        --probe-dir results/probing/truthfulqa_residual --layer 13 \
        --targets mixture --model meta-llama/Meta-Llama-3-8B-Instruct

    # DIM cross-domain transfer:
    python truthfulqa_probe_transfer_pipeline.py \
        --direction-format dim --dim-dir results/probing/mixture_dim \
        --layer 14 --targets sypr,social,truthfulqa,are_you_sure,moral
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SYCOPHANCY_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
for p in (SYCOPHANCY_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sklearn.metrics import roc_auc_score

from utils.model import load_model_and_tokenizer, cleanup as cleanup_model
from utils.inference import build_chat_prompt
from sycophancy_model_registry import get_model_config
from sycophancy_probes import LinearProbe, _probe_accuracy, _probe_scores, _fold_auc, wilson_ci
from mixture_residual_probe_pipeline import collect_residual_only

GENERATIONS_DIR = SYCOPHANCY_DIR / "results" / "generations"
DEFAULT_MORAL_JUDGED = GENERATIONS_DIR / "AITA-NTA-FLIP_moral_sycophancy_judged.jsonl"

TARGETS = {
    "mixture": GENERATIONS_DIR / "sycophancy_mixture" / "mixture.jsonl",
    "sypr": GENERATIONS_DIR / "sypr_praise_llama31_full" / "checkpoint.jsonl",
    "social": GENERATIONS_DIR / "social_sycophancy_pooled" / "pooled.jsonl",
    "truthfulqa": GENERATIONS_DIR / "truthfulqa_sycophancyeval" / "checkpoint.jsonl",
    "are_you_sure": GENERATIONS_DIR / "are_you_sure_pooled" / "pooled.jsonl",
    "moral": DEFAULT_MORAL_JUDGED,
    "syconbench": GENERATIONS_DIR / "syconbench" / "conversations.jsonl",
    "dissociating_sycophancy": GENERATIONS_DIR / "dissociating_sycophancy" / "conversations_judged.jsonl",
}

# syconbench's 5-turn conversations run 571-2705 tokens (mean ~1675) -- well
# past the 1024-token default, and truncation_side="right" would cut off the
# final turn we're scoring rather than the start. See category_residual_
# probe_pipeline.py's CATEGORY_MAX_LENGTH for the same issue on the training side.
TARGET_MAX_LENGTH = {
    "syconbench": 3072,
}

# syconbench's "ever flipped" label reflects the WHOLE 5-turn conversation,
# not just the final turn -- default "mean" pooling (last delimiter to end)
# only covers the final turn, which mismatches label scope for the ~80% of
# positive examples that flip early then recover by the end (verified
# against debate_setting's own alignment arrays). "mean_all_turns" pools
# every assistant turn's response tokens instead, matching label scope.
TARGET_POOLING = {
    "syconbench": "mean_all_turns",
}


def load_target(path: Path) -> tuple:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    texts = [r["text"] for r in rows]
    labels = np.array([r["label"] for r in rows], dtype=np.float32)
    # mixture rows have "category" (sypr/social/moral/are_you_sure); truthfulqa
    # rows instead have "template" (plain vs. pushback_incorrect); are_you_sure_
    # pooled rows have neither but do have "domain" (mmlu_mc_cot/truthful_qa_mc/
    # aqua_mc/math_mc_cot are multiple-choice, trivia_qa/truthful_qa are
    # freeform -- "Give me your best guess", no letter options). Falling back
    # through all three lets score_by_group break out per-template /
    # per-domain accuracy-AUC for free, without a separate filtered target or
    # extraction pass.
    categories = [r.get("category") or r.get("template") or r.get("domain") for r in rows]
    return texts, labels, categories


def load_moral_pairs(path: Path) -> tuple:
    """Full (uncapped) moral judged pool, same label definition as
    moral_avg_residual_probe_pipeline.py's select_moral_pairs: positive =
    both NTA, negative = both YTA or either side OTHER/unclear, Mixed pairs
    (one NTA one YTA) excluded entirely. Returns (rows, labels) -- caller
    builds og/flip chat-prompt texts with the tokenizer and extracts
    activations for each side separately."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    selected, labels = [], []
    for r in rows:
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
    Distinct from sycophancy_probes.bootstrap_ci, which bootstraps a 1-D
    array of already-computed per-fold statistics rather than raw examples.
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


class DimScorer:
    """Duck-typed stand-in for the frozen LinearProbe in _probe_accuracy /
    _probe_scores (both just call `probe(X_t)` under no_grad and threshold
    at 0): score(X) = sign * (X @ unit_direction - threshold), so score > 0
    means predicted sycophantic, matching the probe path's logit > 0 rule.
    AUC is computed from these scores too -- the constant threshold shift
    and sign flip leave AUC's ranking semantics intact (a flipped direction
    without the sign correction would report 1 - AUC)."""

    def __init__(self, vector: torch.Tensor, threshold: float, flipped: bool):
        v = vector.float()
        self.unit_direction = v / (v.norm() + 1e-8)
        self.threshold = float(threshold)
        self.sign = -1.0 if flipped else 1.0

    def __call__(self, X_t: torch.Tensor) -> torch.Tensor:
        return self.sign * (X_t @ self.unit_direction - self.threshold)


def dim_cv_out_of_sample(X: np.ndarray, y: np.ndarray, strata: list, n_folds: int, seed: int) -> np.ndarray:
    """Genuine out-of-sample DIM scores for EVERY row of the fitting
    distribution (the mixture): stratified k-fold by (category, label); per
    fold, a fresh difference-in-means direction + train-midpoint threshold is
    fit on the other k-1 folds and only the held-out fold is scored with it.
    Complements the frozen-direction score on the full mixture, which is
    ~99% in-sample (the saved direction was fit on all but --n-heldout=20ish
    rows of it). No orientation handling needed: (mean1 - mean0) always
    projects the class-1 TRAIN mean above class-0 by construction.
    Returns oriented scores, >0 == predicted sycophantic."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for s in sorted(set(zip(strata, y.tolist()))):
        idx = np.nonzero([(st, yy) == s for st, yy in zip(strata, y.tolist())])[0]
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % n_folds
    scores = np.zeros(len(y), dtype=np.float64)
    for f in range(n_folds):
        te, tr = fold_of == f, fold_of != f
        Xtr, ytr = X[tr], y[tr]
        d = Xtr[ytr == 1].mean(axis=0) - Xtr[ytr == 0].mean(axis=0)
        d = d / (np.linalg.norm(d) + 1e-8)
        thr = ((Xtr[ytr == 1] @ d).mean() + (Xtr[ytr == 0] @ d).mean()) / 2.0
        scores[te] = X[te] @ d - thr
    return scores


def report_from_scores(scores: np.ndarray, labels: np.ndarray, groups: list) -> dict:
    """Same output shape as score_by_group, but from precomputed continuous
    scores (>0 == predicted positive) instead of a probe callable -- needed
    for CV out-of-sample scores, where each row was scored by a different
    fold's direction so no single frozen scorer exists."""
    out = {}
    for name, idx in [("__all__", list(range(len(labels))))] + [
        (g, [i for i, gg in enumerate(groups) if gg == g]) for g in sorted(set(groups)) if g is not None
    ]:
        sg, yg = scores[idx], labels[idx]
        n_correct = int(((sg > 0).astype(np.float32) == yg).sum())
        n_total = len(yg)
        auc = _fold_auc(yg, sg)
        out[name] = {
            "n": n_total, "n_pos": int((yg == 1).sum()),
            "accuracy": n_correct / n_total if n_total else 0.0,
            "ci": wilson_ci(n_correct, n_total),
            "auc_roc": auc,
            "auc_roc_ci": bootstrap_auc_ci(yg, sg) if auc is not None else None,
        }
    return out


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
    parser.add_argument("--direction-format", type=str, default="probe", choices=["probe", "dim"],
                         help="'probe': frozen logistic probe from --probe-dir. 'dim': frozen DIM direction + "
                              "stored train-midpoint threshold from --dim-dir (see module docstring).")
    parser.add_argument("--probe-dir", type=str, default=str(SYCOPHANCY_DIR / "results" / "probing" / "truthfulqa_residual"))
    parser.add_argument("--dim-dir", type=str, default=str(SYCOPHANCY_DIR / "results" / "probing" / "mixture_dim"),
                         help="Directory holding residual_dim_vectors.pt + dim_classification_results.json (--direction-format dim only).")
    parser.add_argument("--layer", type=int, default=None,
                         help="Layer of the saved direction to score with. Default: 13 for probe (the trained "
                              "best layer), or the dim run's recorded best_layer for dim.")
    parser.add_argument("--targets", type=str, default="mixture", help="Comma-separated subset of: " + ",".join(TARGETS.keys()))
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Default: cross_domain_transfer/truthfulqa_probe_transfer for probe, "
                              "cross_domain_transfer/mixture_dim_transfer for dim.")
    parser.add_argument("--dim-cv-folds", type=int, default=5,
                         help="For --direction-format dim + the 'mixture' target: also report genuine "
                              "out-of-sample scores via stratified k-fold DIM refitting (0 disables). "
                              "The frozen-direction score on the full mixture is ~99% in-sample; this is "
                              "the honest in-distribution number.")
    parser.add_argument("--seed", type=int, default=0, help="Fold-assignment seed for --dim-cv-folds.")
    parser.add_argument("--pooling-override", type=str, default=None,
                         help="Overrides TARGET_POOLING for every selected target (e.g. 'mean_first_turn' to "
                              "test pooling only the first assistant turn of a multi-turn target like syconbench). "
                              "Omit to use each target's TARGET_POOLING entry (default 'mean').")
    args = parser.parse_args()

    selected = args.targets.split(",")
    for name in selected:
        if name not in TARGETS:
            print(f"ERROR: unknown target {name!r}, must be one of {list(TARGETS)}", file=sys.stderr)
            sys.exit(1)

    if args.direction_format == "probe":
        if args.layer is None:
            args.layer = 13
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
        direction_source = str(weights_path)
        direction_meta = {}
        print(f"Loaded frozen probe from {weights_path} @ layer {args.layer} (input_dim={input_dim})")
    else:
        vectors_path = Path(args.dim_dir) / "residual_dim_vectors.pt"
        metrics_path = Path(args.dim_dir) / "dim_classification_results.json"
        vectors = torch.load(vectors_path, map_location="cpu")
        dim_metrics = json.loads(metrics_path.read_text())
        if args.layer is None:
            args.layer = int(dim_metrics["best_layer"])
            print(f"--layer not given: using the dim run's recorded best_layer={args.layer}")
        if args.layer not in vectors:
            print(f"ERROR: layer {args.layer} not in {vectors_path} (available: {sorted(vectors)})", file=sys.stderr)
            sys.exit(1)
        layer_metrics = dim_metrics["per_layer"][str(args.layer)]
        flipped = layer_metrics["train_effect_size"] < 0
        probe = DimScorer(vectors[args.layer], layer_metrics["threshold"], flipped)
        direction_source = str(vectors_path)
        direction_meta = {
            "threshold": layer_metrics["threshold"],
            "orientation_flipped": flipped,
            "train_effect_size": layer_metrics["train_effect_size"],
            "dim_heldout_accuracy": layer_metrics["accuracy"],
            "dim_heldout_auroc": layer_metrics["auroc"],
        }
        print(f"Loaded frozen DIM direction from {vectors_path} @ layer {args.layer} "
              f"(threshold={layer_metrics['threshold']:.4f}, flipped={flipped}, "
              f"train_d={layer_metrics['train_effect_size']:+.3f})")

    if args.output_dir is None:
        default_name = "truthfulqa_probe_transfer" if args.direction_format == "probe" else "mixture_dim_transfer"
        args.output_dir = str(SYCOPHANCY_DIR / "results" / "cross_domain_transfer" / default_name)

    print(f"\nLoading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    model_config["model_name"] = args.model
    n_layers = model_config["n_layers"]

    all_results = {
        "direction_format": args.direction_format,
        "probe_source": direction_source, "probe_layer": args.layer,
        "direction_meta": direction_meta, "model": args.model, "targets": {},
    }
    for name in selected:
        path = TARGETS[name]
        print(f"\n{'='*70}\nTarget: {name}  ({path})\n{'='*70}")

        if name == "moral":
            rows, labels = load_moral_pairs(path)
            n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
            print(f"Loaded {len(rows)} pairs ({n_pos} positive / {n_neg} negative, Mixed excluded)")

            og_texts = [build_chat_prompt(tokenizer, r["original_post_prompt"], system_prompt=None) + r["original_post_response"] for r in rows]
            flip_texts = [build_chat_prompt(tokenizer, r["flipped_story_prompt"], system_prompt=None) + r["flipped_story_response"] for r in rows]
            print(f"Extracting residual activations for {len(rows)} pairs, original-post side...")
            og_activations = collect_residual_only(model, tokenizer, og_texts, model_config)
            print(f"Extracting residual activations for {len(rows)} pairs, flipped-story side...")
            flip_activations = collect_residual_only(model, tokenizer, flip_texts, model_config)
            residual_activations = (og_activations + flip_activations) / 2.0
            X = residual_activations[args.layer]
            categories = [None] * len(rows)
            n_total = len(rows)
        elif name == "mixture" and args.direction_format == "dim":
            # Match DIM training's extraction exactly: moral rows og+flip
            # averaged (moralfix), everything else single-pass -- reuses
            # mixture_dim_pipeline.extract_all_activations so the frozen
            # direction is scored on the same kind of activations it was fit
            # on (the generic branch below would single-pass moral rows).
            from mixture_dim_pipeline import extract_all_activations
            rows = [json.loads(line) for line in open(path, encoding="utf-8")]
            labels = np.array([r["label"] for r in rows], dtype=np.float32)
            categories = [r["category"] for r in rows]
            n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
            print(f"Loaded {len(rows)} mixture rows ({n_pos} positive / {n_neg} negative), moralfix extraction")
            residual_activations = extract_all_activations(model, tokenizer, model_config, rows, DEFAULT_MORAL_JUDGED)
            X = residual_activations[args.layer]
            n_total = len(rows)
        else:
            texts, labels, categories = load_target(path)
            n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
            print(f"Loaded {len(texts)} rows ({n_pos} positive / {n_neg} negative)")

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

        if name == "mixture" and args.direction_format == "dim":
            all_results["targets"][name]["frozen_direction_note"] = (
                "the 'scores' block scores the FROZEN saved direction, which was fit on all but "
                "~20 of these rows -- treat as in-sample fit quality, not generalization"
            )
            if args.dim_cv_folds > 0:
                print(f"Computing {args.dim_cv_folds}-fold out-of-sample DIM scores on the mixture "
                      f"(stratified by category x label, seed={args.seed})...")
                oos_scores = dim_cv_out_of_sample(X, labels, categories, args.dim_cv_folds, args.seed)
                oos = report_from_scores(oos_scores, labels, categories)
                all_results["targets"][name]["out_of_sample_cv"] = {
                    "n_folds": args.dim_cv_folds, "seed": args.seed, "scores": oos,
                }
                overall_oos = oos["__all__"]
                print(f"[mixture OOS {args.dim_cv_folds}-fold] overall: accuracy={overall_oos['accuracy']:.3f} "
                      f"CI=[{overall_oos['ci'][0]:.3f},{overall_oos['ci'][1]:.3f}] AUC={overall_oos['auc_roc']}")
                for cat, r in oos.items():
                    if cat != "__all__":
                        print(f"    {cat}: accuracy={r['accuracy']:.3f} (n={r['n']}) AUC={r['auc_roc']}")

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
