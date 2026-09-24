#!/usr/bin/env python3
"""Cache + train residual probes on the 4-way sycophancy mixture (sypr / social /
are_you_sure / moral; built by evaluations/generation/build_gemma_sycophancy_mixture.py).

Extraction (the moral-fix recipe of the retired mixture_residual_probe_pipeline_v2.py):
sypr/social/are_you_sure rows get one forward pass over row["text"]; moral rows
are looked up by meta.row_id in the judged AITA-NTA-FLIP file and get TWO passes
(original post, flipped story) averaged per layer, since the mixture's single
concatenated moral text truncates the flipped side away for many pairs.

Training: one L2 C sweep on the pooled mixture, or with --per-category one sweep
per category on that category's rows (the retired
mixture_category_residual_probe_pipeline.py). CV groups: meta.row_id,
utterance_text or question within a category, else the row itself.

--heldout-targets then scores each layer's best-C probe zero-shot on held-out
pools that exclude every mixture row (the retired
gemma_mixture_probe_transfer_pipeline.py, now with the sklearn probe).

Usage:
    python -m probing.analyze_probes.cache_train_gemma_mixture \\
        --layers 31 --output probing/results/probes/google__gemma-4-12B-it/mixture_l2/weights.pkl \\
        --heldout-targets sypr,social,are_you_sure,truthfulqa,moral
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from probing.analyze_probes import probes_core as core  # noqa: E402

GEMMA_MODEL = "google/gemma-4-12B-it"
GEMMA_DATA = REPO_ROOT / "probing" / "data" / "baseline" / "google__gemma-4-12B-it"
GEMMA_PROBES = REPO_ROOT / "probing" / "results" / "probes" / "google__gemma-4-12B-it"
DEFAULT_MIXTURE = GEMMA_PROBES / "gemma_sycophancy_mixture" / "mixture.jsonl"
DEFAULT_JUDGED = GEMMA_DATA / "aita_nta_flip" / "judged.jsonl"
OOD_DIR = GEMMA_PROBES / "ood_pools"
MORAL_EXCLUDED_ROW_IDS_PATH = OOD_DIR / "moral_excluded_row_ids.json"

# Held-out pools replay build_gemma_sycophancy_mixture.py's sampling to exclude
# exactly the rows the mixture drew. are_you_sure has ZERO positives left (all
# 205 went into the mixture) and sypr only ~13, so read those cells with care.
TARGETS = {
    "sypr": OOD_DIR / "sypr_heldout.jsonl",
    "social": OOD_DIR / "social_heldout.jsonl",
    "are_you_sure": OOD_DIR / "are_you_sure_heldout.jsonl",
    "truthfulqa": GEMMA_DATA / "truthfulqa" / "checkpoint.jsonl",  # never in the mixture at all
    "moral": DEFAULT_JUDGED,  # filtered against MORAL_EXCLUDED_ROW_IDS_PATH in load_moral_pairs
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def load_judged_by_row_id(path: Path) -> dict:
    return {rec["row_id"]: rec for rec in read_jsonl(path)}


def mixture_group(row: dict, i: int) -> str:
    meta = row.get("meta") or {}
    for key in ("row_id", "utterance_text", "question"):
        if meta.get(key):
            return f"{row['category']}:{meta[key]}"
    return f"row:{i}"


def moral_side_texts(tokenizer, recs: list[dict], side: str, raw_prompts: bool) -> list[str]:
    """gemma's judged.jsonl prompts are already full chat-templated strings, so by
    default prompt + response is concatenated directly; --raw-moral-prompts wraps
    them with the chat template first (llama-era judged files)."""
    if not raw_prompts:
        return [r[f"{side}_prompt"] + r[f"{side}_response"] for r in recs]
    from utils.inference import build_chat_prompt

    return [build_chat_prompt(tokenizer, r[f"{side}_prompt"], system_prompt=None) + r[f"{side}_response"] for r in recs]


def extract_moral_two_pass(model, tokenizer, model_config, recs, args, pooling) -> np.ndarray:
    og_texts = moral_side_texts(tokenizer, recs, "original_post", args.raw_moral_prompts)
    flip_texts = moral_side_texts(tokenizer, recs, "flipped_story", args.raw_moral_prompts)
    print(f"Extracting residual activations for {len(recs)} moral pairs, original-post side...")
    og = core.collect_residual_only(model, tokenizer, og_texts, model_config, args.max_length, pooling)
    print(f"Extracting residual activations for {len(recs)} moral pairs, flipped-story side...")
    flip = core.collect_residual_only(model, tokenizer, flip_texts, model_config, args.max_length, pooling)
    return (og + flip) / 2.0


def extract_mixture(model, tokenizer, model_config, rows, judged_by_row_id, args, pooling) -> np.ndarray:
    moral_idx = [i for i, r in enumerate(rows) if r["category"] == "moral"]
    other_idx = [i for i, r in enumerate(rows) if r["category"] != "moral"]
    print(f"moral rows needing two-pass fix: {len(moral_idx)}, other rows (single pass): {len(other_idx)}")

    print(f"\n[1/2] Extracting residual activations for {len(other_idx)} non-moral rows...")
    other = core.collect_residual_only(model, tokenizer, [rows[i]["text"] for i in other_idx],
                                       model_config, args.max_length, pooling)
    print("\n[2/2] moral rows")
    moral = extract_moral_two_pass(model, tokenizer, model_config,
                                   [judged_by_row_id[rows[i]["meta"]["row_id"]] for i in moral_idx], args, pooling)

    # Reassemble into original row order: (n_layers, n_total_rows, hidden_dim)
    combined = np.zeros((model_config["n_layers"], len(rows), model_config["hidden_dim"]), dtype=np.float32)
    for slot, i in enumerate(other_idx):
        combined[:, i, :] = other[:, slot, :]
    for slot, i in enumerate(moral_idx):
        combined[:, i, :] = moral[:, slot, :]
    return combined


# ---------------------------------------------------------------------------
# Held-out transfer targets
# ---------------------------------------------------------------------------


def load_target(path: Path) -> tuple:
    rows = read_jsonl(path)
    texts = [r["text"] for r in rows]
    labels = np.array([int(float(r["label"])) for r in rows], dtype=int)
    # mixture-style rows carry "category"; truthfulqa rows "template"; are_you_sure
    # pool rows "domain" -- whichever exists gives a per-group breakdown for free.
    categories = [r.get("category") or r.get("template") or r.get("domain") for r in rows]
    return texts, labels, categories


def load_moral_pairs(path: Path) -> tuple:
    """Held-out moral pool: same label definition as the mixture (both NTA = 1;
    both YTA or either OTHER = 0; Mixed excluded), minus the row_ids the
    mixture drew (MORAL_EXCLUDED_ROW_IDS_PATH)."""
    excluded_ids = set(json.loads(MORAL_EXCLUDED_ROW_IDS_PATH.read_text())) if MORAL_EXCLUDED_ROW_IDS_PATH.exists() else set()
    selected, labels = [], []
    for r in read_jsonl(path):
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
    return selected, np.array(labels, dtype=int)


def subsample_indices(labels: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    """Stratified subsample: up to max_per_class examples of each label, shuffled
    seed-reproducibly. Only ever shrinks a class, never pads it."""
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


def bootstrap_auc_ci(y: np.ndarray, scores: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05, seed: int = 42):
    """Percentile bootstrap CI for an AUC point estimate, resampling (label,
    score) pairs together. None if every draw collapses to one class."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], scores[idx]))
    if not boots:
        return None
    return (float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2))))


def score_by_group(scaler, clf, X: np.ndarray, labels: np.ndarray, groups: list) -> dict:
    """Accuracy/Wilson CI/AUC(+bootstrap CI) overall and per non-None group value."""
    out = {}
    for name, idx in [("__all__", list(range(len(labels))))] + [
        (g, [i for i, gg in enumerate(groups) if gg == g]) for g in sorted({g for g in groups if g is not None})
    ]:
        yg = labels[idx]
        s = core.score(scaler, clf, X[idx])
        n_correct = int(((s > 0).astype(int) == yg).sum())
        auc = core.safe_auc(yg, s)
        out[name] = {
            "n": len(idx), "n_pos": int((yg == 1).sum()),
            "accuracy": n_correct / len(idx) if idx else 0.0, "ci": core.wilson_ci(n_correct, len(idx)),
            "auc_roc": auc, "auc_roc_ci": bootstrap_auc_ci(yg, s) if auc is not None else None,
        }
    return out


def score_heldout(model, tokenizer, model_config, payload: dict, args, pooling) -> dict:
    results = {}
    for name in args.heldout_targets.split(","):
        path = TARGETS[name]
        print(f"\n{'=' * 70}\nHeld-out target: {name}  ({path})\n{'=' * 70}")
        if name == "moral":
            recs, labels = load_moral_pairs(path)
            if args.max_per_class is not None:
                keep = subsample_indices(labels, args.max_per_class, args.seed)
                recs, labels = [recs[i] for i in keep], labels[keep]
            residual = extract_moral_two_pass(model, tokenizer, model_config, recs, args, pooling)
            categories = [None] * len(recs)
        else:
            texts, labels, categories = load_target(path)
            if args.max_per_class is not None:
                keep = subsample_indices(labels, args.max_per_class, args.seed)
                texts, labels, categories = [texts[i] for i in keep], labels[keep], [categories[i] for i in keep]
            residual = core.collect_residual_only(model, tokenizer, texts, model_config, args.max_length, pooling)

        results[name] = {}
        for layer, entry in payload["layers"].items():
            X, y, cats = residual[layer], labels, categories
            finite = np.isfinite(X).all(axis=1)
            if not finite.all():
                print(f"WARNING: dropping {int((~finite).sum())} row(s) with non-finite activations at layer {layer}")
                X, y, cats = X[finite], y[finite], [c for c, k in zip(cats, finite) if k]
            best = entry["by_C"][entry["best_C"]]
            scores = score_by_group(best["scaler"], best["model"], X, y, cats)
            results[name][str(layer)] = {"best_C": entry["best_C"], "n_dropped_nan": int((~finite).sum()), "scores": scores}
            overall = scores["__all__"]
            print(f"[{name} L{layer}] acc={overall['accuracy']:.3f} AUC={overall['auc_roc']} (n={overall['n']}, n_pos={overall['n_pos']})")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mixture", type=Path, default=DEFAULT_MIXTURE)
    parser.add_argument("--judged", type=Path, default=DEFAULT_JUDGED, help="Judged AITA-NTA-FLIP file for moral rows' two passes.")
    parser.add_argument("--raw-moral-prompts", action="store_true", help="Judged prompts are NOT chat-templated yet; wrap them.")
    parser.add_argument("--per-category", action="store_true", help="One sweep per category instead of the pooled mixture.")
    parser.add_argument("--heldout-targets", default=None, help="Comma-separated subset of: " + ",".join(TARGETS))
    parser.add_argument("--max-per-class", type=int, default=None, help="Stratified cap per label for held-out targets.")
    core.add_sweep_args(parser, default_model=GEMMA_MODEL)
    args = parser.parse_args()

    if args.heldout_targets:
        unknown = [t for t in args.heldout_targets.split(",") if t not in TARGETS]
        if unknown:
            parser.error(f"unknown held-out target(s) {unknown}; choose from {list(TARGETS)}")
        if args.per_category:
            parser.error("--heldout-targets scores the pooled mixture probe; drop --per-category")

    rows = read_jsonl(args.mixture)
    labels = np.array([int(r["label"]) for r in rows], dtype=int)
    categories = [r["category"] for r in rows]
    print(f"Loaded {len(rows)} rows from {args.mixture}; categories: {dict(Counter(categories))}")
    judged_by_row_id = load_judged_by_row_id(args.judged)
    missing = [i for i, r in enumerate(rows) if r["category"] == "moral" and r["meta"]["row_id"] not in judged_by_row_id]
    if missing:
        raise ValueError(f"{len(missing)} moral row_ids not found in {args.judged}")

    from utils.model import cleanup as cleanup_model

    print(f"\nLoading {args.model}...")
    model, tokenizer, model_config = core.load_model_and_config(args.model)
    layers = core.select_layers(model_config["n_layers"], args.layers)
    pooling = core.pooling_for(args.average)
    residual = extract_mixture(model, tokenizer, model_config, rows, judged_by_row_id, args, pooling)
    groups = np.array([mixture_group(r, i) for i, r in enumerate(rows)])
    meta = {"recipe": "mixture_moral_two_pass", "mixture": str(args.mixture), "judged": str(args.judged)}

    if args.per_category:
        cleanup_model(model, tokenizer)
        output = args.output
        for name in sorted(set(categories)):
            idx = np.array([i for i, c in enumerate(categories) if c == name])
            print(f"\n{'=' * 70}\nCategory: {name} ({len(idx)} rows)\n{'=' * 70}")
            args.output = output.with_name(f"{output.stem}_{name}{output.suffix}")
            cache_out = args.cache_out
            if cache_out:
                args.cache_out = cache_out.with_name(f"{cache_out.stem}_{name}{cache_out.suffix}")
            core.cache_and_train(args, residual[:, idx, :], labels[idx], groups[idx], layers, dict(meta, category=name))
            args.cache_out = cache_out
        return

    payload = core.cache_and_train(args, residual, labels, groups, layers, meta, category=np.array(categories))
    if args.heldout_targets:
        heldout = score_heldout(model, tokenizer, model_config, payload, args, pooling)
        out_path = args.output.with_name(args.output.name + ".heldout.json")
        out_path.write_text(json.dumps(heldout, indent=2), encoding="utf-8")
        print(f"\nheld-out scores -> {out_path}")
    cleanup_model(model, tokenizer)


if __name__ == "__main__":
    main()
