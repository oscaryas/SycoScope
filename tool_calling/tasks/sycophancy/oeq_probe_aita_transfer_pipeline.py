#!/usr/bin/env python3
"""
OEQ sycophancy-probe training (balanced, full dataset) + AITA transfer-accuracy
check. Successor to oeq_probe_pipeline.py's cross-dataset step, but replaces
activation steering + fresh generation + fresh judging with something more
direct: score the already-trained OEQ probe's classification accuracy against
activations from AITA responses whose sycophancy verdict we already have from
this session's judging runs. No steering, no generation, no live judge calls
(no ANTHROPIC_API_KEY needed).

Label sources (both pre-judged, from `results/`, not raw SAE/results/ generations):
  - OEQ: OEQ_social_sycophancy_judged.jsonl's "validation" metric (1 = emotionally
    validates the user, 0 = doesn't).
  - AITA positive (sycophantic): AITA-NTA-FLIP_moral_sycophancy_judged.jsonl rows
    where BOTH original_post_verdict and flipped_story_verdict are "NTA" (the
    strict moral-sycophancy-positive pairs) -- both sides' response text used.
  - AITA negative (not sycophantic): every individual "YTA"-verdict response,
    pooled from AITA-NTA-FLIP (either side, independently) and AITA-NTA-OG.

Both OEQ and AITA data get balanced via undersampling (equal classes) rather
than upweighting. The OEQ pool is further split into a train partition (used
for probe training + 5-fold CV) and a held-out partition (never touched during
training), both balanced, so we get a genuine held-out OEQ accuracy figure to
compare the AITA-transfer accuracy against.

Usage:
    python oeq_probe_aita_transfer_pipeline.py --dry-run   # data-only sanity check, no GPU
    python oeq_probe_aita_transfer_pipeline.py --model meta-llama/Meta-Llama-3-8B-Instruct
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for p in (HERE, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sycophancy_probes import (  # noqa: E402
    LinearProbe,
    _probe_accuracy,
    collect_activations,
    collect_sentence_activations,
    train_probe,
    save_probe_results,
    wilson_ci,
)
from social_sycophancy_judge import build_labeled_text  # noqa: E402

DEFAULT_OEQ_LABELS_PATH = HERE / "results" / "OEQ_social_sycophancy_judged.jsonl"
DEFAULT_AITA_FLIP_PATH = HERE / "results" / "AITA-NTA-FLIP_moral_sycophancy_judged.jsonl"
DEFAULT_AITA_OG_PATH = HERE / "results" / "AITA-NTA-OG_verdict_judged.jsonl"
# Separate default output dirs per pooling mode -- sentence-level results must
# not silently overwrite the already-completed whole-sequence-pooling run.
DEFAULT_OUTPUT_DIR_RESPONSE = HERE / "results" / "oeq_probe_aita_transfer"
DEFAULT_OUTPUT_DIR_SENTENCE = HERE / "results" / "oeq_probe_aita_transfer_sentence"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


# ---------------------------------------------------------------------------
# Data building (no GPU)
# ---------------------------------------------------------------------------

def _undersample_balance(pos: list, neg: list, rng: np.random.Generator) -> tuple:
    """Equal-class balancing: sample min(len(pos), len(neg)) from each class
    (a no-op shuffle for whichever side is already the minority)."""
    n_min = min(len(pos), len(neg))
    pos_bal = [pos[i] for i in rng.choice(len(pos), size=n_min, replace=False)]
    neg_bal = [neg[i] for i in rng.choice(len(neg), size=n_min, replace=False)]
    return pos_bal, neg_bal, n_min


def build_oeq_pool_and_split(labels_path: Path, holdout_frac: float, rng: np.random.Generator, max_per_class: int = None) -> dict:
    """
    Load OEQ_social_sycophancy_judged.jsonl, keep validation-labeled rows,
    undersample-balance, then split into a train partition and a held-out
    partition, BOTH stratified so both stay class-balanced.

    max_per_class: if given, caps each class's pool before balancing --
    smoke-test-only knob to exercise the full GPU code path on a handful of
    examples instead of the real ~311-per-class pool.
    """
    records = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("validation") is None:
                continue
            records.append(rec)
    records.sort(key=lambda r: (str(r["row_id"]), r["prompt_col"]))

    pos = [r for r in records if r["validation"] == 1]
    neg = [r for r in records if r["validation"] == 0]
    if max_per_class is not None:
        pos = pos[:max_per_class]
        neg = neg[:max_per_class]
    pos_bal, neg_bal, n_min = _undersample_balance(pos, neg, rng)

    n_heldout_per_class = round(holdout_frac * n_min)
    heldout = pos_bal[:n_heldout_per_class] + neg_bal[:n_heldout_per_class]
    train = pos_bal[n_heldout_per_class:] + neg_bal[n_heldout_per_class:]
    rng.shuffle(heldout)
    rng.shuffle(train)

    return {
        "train_records": train,
        "heldout_records": heldout,
        "n_pos_total": len(pos),
        "n_neg_total": len(neg),
        "n_judged_total": len(records),
        "n_min": n_min,
        "n_heldout_per_class": n_heldout_per_class,
    }


def build_aita_transfer_set(flip_path: Path, og_path: Path, rng: np.random.Generator, max_per_class: int = None) -> dict:
    """
    Positive (label=1): AITA-NTA-FLIP rows where BOTH sides are NTA-verdict
    (the strict moral-sycophancy-positive pairs) -- both sides' response text.
    Negative (label=0): every individual YTA-verdict response, pooled from
    AITA-NTA-FLIP (either side, independently) and AITA-NTA-OG. Undersampled
    to match the positive count.

    max_per_class: smoke-test-only cap, applied to positive/negative-candidate
    pools before undersampling.
    """
    flip_rows = [json.loads(line) for line in open(flip_path, encoding="utf-8") if line.strip()]
    og_rows = [json.loads(line) for line in open(og_path, encoding="utf-8") if line.strip()]

    positive = []
    for r in flip_rows:
        if r["original_post_verdict"] == "NTA" and r["flipped_story_verdict"] == "NTA":
            positive.append({
                "prompt": r["original_post_prompt"], "response": r["original_post_response"],
                "label": 1, "row_id": r["row_id"], "side": "original_post", "source": "FLIP",
            })
            positive.append({
                "prompt": r["flipped_story_prompt"], "response": r["flipped_story_response"],
                "label": 1, "row_id": r["row_id"], "side": "flipped_story", "source": "FLIP",
            })

    negative_candidates = []
    for r in flip_rows:
        if r["original_post_verdict"] == "YTA":
            negative_candidates.append({
                "prompt": r["original_post_prompt"], "response": r["original_post_response"],
                "label": 0, "row_id": r["row_id"], "side": "original_post", "source": "FLIP",
            })
        if r["flipped_story_verdict"] == "YTA":
            negative_candidates.append({
                "prompt": r["flipped_story_prompt"], "response": r["flipped_story_response"],
                "label": 0, "row_id": r["row_id"], "side": "flipped_story", "source": "FLIP",
            })
    for r in og_rows:
        if r["verdict"] == "YTA":
            negative_candidates.append({
                "prompt": r["prompt"], "response": r["response"],
                "label": 0, "row_id": r["row_id"], "side": r.get("prompt_col", "original_post"), "source": "OG",
            })

    if max_per_class is not None:
        positive = positive[:max_per_class]
        negative_candidates = negative_candidates[:max_per_class]

    n_neg = min(len(positive), len(negative_candidates))
    neg_idx = rng.choice(len(negative_candidates), size=n_neg, replace=False)
    negative = [negative_candidates[i] for i in neg_idx]

    combined = positive + negative
    rng.shuffle(combined)

    return {
        "records": combined,
        "n_pos": len(positive),
        "n_neg": len(negative),
        "n_neg_candidates": len(negative_candidates),
    }


# ---------------------------------------------------------------------------
# Activation extraction + probe training (GPU)
# ---------------------------------------------------------------------------

def extract_all_activations(model, tokenizer, model_config: dict, oeq_split: dict, aita_set: dict) -> tuple:
    """Single collect_activations call over OEQ-train + OEQ-heldout + AITA-transfer
    (in that order), sliced back into three views afterward -- one model
    forward-pass sweep instead of three separate hook-attach/detach cycles."""
    train_texts = [build_labeled_text(tokenizer, r) for r in oeq_split["train_records"]]
    heldout_texts = [build_labeled_text(tokenizer, r) for r in oeq_split["heldout_records"]]
    aita_texts = [build_labeled_text(tokenizer, r) for r in aita_set["records"]]

    n_train, n_heldout, n_aita = len(train_texts), len(heldout_texts), len(aita_texts)
    all_texts = train_texts + heldout_texts + aita_texts
    print(f"Extracting activations for {len(all_texts)} examples "
          f"({n_train} OEQ-train + {n_heldout} OEQ-heldout + {n_aita} AITA-transfer)...")
    activations = collect_activations(model, tokenizer, all_texts, model_config)

    slices = {
        "train": slice(0, n_train),
        "heldout": slice(n_train, n_train + n_heldout),
        "aita": slice(n_train + n_heldout, n_train + n_heldout + n_aita),
    }

    y_train = np.array([r["validation"] for r in oeq_split["train_records"]], dtype=np.float32)
    y_heldout = np.array([r["validation"] for r in oeq_split["heldout_records"]], dtype=np.float32)
    y_aita = np.array([r["label"] for r in aita_set["records"]], dtype=np.float32)
    labels_all = np.concatenate([y_train, y_heldout, y_aita])

    return activations, slices, labels_all


def extract_all_sentence_activations(model, tokenizer, model_config: dict, oeq_split: dict, aita_set: dict) -> tuple:
    """
    Sentence-level counterpart to extract_all_activations: one forward pass
    per RESPONSE (not per sentence -- preserves true generation context),
    but each response's individual sentences get their own pooled vector
    and inherit the parent response's label.

    collect_sentence_activations processes records in order and emits all
    of one record's sentences contiguously, so -- since all_records is
    itself ordered train-then-heldout-then-aita -- the resulting sentence
    rows stay contiguous by partition too. That means simple slice() objects
    still work here, just with boundaries computed from actual sentence
    counts (which vary per response) instead of a fixed record-count
    multiplier. Everything downstream (train_probes_on_train_slice,
    score_generalization, plot_generalization) is unchanged: none of it
    cares whether a row is a whole response or one sentence.
    """
    train_records = oeq_split["train_records"]
    heldout_records = oeq_split["heldout_records"]
    aita_records = aita_set["records"]
    n_train_rec = len(train_records)
    n_heldout_rec = len(heldout_records)

    all_records = train_records + heldout_records + aita_records
    print(
        f"Extracting sentence-level activations for {len(all_records)} records "
        f"({n_train_rec} OEQ-train + {n_heldout_rec} OEQ-heldout + {len(aita_records)} AITA-transfer)..."
    )
    sentence_records, activations = collect_sentence_activations(model, tokenizer, all_records, model_config)

    n_train_sent = sum(1 for sr in sentence_records if sr["parent_index"] < n_train_rec)
    n_heldout_sent = sum(
        1 for sr in sentence_records if n_train_rec <= sr["parent_index"] < n_train_rec + n_heldout_rec
    )
    n_aita_sent = len(sentence_records) - n_train_sent - n_heldout_sent

    slices = {
        "train": slice(0, n_train_sent),
        "heldout": slice(n_train_sent, n_train_sent + n_heldout_sent),
        "aita": slice(n_train_sent + n_heldout_sent, len(sentence_records)),
    }

    labels_all = np.array(
        [
            sr["validation"] if sr["parent_index"] < n_train_rec + n_heldout_rec else sr["label"]
            for sr in sentence_records
        ],
        dtype=np.float32,
    )

    sentence_counts = {
        "n_train_records": n_train_rec, "n_heldout_records": n_heldout_rec, "n_aita_records": len(aita_records),
        "n_train_sentences": n_train_sent, "n_heldout_sentences": n_heldout_sent, "n_aita_sentences": n_aita_sent,
    }
    return activations, slices, labels_all, sentence_counts


def _checkpoint_path(output_dir: Path, component: str) -> Path:
    return output_dir / f"_checkpoint_{component}.pkl"


def _load_checkpoint(output_dir: Path, component: str) -> dict:
    path = _checkpoint_path(output_dir, component)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def _save_checkpoint(output_dir: Path, component: str, checkpoint: dict) -> None:
    # Write to a temp file then atomically rename -- a session dying mid-write
    # (the whole reason this checkpointing exists) must never leave a
    # truncated/corrupt checkpoint file behind.
    path = _checkpoint_path(output_dir, component)
    tmp_path = path.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(checkpoint, f)
    tmp_path.replace(path)


def _checkpoint_entry_to_state(entry: dict) -> dict:
    return {
        "model_state": entry["model_state"], "proj_std": entry["proj_std"], "input_dim": entry["input_dim"],
        "ci_lower": entry["ci_lower"], "ci_upper": entry["ci_upper"],
    }


def _checkpoint_to_result_dicts(checkpoint: dict) -> tuple:
    accuracy_dict = {k: v["accuracy"] for k, v in checkpoint.items()}
    ci_dict = {k: (v["ci_lower"], v["ci_upper"]) for k, v in checkpoint.items()}
    state_dict = {k: _checkpoint_entry_to_state(v) for k, v in checkpoint.items()}
    auc_dict = {k: v["auc_roc"] for k, v in checkpoint.items()}
    auc_ci_dict = {k: (v["auc_ci_lower"], v["auc_ci_upper"]) for k, v in checkpoint.items()}
    return accuracy_dict, ci_dict, state_dict, auc_dict, auc_ci_dict


def train_mha_probes_checkpointed(mha_activations, labels, n_layers, n_heads, output_dir: Path, save_every: int = 5) -> tuple:
    """Same contract as sycophancy_probes.train_mha_probes, but checkpoints
    each (layer, head) probe to disk as it finishes and skips anything
    already checkpointed on entry -- so a Colab session dying mid-training
    (this pipeline's actual failure mode at full-dataset scale, where MHA
    training alone can take hours) costs at most `save_every` probes of
    lost work on resume, not the whole run."""
    checkpoint = _load_checkpoint(output_dir, "mha")
    n_total = n_layers * n_heads
    if checkpoint:
        print(f"  Resuming MHA: {len(checkpoint)}/{n_total} probes already checkpointed.")
    since_save = 0
    for layer in range(n_layers):
        for head in range(n_heads):
            key = (layer, head)
            if key in checkpoint:
                continue
            X = mha_activations[layer, head]
            # batch_size scaled with dataset size: a fixed batch_size=25 (tuned
            # for the original response-level scale, ~498 train examples ->
            # ~20 batches/epoch) left unscaled at sentence-level scale
            # (~11,000 train examples -> ~440 batches/epoch, ~22x more
            # gradient steps per probe for no accuracy benefit) projected to
            # ~39 hours for MHA training alone. This keeps batches/epoch
            # roughly constant regardless of scale.
            result = train_probe(X, labels, balance_method="undersample", batch_size=max(25, len(labels) // 20))
            w = result["model_state"]["linear.weight"][0].numpy()
            direction = w / (np.linalg.norm(w) + 1e-8)
            proj_std = float(np.std(X @ direction))
            checkpoint[key] = {
                "accuracy": result["accuracy"], "ci_lower": result["ci_lower"], "ci_upper": result["ci_upper"],
                "auc_roc": result["auc_roc"], "auc_ci_lower": result["auc_ci_lower"], "auc_ci_upper": result["auc_ci_upper"],
                "model_state": result["model_state"], "proj_std": proj_std, "input_dim": result["input_dim"],
            }
            since_save += 1
            print(
                f"  MHA layer={layer} head={head}: acc={result['accuracy']:.3f} "
                f"CI=[{result['ci_lower']:.3f},{result['ci_upper']:.3f}] AUC={result['auc_roc']:.3f} "
                f"[{len(checkpoint)}/{n_total}]"
            )
            if since_save >= save_every:
                _save_checkpoint(output_dir, "mha", checkpoint)
                since_save = 0
    _save_checkpoint(output_dir, "mha", checkpoint)
    return _checkpoint_to_result_dicts(checkpoint)


def _train_layer_probes_checkpointed(activations, labels, n_layers, output_dir: Path, component: str, save_every: int = 5) -> tuple:
    """Shared checkpointed-training body for MLP/residual (keyed by plain
    layer int, unlike MHA's (layer, head) tuple)."""
    checkpoint = _load_checkpoint(output_dir, component)
    if checkpoint:
        print(f"  Resuming {component.upper()}: {len(checkpoint)}/{n_layers} layers already checkpointed.")
    since_save = 0
    for layer in range(n_layers):
        if layer in checkpoint:
            continue
        X = activations[layer]
        result = train_probe(X, labels, balance_method="undersample", batch_size=max(25, len(labels) // 20))
        w = result["model_state"]["linear.weight"][0].numpy()
        direction = w / (np.linalg.norm(w) + 1e-8)
        proj_std = float(np.std(X @ direction))
        checkpoint[layer] = {
            "accuracy": result["accuracy"], "ci_lower": result["ci_lower"], "ci_upper": result["ci_upper"],
            "auc_roc": result["auc_roc"], "auc_ci_lower": result["auc_ci_lower"], "auc_ci_upper": result["auc_ci_upper"],
            "model_state": result["model_state"], "proj_std": proj_std, "input_dim": result["input_dim"],
        }
        since_save += 1
        print(
            f"  {component.upper()} layer={layer}: acc={result['accuracy']:.3f} "
            f"CI=[{result['ci_lower']:.3f},{result['ci_upper']:.3f}] AUC={result['auc_roc']:.3f} "
            f"[{len(checkpoint)}/{n_layers}]"
        )
        if since_save >= save_every:
            _save_checkpoint(output_dir, component, checkpoint)
            since_save = 0
    _save_checkpoint(output_dir, component, checkpoint)
    return _checkpoint_to_result_dicts(checkpoint)


def train_probes_on_train_slice(
    activations: dict, slices: dict, labels_all: np.ndarray, model_config: dict, output_dir: Path,
) -> dict:
    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]
    train_slice = slices["train"]
    y_train = labels_all[train_slice]

    mha_train = activations["mha"][:, :, train_slice, :]
    mlp_train = activations["mlp"][:, train_slice, :]
    res_train = activations["residual"][:, train_slice, :]

    print("\n=== Training probes on OEQ-train partition (balance_method=undersample, checkpointed) ===")
    mha_acc, mha_ci, mha_states, mha_auc, mha_auc_ci = train_mha_probes_checkpointed(
        mha_train, y_train, n_layers, n_heads, output_dir
    )
    mlp_acc, mlp_ci, mlp_states, mlp_auc, mlp_auc_ci = _train_layer_probes_checkpointed(
        mlp_train, y_train, n_layers, output_dir, "mlp"
    )
    res_acc, res_ci, res_states, res_auc, res_auc_ci = _train_layer_probes_checkpointed(
        res_train, y_train, n_layers, output_dir, "residual"
    )
    return {
        "mha_accuracy": mha_acc, "mha_ci": mha_ci, "mha_states": mha_states, "mha_auc": mha_auc, "mha_auc_ci": mha_auc_ci,
        "mlp_accuracy": mlp_acc, "mlp_ci": mlp_ci, "mlp_states": mlp_states, "mlp_auc": mlp_auc, "mlp_auc_ci": mlp_auc_ci,
        "residual_accuracy": res_acc, "residual_ci": res_ci, "residual_states": res_states, "residual_auc": res_auc, "residual_auc_ci": res_auc_ci,
    }


# ---------------------------------------------------------------------------
# Generalization scoring: reconstruct each saved probe, score against new activations.
# No such "score a saved probe on new data" utility exists in sycophancy_probes.py
# (confirmed by full-file read) -- mirrors the state-reconstruction pattern
# sycophancy_steering.load_steering_vectors already uses for a different purpose.
# ---------------------------------------------------------------------------

def score_probe(state: dict, X: np.ndarray, y: np.ndarray) -> tuple:
    """Returns (accuracy, (ci_lower, ci_upper)) -- Wilson CI on this single
    scoring pass's correct/total count, same CI convention train_probe uses
    (sycophancy_probes.py's wilson_ci) for its pooled 5-fold CV accuracy."""
    model_state = state["model_state"]
    input_dim = model_state["linear.weight"].shape[1]
    probe = LinearProbe(input_dim)
    probe.load_state_dict(model_state)
    probe.eval()
    n_correct, n_total, accuracy = _probe_accuracy(probe, X, y)
    ci = wilson_ci(n_correct, n_total)
    return accuracy, ci


def score_all_layers(states: dict, component_activations: np.ndarray, idx_slice: slice, y: np.ndarray, component: str) -> tuple:
    """Returns (accuracy_dict, ci_dict), both keyed like `states`."""
    acc_out, ci_out = {}, {}
    for key, state in states.items():
        if component == "mha":
            layer, head = key
            X = component_activations[layer, head, idx_slice, :]
        else:
            X = component_activations[key, idx_slice, :]
        accuracy, ci = score_probe(state, X, y)
        acc_out[key] = accuracy
        ci_out[key] = ci
    return acc_out, ci_out


def score_generalization(probe_results: dict, activations: dict, slices: dict, labels_all: np.ndarray) -> tuple:
    """Returns (gen_scores, gen_ci), each {"heldout"|"aita": {"mha"|"mlp"|"residual": {key: value}}}."""
    gen_scores, gen_ci = {}, {}
    for eval_name in ("heldout", "aita"):
        idx_slice = slices[eval_name]
        y = labels_all[idx_slice]
        gen_scores[eval_name] = {}
        gen_ci[eval_name] = {}
        for component in ("mha", "mlp", "residual"):
            acc, ci = score_all_layers(
                probe_results[f"{component}_states"], activations[component], idx_slice, y, component
            )
            gen_scores[eval_name][component] = acc
            gen_ci[eval_name][component] = ci
    return gen_scores, gen_ci


# ---------------------------------------------------------------------------
# Plotting -- per component panel: train CV accuracy, OEQ-heldout accuracy,
# AITA-transfer accuracy, all by layer. For MHA, all three series report the
# value at whichever head the TRAIN CV accuracy picked best for that layer,
# so a line stays paired with one head rather than being independently
# re-maximized per series (same convention as oeq_probe_pipeline.py's
# _best_mha_per_layer).
# ---------------------------------------------------------------------------

def _best_mha_per_layer(mha_dict: dict, mha_acc_for_argmax: dict, n_layers: int) -> list:
    out = []
    for l in range(n_layers):
        heads_at_layer = [(h, acc) for (layer, h), acc in mha_acc_for_argmax.items() if layer == l]
        if not heads_at_layer:
            out.append(None)
            continue
        best_head = max(heads_at_layer, key=lambda t: t[1])[0]
        out.append(mha_dict.get((l, best_head)))
    return out


def plot_generalization(
    probe_results: dict, gen_scores: dict, gen_ci: dict, n_layers: int, out_path: Path, pooling: str = "response",
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    components = [
        ("mha", "MHA (best head, by train accuracy)"),
        ("mlp", "MLP"),
        ("residual", "Residual"),
    ]
    layers = range(n_layers)
    series_style = [
        ("Train (5-fold CV)", "train_acc", "train_ci", "#0072B2"),
        ("OEQ held-out", "heldout_acc", "heldout_ci", "#009E73"),
        ("AITA transfer", "aita_acc", "aita_ci", "#D55E00"),
    ]

    for ax, (component, title) in zip(axes, components):
        train_acc_dict = probe_results[f"{component}_accuracy"]
        train_ci_dict = probe_results[f"{component}_ci"]
        heldout_acc_dict = gen_scores["heldout"][component]
        heldout_ci_dict = gen_ci["heldout"][component]
        aita_acc_dict = gen_scores["aita"][component]
        aita_ci_dict = gen_ci["aita"][component]

        if component == "mha":
            vals = {
                "train_acc": _best_mha_per_layer(train_acc_dict, train_acc_dict, n_layers),
                "train_ci": _best_mha_per_layer(train_ci_dict, train_acc_dict, n_layers),
                "heldout_acc": _best_mha_per_layer(heldout_acc_dict, train_acc_dict, n_layers),
                "heldout_ci": _best_mha_per_layer(heldout_ci_dict, train_acc_dict, n_layers),
                "aita_acc": _best_mha_per_layer(aita_acc_dict, train_acc_dict, n_layers),
                "aita_ci": _best_mha_per_layer(aita_ci_dict, train_acc_dict, n_layers),
            }
        else:
            vals = {
                "train_acc": [train_acc_dict.get(l) for l in layers],
                "train_ci": [train_ci_dict.get(l) for l in layers],
                "heldout_acc": [heldout_acc_dict.get(l) for l in layers],
                "heldout_ci": [heldout_ci_dict.get(l) for l in layers],
                "aita_acc": [aita_acc_dict.get(l) for l in layers],
                "aita_ci": [aita_ci_dict.get(l) for l in layers],
            }

        for label, acc_key, ci_key, color in series_style:
            acc_vals = vals[acc_key]
            (line,) = ax.plot(layers, acc_vals, "o-", label=label, color=color)
            ci_vals = vals[ci_key]
            lo = [p[0] if p else v for p, v in zip(ci_vals, acc_vals)]
            hi = [p[1] if p else v for p, v in zip(ci_vals, acc_vals)]
            ax.fill_between(layers, lo, hi, alpha=0.15, color=color)

        ax.axhline(0.5, color="gray", linestyle="--", label="Chance")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy (shaded = Wilson CI)")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=8.5)

    pooling_label = "sentence-level pooling" if pooling == "sentence" else "whole-sequence pooling"
    fig.suptitle(f"OEQ probe ({pooling_label}): train vs. held-out vs. AITA-transfer accuracy by layer")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _stringify_keys(d: dict) -> dict:
    return {str(k): v for k, v in d.items()}


def write_summary(
    out_dir: Path, model_name: str, oeq_split: dict, aita_set: dict, probe_results: dict, gen_scores: dict,
    gen_ci: dict, pooling: str = "response", sentence_counts: dict = None,
):
    def best_with_ci(acc_dict, ci_dict):
        best_key = max(acc_dict, key=acc_dict.get)
        acc = acc_dict[best_key]
        lo, hi = ci_dict[best_key]
        return f"{acc:.3f} [{lo:.3f},{hi:.3f}]"

    def best_line(component):
        train_str = best_with_ci(probe_results[f"{component}_accuracy"], probe_results[f"{component}_ci"])
        heldout_str = best_with_ci(gen_scores["heldout"][component], gen_ci["heldout"][component])
        aita_str = best_with_ci(gen_scores["aita"][component], gen_ci["aita"][component])
        return f"- **{component.upper()}**: train(CV)={train_str}  OEQ-heldout={heldout_str}  AITA-transfer={aita_str}"

    pooling_desc = (
        "mean over each SENTENCE's own tokens within the response (prompt excluded); "
        "one example per sentence, inheriting its parent response's label"
        if pooling == "sentence" else
        "mean over the entire chat-formatted sequence (prompt + response)"
    )

    lines = [
        f"# OEQ Probe (balanced, full dataset, {pooling}-level pooling) + AITA Transfer Accuracy -- Summary",
        "",
        f"Model: `{model_name}` * Label: ELEPHANT \"validation\" metric * "
        "balance_method=\"undersample\" only * direction method: linear probe "
        "(`nn.Linear` + `BCEWithLogitsLoss`, Adam lr=0.001, 25 epochs, batch size 25, "
        f"5-fold stratified CV on the train partition) * pooling: {pooling_desc}.",
        "",
        "## Class balance",
        "",
        f"OEQ judged pool: {oeq_split['n_pos_total']} pos / {oeq_split['n_neg_total']} neg "
        f"out of {oeq_split['n_judged_total']} judged. Balanced pool size (per class): "
        f"{oeq_split['n_min']}. Held out (per class): {oeq_split['n_heldout_per_class']} "
        f"-> train={2*(oeq_split['n_min']-oeq_split['n_heldout_per_class'])} responses, "
        f"heldout={2*oeq_split['n_heldout_per_class']} responses.",
        "",
        f"AITA transfer set: {aita_set['n_pos']} pos (AITA-NTA-FLIP both-NTA pairs) / "
        f"{aita_set['n_neg']} neg (undersampled from {aita_set['n_neg_candidates']} individual "
        "YTA-verdict candidates across AITA-NTA-FLIP + AITA-NTA-OG) responses.",
    ]

    if pooling == "sentence" and sentence_counts:
        lines += [
            "",
            "Sentence-level expansion (balancing was done at the response level, above, "
            "*before* splitting into sentences -- per-class sentence counts can drift from "
            "exactly 50/50 if one class's responses tend to run longer/shorter):",
            "",
            f"- Train: {sentence_counts['n_train_records']} responses -> "
            f"{sentence_counts['n_train_sentences']} sentences",
            f"- OEQ held-out: {sentence_counts['n_heldout_records']} responses -> "
            f"{sentence_counts['n_heldout_sentences']} sentences",
            f"- AITA transfer: {sentence_counts['n_aita_records']} responses -> "
            f"{sentence_counts['n_aita_sentences']} sentences",
        ]

    lines += [
        "",
        "## Best-layer accuracy by series",
        "",
        best_line("mha"),
        best_line("mlp"),
        best_line("residual"),
        "",
        "## Plots",
        "",
        "- `plots/generalization_by_layer.png` -- train (5-fold CV) / OEQ-held-out / "
        "AITA-transfer accuracy, all layers, per component.",
        "",
        "Full probe checkpoints: `mha_probe_weights.pth`/`mha_projection_stds.pt` "
        "(and mlp/residual equivalents), covering every (layer, head)/layer. "
        "Per-layer accuracy tables: `oeq_heldout_accuracy.json`, `aita_transfer_accuracy.json`, "
        "with matching Wilson-CI tables `oeq_heldout_ci.json`, `aita_transfer_ci.json`.",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oeq-labels-path", type=Path, default=DEFAULT_OEQ_LABELS_PATH)
    parser.add_argument("--aita-flip-path", type=Path, default=DEFAULT_AITA_FLIP_PATH)
    parser.add_argument("--aita-og-path", type=Path, default=DEFAULT_AITA_OG_PATH)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to results/oeq_probe_aita_transfer(_sentence) depending on --pooling.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Build/report the data splits only, no model load or GPU work.")
    parser.add_argument(
        "--pooling", choices=["response", "sentence"], default="sentence",
        help="'response' (original): mean over the whole chat-formatted prompt+response sequence, "
        "one example per response. 'sentence': mean over each sentence's own tokens within just the "
        "response (prompt excluded), one example per sentence, inheriting the parent response's label.",
    )
    parser.add_argument(
        "--smoke-test", type=int, default=None, metavar="N",
        help="Cap each class's pool to N examples before balancing, to exercise the full GPU "
        "code path (activation extraction, probe training, scoring, plotting) on a small, "
        "fast slice before committing to the full run.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR_SENTENCE if args.pooling == "sentence" else DEFAULT_OUTPUT_DIR_RESPONSE

    rng = np.random.default_rng(args.seed)

    print("Building OEQ balanced pool + train/heldout split...")
    oeq_split = build_oeq_pool_and_split(args.oeq_labels_path, args.holdout_frac, rng, max_per_class=args.smoke_test)
    print(
        f"  OEQ judged: {oeq_split['n_pos_total']} pos / {oeq_split['n_neg_total']} neg "
        f"out of {oeq_split['n_judged_total']}"
    )
    print(f"  Balanced pool (per class): {oeq_split['n_min']}")
    print(f"  Train: {len(oeq_split['train_records'])}  Heldout: {len(oeq_split['heldout_records'])}")

    print("\nBuilding AITA transfer set...")
    aita_set = build_aita_transfer_set(args.aita_flip_path, args.aita_og_path, rng, max_per_class=args.smoke_test)
    print(
        f"  Positive (FLIP both-NTA pairs): {aita_set['n_pos']}  "
        f"Negative (undersampled from {aita_set['n_neg_candidates']} YTA candidates): {aita_set['n_neg']}"
    )
    print(f"  Total transfer set: {len(aita_set['records'])}")

    if args.dry_run:
        print("\n--dry-run: stopping before model load / activation extraction.")
        return

    from utils.model import load_model_and_tokenizer, cleanup as cleanup_model
    from sycophancy_model_registry import get_model_config

    print(f"\nLoading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    model_config["model_name"] = args.model

    # Created before training (not just before saving) -- checkpointing needs
    # the directory to exist while training is still in progress, since a
    # Colab session can die mid-training at full-dataset scale (this
    # pipeline's actual failure mode: MHA training alone can take hours).
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sentence_counts = None
    if args.pooling == "sentence":
        activations, slices, labels_all, sentence_counts = extract_all_sentence_activations(
            model, tokenizer, model_config, oeq_split, aita_set
        )
    else:
        activations, slices, labels_all = extract_all_activations(model, tokenizer, model_config, oeq_split, aita_set)

    probe_results = train_probes_on_train_slice(activations, slices, labels_all, model_config, args.output_dir)

    print("\n=== Scoring generalization (OEQ held-out + AITA transfer) ===")
    gen_scores, gen_ci = score_generalization(probe_results, activations, slices, labels_all)

    save_probe_results(probe_results, str(args.output_dir), model_name=args.model)

    # Checkpoints duplicate every probe's full model_state, already persisted
    # by save_probe_results above in the standard {component}_probe_weights.pth
    # format -- safe to remove now that the run has reached this point.
    for component in ("mha", "mlp", "residual"):
        _checkpoint_path(args.output_dir, component).unlink(missing_ok=True)

    with open(args.output_dir / "oeq_heldout_accuracy.json", "w") as f:
        json.dump({c: _stringify_keys(gen_scores["heldout"][c]) for c in ("mha", "mlp", "residual")}, f, indent=2)
    with open(args.output_dir / "aita_transfer_accuracy.json", "w") as f:
        json.dump({c: _stringify_keys(gen_scores["aita"][c]) for c in ("mha", "mlp", "residual")}, f, indent=2)
    with open(args.output_dir / "oeq_heldout_ci.json", "w") as f:
        json.dump({c: _stringify_keys(gen_ci["heldout"][c]) for c in ("mha", "mlp", "residual")}, f, indent=2)
    with open(args.output_dir / "aita_transfer_ci.json", "w") as f:
        json.dump({c: _stringify_keys(gen_ci["aita"][c]) for c in ("mha", "mlp", "residual")}, f, indent=2)

    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_generalization(
        probe_results, gen_scores, gen_ci, model_config["n_layers"], plots_dir / "generalization_by_layer.png",
        pooling=args.pooling,
    )

    write_summary(
        args.output_dir, args.model, oeq_split, aita_set, probe_results, gen_scores, gen_ci,
        pooling=args.pooling, sentence_counts=sentence_counts,
    )

    del activations
    cleanup_model(model, tokenizer)
    print(f"\nDone. Results under {args.output_dir}/")


if __name__ == "__main__":
    main()
