#!/usr/bin/env python3
"""
Train MHA, MLP, and residual stream probes matching the paper's setup:
  - PyTorch nn.Linear probes with BCEWithLogitsLoss
  - Adam optimizer, lr=0.001, 25 epochs, batch size 25, 80/20 split

Activation extraction uses hooks registered by sycophancy_model_registry.

Usage (from agent):
    from sycophancy_probes import extract_and_train_all, save_probe_results
    results = extract_and_train_all(model, tokenizer, texts, labels, model_config)
    save_probe_results(results, "final_probe")
"""
import gc
import json
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import Adam


# ---------------------------------------------------------------------------
# Statistics utilities
# ---------------------------------------------------------------------------

def wilson_ci(n_correct: int, n_total: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a proportion. Fast — no resampling needed."""
    if n_total == 0:
        return (0.0, 0.0)
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_ci(
    values: np.ndarray,
    stat_fn=np.mean,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> tuple:
    """Bootstrap percentile CI for any statistic over a 1-D array."""
    rng = np.random.default_rng(42)
    boots = [stat_fn(rng.choice(values, size=len(values), replace=True))
             for _ in range(n_bootstrap)]
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return (float(lo), float(hi))


def bootstrap_delta_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> tuple:
    """Bootstrap CI on the difference mean(b) - mean(a)."""
    rng = np.random.default_rng(42)
    deltas = [
        np.mean(rng.choice(b, size=len(b), replace=True)) -
        np.mean(rng.choice(a, size=len(a), replace=True))
        for _ in range(n_bootstrap)
    ]
    lo = np.percentile(deltas, 100 * alpha / 2)
    hi = np.percentile(deltas, 100 * (1 - alpha / 2))
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Probe model
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _fit_probe(X: np.ndarray, y: np.ndarray, input_dim: int, n_epochs: int, batch_size: int, lr: float) -> nn.Module:
    """Fit one LinearProbe on the given (train) data and return it."""
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)

    # Class-weighted loss: if the split is skewed (e.g. 85% label=1), pos_weight
    # scales the positive term so the minority class's gradient isn't drowned
    # out and the probe can't trivially win by always predicting the majority.
    n_pos = float((y_t == 1).sum())
    n_neg = float((y_t == 0).sum())
    pos_weight = torch.tensor(n_neg / n_pos) if n_pos > 0 and n_neg > 0 else torch.tensor(1.0)

    probe = LinearProbe(input_dim)
    optimizer = Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    probe.train()
    for _ in range(n_epochs):
        perm = torch.randperm(len(X_t))
        for start in range(0, len(X_t), batch_size):
            batch_idx = perm[start : start + batch_size]
            loss = criterion(probe(X_t[batch_idx]), y_t[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    probe.eval()
    return probe


def _probe_accuracy(probe: nn.Module, X: np.ndarray, y: np.ndarray) -> tuple:
    """Returns (n_correct, n_total, accuracy) for probe on (X, y)."""
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)
    with torch.no_grad():
        preds = (probe(X_t) > 0).float()
        n_correct = int((preds == y_t).sum().item())
    n_total = len(y_t)
    return n_correct, n_total, n_correct / n_total if n_total else 0.0


def _probe_scores(probe: nn.Module, X: np.ndarray) -> np.ndarray:
    """Raw pre-threshold logits -- roc_auc_score needs continuous scores, not
    the thresholded 0/1 predictions _probe_accuracy computes."""
    X_t = torch.FloatTensor(X)
    with torch.no_grad():
        return probe(X_t).numpy()


def _fold_auc(y_true: np.ndarray, scores: np.ndarray):
    """None if a fold's test split has only one class present (roc_auc_score
    is undefined there) -- stratified folds make this rare but not impossible
    at small n, so this is a real guard, not defensive boilerplate."""
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def _stratified_folds(y: np.ndarray, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Assign each example to one of n_folds folds, preserving class balance per fold."""
    fold_of = np.empty(len(y), dtype=int)
    for cls in np.unique(y):
        cls_idx = np.nonzero(y == cls)[0]
        rng.shuffle(cls_idx)
        fold_of[cls_idx] = np.arange(len(cls_idx)) % n_folds
    return fold_of


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 25,
    batch_size: int = 25,
    lr: float = 0.001,
    n_folds: int = 5,
    seed: Optional[int] = None,
    balance_method: str = "undersample",
) -> dict:
    """
    Train a linear probe on activations X with binary labels y, evaluated
    with n_folds-fold stratified cross-validation: every example serves as
    test data in exactly one fold, so the reported accuracy/CI reflect the
    whole dataset rather than one lucky/unlucky 80/20 split. The returned
    probe weights ("model_state", used downstream for steering directions)
    are refit on the *full* (balanced or reweighted) dataset after CV, per
    standard practice -- cross-validate to evaluate, refit on all data to
    deploy.

    balance_method controls how class imbalance is handled before fitting:
    - "undersample" (default): drop majority-class examples down to
      min(n_pos, n_neg) so the two classes are equal size -- a skewed input
      (e.g. 80 pos / 910 neg) doesn't just get trivially "solved" by the
      probe predicting the majority class every time. Every example that
      survives is real data, but 830 majority-class examples go unused.
    - "upweight": keep every example; class imbalance is instead corrected
      via _fit_probe's existing pos_weight-scaled BCEWithLogitsLoss (already
      computed per training fold from whatever y ends up in that fold). Uses
      the full dataset -- no examples discarded -- at the cost of a noisier
      gradient signal from the minority class's small absolute count.

    Returns dict with keys: accuracy (pooled across folds), fold_accuracies
    (list, one per fold), train_accuracy (mean across folds' training
    portions), ci_lower, ci_upper (Wilson CI on pooled fold predictions),
    n_test (total pooled test examples across folds), model_state,
    input_dim, auc_roc (mean of per-fold AUC-ROC on held-out scores,
    diagnostic only -- direction selection stays accuracy-based), fold_aucs,
    auc_ci_lower/auc_ci_upper (bootstrap CI over fold_aucs).
    """
    if balance_method not in ("undersample", "upweight"):
        raise ValueError(f"balance_method must be 'undersample' or 'upweight', got {balance_method!r}")

    rng = np.random.default_rng(seed)

    if balance_method == "undersample":
        pos_idx = np.nonzero(y == 1)[0]
        neg_idx = np.nonzero(y == 0)[0]
        n_min = min(len(pos_idx), len(neg_idx))
        keep_idx = np.concatenate([
            rng.choice(pos_idx, size=n_min, replace=False),
            rng.choice(neg_idx, size=n_min, replace=False),
        ])
        rng.shuffle(keep_idx)
        X, y = X[keep_idx], y[keep_idx]

    input_dim = X.shape[-1]
    fold_of = _stratified_folds(y, n_folds, rng)

    fold_accuracies = []
    train_accuracies = []
    fold_aucs = []
    total_correct = 0
    total_test = 0
    for fold in range(n_folds):
        test_mask = fold_of == fold
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        probe = _fit_probe(X[train_mask], y[train_mask], input_dim, n_epochs, batch_size, lr)
        _, _, train_acc = _probe_accuracy(probe, X[train_mask], y[train_mask])
        n_correct, n_test_fold, fold_acc = _probe_accuracy(probe, X[test_mask], y[test_mask])
        fold_auc = _fold_auc(y[test_mask], _probe_scores(probe, X[test_mask]))

        fold_accuracies.append(fold_acc)
        train_accuracies.append(train_acc)
        if fold_auc is not None:
            fold_aucs.append(fold_auc)
        total_correct += n_correct
        total_test += n_test_fold

    # Refit on all (balanced) data for the returned probe weights.
    final_probe = _fit_probe(X, y, input_dim, n_epochs, batch_size, lr)

    ci_lower, ci_upper = wilson_ci(total_correct, total_test)
    if fold_aucs:
        auc_roc = float(np.mean(fold_aucs))
        auc_ci_lower, auc_ci_upper = bootstrap_ci(np.array(fold_aucs))
    else:
        auc_roc, auc_ci_lower, auc_ci_upper = 0.5, 0.5, 0.5

    return {
        "accuracy": total_correct / total_test if total_test else 0.0,
        "fold_accuracies": fold_accuracies,
        "train_accuracy": float(np.mean(train_accuracies)) if train_accuracies else 0.0,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_test": total_test,
        "model_state": final_probe.state_dict(),
        "input_dim": input_dim,
        "auc_roc": auc_roc,
        "fold_aucs": fold_aucs,
        "auc_ci_lower": auc_ci_lower,
        "auc_ci_upper": auc_ci_upper,
    }


# ---------------------------------------------------------------------------
# Per-component probe training
# ---------------------------------------------------------------------------

def train_mha_probes(
    mha_activations: np.ndarray,
    labels: np.ndarray,
    n_layers: int,
    n_heads: int,
    **probe_kwargs,
) -> tuple:
    """
    Train one probe per (layer, head).

    Args:
        mha_activations: shape (n_layers, n_heads, n_examples, head_dim)
        labels: shape (n_examples,)

    Returns:
        accuracy_dict: {(layer, head): accuracy}
        ci_dict:       {(layer, head): (ci_lower, ci_upper)}
        state_dict:    {(layer, head): {"model_state": ..., "proj_std": float}}
        auc_dict:      {(layer, head): auc_roc}
        auc_ci_dict:   {(layer, head): (auc_ci_lower, auc_ci_upper)}
    """
    accuracy_dict = {}
    ci_dict = {}
    state_dict = {}
    auc_dict = {}
    auc_ci_dict = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            X = mha_activations[layer, head]   # (n_examples, head_dim)
            result = train_probe(X, labels, **probe_kwargs)
            accuracy_dict[(layer, head)] = result["accuracy"]
            ci_dict[(layer, head)] = (result["ci_lower"], result["ci_upper"])
            auc_dict[(layer, head)] = result["auc_roc"]
            auc_ci_dict[(layer, head)] = (result["auc_ci_lower"], result["auc_ci_upper"])

            # Projection std: std of activations projected onto probe direction
            w = result["model_state"]["linear.weight"][0].numpy()
            direction = w / (np.linalg.norm(w) + 1e-8)
            proj_std = float(np.std(X @ direction))

            state_dict[(layer, head)] = {
                "model_state": result["model_state"],
                "proj_std": proj_std,
                "input_dim": result["input_dim"],
                "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"],
            }
            print(f"  MHA layer={layer} head={head}: acc={result['accuracy']:.3f} "
                  f"CI=[{result['ci_lower']:.3f},{result['ci_upper']:.3f}] "
                  f"AUC={result['auc_roc']:.3f}")
    return accuracy_dict, ci_dict, state_dict, auc_dict, auc_ci_dict


def train_mlp_probes(
    mlp_activations: np.ndarray,
    labels: np.ndarray,
    n_layers: int,
    **probe_kwargs,
) -> dict:
    """
    Train one probe per MLP layer.

    Args:
        mlp_activations: shape (n_layers, n_examples, hidden_dim)
        labels: shape (n_examples,)

    Returns:
        accuracy_dict: {layer: accuracy}
        ci_dict:       {layer: (ci_lower, ci_upper)}
        state_dict:    {layer: {"model_state": ..., "proj_std": float, "input_dim": int}}
        auc_dict:      {layer: auc_roc}
        auc_ci_dict:   {layer: (auc_ci_lower, auc_ci_upper)}
    """
    accuracy_dict = {}
    ci_dict = {}
    state_dict = {}
    auc_dict = {}
    auc_ci_dict = {}
    for layer in range(n_layers):
        X = mlp_activations[layer]   # (n_examples, hidden_dim)
        result = train_probe(X, labels, **probe_kwargs)
        accuracy_dict[layer] = result["accuracy"]
        ci_dict[layer] = (result["ci_lower"], result["ci_upper"])
        auc_dict[layer] = result["auc_roc"]
        auc_ci_dict[layer] = (result["auc_ci_lower"], result["auc_ci_upper"])

        # Projection std: std of activations projected onto probe direction
        # (also used to scale this direction as a steering vector).
        w = result["model_state"]["linear.weight"][0].numpy()
        direction = w / (np.linalg.norm(w) + 1e-8)
        proj_std = float(np.std(X @ direction))

        state_dict[layer] = {
            "model_state": result["model_state"],
            "proj_std": proj_std,
            "input_dim": result["input_dim"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
        }
        print(f"  MLP layer={layer}: acc={result['accuracy']:.3f} "
              f"CI=[{result['ci_lower']:.3f},{result['ci_upper']:.3f}] "
              f"AUC={result['auc_roc']:.3f}")
    return accuracy_dict, ci_dict, state_dict, auc_dict, auc_ci_dict


def train_residual_probes(
    residual_activations: np.ndarray,
    labels: np.ndarray,
    n_layers: int,
    **probe_kwargs,
) -> tuple:
    """
    Train one probe per residual stream layer.

    Args:
        residual_activations: shape (n_layers, n_examples, hidden_dim)
        labels: shape (n_examples,)

    Returns:
        accuracy_dict: {layer: accuracy}
        ci_dict:       {layer: (ci_lower, ci_upper)}
        state_dict:    {layer: {"model_state": ..., "proj_std": float, "input_dim": int}}
        auc_dict:      {layer: auc_roc}
        auc_ci_dict:   {layer: (auc_ci_lower, auc_ci_upper)}
    """
    accuracy_dict = {}
    ci_dict = {}
    state_dict = {}
    auc_dict = {}
    auc_ci_dict = {}
    for layer in range(n_layers):
        X = residual_activations[layer]   # (n_examples, hidden_dim)
        result = train_probe(X, labels, **probe_kwargs)
        accuracy_dict[layer] = result["accuracy"]
        ci_dict[layer] = (result["ci_lower"], result["ci_upper"])
        auc_dict[layer] = result["auc_roc"]
        auc_ci_dict[layer] = (result["auc_ci_lower"], result["auc_ci_upper"])

        w = result["model_state"]["linear.weight"][0].numpy()
        direction = w / (np.linalg.norm(w) + 1e-8)
        proj_std = float(np.std(X @ direction))

        state_dict[layer] = {
            "model_state": result["model_state"],
            "proj_std": proj_std,
            "input_dim": result["input_dim"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
        }
        print(f"  Residual layer={layer}: acc={result['accuracy']:.3f} "
              f"CI=[{result['ci_lower']:.3f},{result['ci_upper']:.3f}] "
              f"AUC={result['auc_roc']:.3f}")
    return accuracy_dict, ci_dict, state_dict, auc_dict, auc_ci_dict


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def _pool(seq: torch.Tensor, pos: int, pooling: str) -> torch.Tensor:
    """
    seq: (seq_len, dim) activations for one example. pos: index of the last
    answer_token_id occurrence, or -1 if not found.

    "last": the single activation at pos (original behavior).
    "mean": mean over the response span -- everything after pos (the
    generated response's own tokens), or the whole sequence if pos is -1
    (no delimiter found to anchor a response-only span).
    """
    if pooling == "last":
        return seq[pos]
    if pooling == "mean":
        start = pos + 1 if pos != -1 else 0
        return seq[start:].mean(dim=0)
    raise ValueError(f"pooling must be 'last' or 'mean', got {pooling!r}")


def _pool_indices(seq: torch.Tensor, indices: list) -> torch.Tensor:
    """Mean over an arbitrary set of token indices of seq (seq_len, dim) --
    for pooling a sub-span (e.g. one sentence) rather than _pool's
    contiguous "position to end" span."""
    idx = torch.tensor(indices, dtype=torch.long)
    return seq[idx].mean(dim=0)


def collect_activations(
    model,
    tokenizer,
    texts: list,
    model_config: dict,
    batch_size: int = 1,
    pooling: str = "mean",
) -> dict:
    """
    Run forward passes with hooks to collect MHA, MLP, and residual activations.

    pooling: "mean" (default) averages over the response token span --
    everything after the answer_token_id delimiter -- for a less
    position-sensitive signal than a single token. "last" instead takes only
    the activation at that one delimiter position (the original behavior).

    Returns dict with keys "mha", "mlp", "residual":
      - "mha":      (n_layers, n_heads, n_examples, head_dim)
      - "mlp":      (n_layers, n_examples, mlp_dim)  [down_proj output dim = hidden_dim]
      - "residual": (n_layers, n_examples, hidden_dim)
    """
    from sycophancy_model_registry import register_hooks, remove_hooks

    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]
    head_dim = model_config["head_dim"]
    hidden_dim = model_config["hidden_dim"]
    answer_token_id = model_config.get("answer_token_id")

    all_mha = []    # list of (n_layers, n_heads, head_dim) per example
    all_mlp = []    # list of (n_layers, hidden_dim) per example
    all_res = []    # list of (n_layers, hidden_dim) per example

    handles, activation_store = register_hooks(model, model_config)

    model.eval()
    with torch.no_grad():
        for i, text in enumerate(texts):
            activation_store["mha"].clear()
            activation_store["mlp"].clear()

            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024,
            )
            input_ids = inputs["input_ids"]
            device = next(model.parameters()).device
            if str(device) != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)

            # Find answer token position (last occurrence of answer_token_id)
            if answer_token_id is not None:
                token_list = input_ids[0].tolist()
                positions = [j for j, t in enumerate(token_list) if t == answer_token_id]
                pos = positions[-1] if positions else -1
            else:
                pos = -1   # fallback: use last token ("last") / whole sequence ("mean")

            # MHA: (n_layers, n_heads * head_dim) → reshape to (n_layers, n_heads, head_dim)
            mha_example = np.zeros((n_layers, n_heads, head_dim), dtype=np.float32)
            for layer_idx, act in activation_store["mha"].items():
                # act shape: (1, seq_len, n_heads * head_dim)
                vec = _pool(act[0], pos, pooling).float().numpy().astype(np.float32)
                mha_example[layer_idx] = vec.reshape(n_heads, head_dim)
            all_mha.append(mha_example)

            # MLP: (n_layers, hidden_dim)
            mlp_example = np.zeros((n_layers, hidden_dim), dtype=np.float32)
            for layer_idx, act in activation_store["mlp"].items():
                # act shape: (1, seq_len, hidden_dim)
                mlp_example[layer_idx] = _pool(act[0], pos, pooling).float().numpy().astype(np.float32)
            all_mlp.append(mlp_example)

            # Residual: hidden_states is tuple of (1, seq_len, hidden_dim) per layer
            res_example = np.zeros((n_layers, hidden_dim), dtype=np.float32)
            hidden_states = outputs.hidden_states  # tuple length = n_layers + 1
            for layer_idx in range(n_layers):
                hs = _pool(hidden_states[layer_idx + 1][0], pos, pooling).cpu().float().numpy().astype(np.float32)
                res_example[layer_idx] = hs
            all_res.append(res_example)

            if (i + 1) % 10 == 0:
                print(f"  Extracted {i+1}/{len(texts)} examples")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    remove_hooks(handles)

    # Stack: (n_examples, n_layers, ...) → transpose to (n_layers, ...)
    mha_arr = np.stack(all_mha, axis=0)          # (n_examples, n_layers, n_heads, head_dim)
    mha_arr = mha_arr.transpose(1, 2, 0, 3)      # (n_layers, n_heads, n_examples, head_dim)

    mlp_arr = np.stack(all_mlp, axis=0)          # (n_examples, n_layers, hidden_dim)
    mlp_arr = mlp_arr.transpose(1, 0, 2)         # (n_layers, n_examples, hidden_dim)

    res_arr = np.stack(all_res, axis=0)          # (n_examples, n_layers, hidden_dim)
    res_arr = res_arr.transpose(1, 0, 2)         # (n_layers, n_examples, hidden_dim)

    return {"mha": mha_arr, "mlp": mlp_arr, "residual": res_arr}


def split_sentences(text: str) -> list:
    """Lightweight regex sentence splitter -- no nltk/spacy dependency.
    Splits on whitespace following a sentence-ending punctuation mark;
    keeps a trailing fragment even without terminal punctuation. Returns
    list of (sentence_text, start_char, end_char), offsets into `text`."""
    spans = []
    start = 0
    for m in re.finditer(r"(?<=[.!?])\s+", text):
        end = m.start()
        sent = text[start:end]
        if sent.strip():
            spans.append((sent, start, end))
        start = m.end()
    tail = text[start:]
    if tail.strip():
        spans.append((tail, start, len(text)))
    return spans


def collect_sentence_activations(
    model,
    tokenizer,
    records: list,
    model_config: dict,
    max_length: int = 1024,
) -> tuple:
    """
    Like collect_activations, but pools per-sentence within each record's
    response text instead of one vector per whole record -- one forward
    pass per record (preserves the true generation context the response
    was produced under), multiple pooled vectors extracted from that single
    pass's cached activations.

    records: list of dicts with "prompt" and "response" keys (any other
    keys, e.g. a label, are carried through unchanged).

    Sentences that fall (even partially) outside the max_length truncation
    window have no surviving tokens and are dropped; the dropped count is
    printed.

    Returns (sentence_records, activations):
      - sentence_records: list of dicts, one per surviving sentence, each
        the parent record's fields plus "sentence_text" and "parent_index"
        (index into `records`) -- record order preserved, sentences within
        a record in reading order.
      - activations: same {"mha", "mlp", "residual"} shape contract as
        collect_activations's return, except n_examples = len(sentence_records),
        and dtype is float16 (not float32) -- see memory note below.

    Memory: at full dataset scale, sentence-splitting can multiply the
    example count ~20x over collect_activations's one-vector-per-response.
    Growing Python lists of per-sentence float32 vectors then np.stack-ing
    them (collect_activations's pattern) OOM-killed a 52GB-RAM Colab VM
    partway through a ~2,100-response / ~40,000-sentence run. To avoid that,
    this function pre-allocates the three output arrays ONCE, sized from a
    cheap pure-Python pre-pass over split_sentences (no forward pass needed
    to upper-bound the count -- the exact count is only smaller if some
    sentences get truncated away), and writes directly into them -- no
    intermediate list, no doubling at stack time. float16 (not float32)
    roughly halves the footprint on top of that; downstream training
    upcasts to float32 per (layer, [head]) slice via torch.FloatTensor(X),
    which is a tiny fraction of the full array, so precision loss here is
    a non-issue for the linear probes trained on these pooled means.
    """
    from sycophancy_model_registry import register_hooks, remove_hooks
    from utils.inference import build_chat_prompt

    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]
    head_dim = model_config["head_dim"]
    hidden_dim = model_config["hidden_dim"]

    presplit = [split_sentences(rec["response"]) for rec in records]
    max_total_sentences = sum(len(s) for s in presplit)

    mha_arr = np.zeros((max_total_sentences, n_layers, n_heads, head_dim), dtype=np.float16)
    mlp_arr = np.zeros((max_total_sentences, n_layers, hidden_dim), dtype=np.float16)
    res_arr = np.zeros((max_total_sentences, n_layers, hidden_dim), dtype=np.float16)
    sentence_records = []
    n_dropped = 0
    write_idx = 0

    handles, activation_store = register_hooks(model, model_config)
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for rec_idx, rec in enumerate(records):
            sentences = presplit[rec_idx]
            if not sentences:
                continue

            chat_prefix = build_chat_prompt(tokenizer, rec["prompt"], system_prompt=None)
            response = rec["response"]
            prefix_char_len = len(chat_prefix)
            full_text = chat_prefix + response

            activation_store["mha"].clear()
            activation_store["mlp"].clear()

            enc = tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=max_length,
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            inputs = dict(enc)
            if str(device) != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)
            # Move every layer's hidden state to CPU ONCE per record here --
            # mirrors what register_hooks' mha_pre_hook/mlp_hook already do
            # (.detach().cpu() at capture time). Without this, the sentence
            # loop below would call .cpu() on a GPU tensor once per
            # (sentence, layer) pair -- at full-dataset scale that's tens of
            # thousands of individual GPU sync round-trips instead of one
            # batched transfer per record.
            hidden_states_cpu = [hs[0].detach().cpu() for hs in outputs.hidden_states[1:]]

            for sent_text, sent_start, sent_end in sentences:
                full_start = prefix_char_len + sent_start
                full_end = prefix_char_len + sent_end
                token_idx = [
                    j for j, (tok_start, tok_end) in enumerate(offsets)
                    if tok_end > tok_start and full_start <= tok_start < full_end
                ]
                if not token_idx:
                    n_dropped += 1
                    continue

                for layer_idx, act in activation_store["mha"].items():
                    vec = _pool_indices(act[0], token_idx).float().numpy()
                    mha_arr[write_idx, layer_idx] = vec.reshape(n_heads, head_dim).astype(np.float16)

                for layer_idx, act in activation_store["mlp"].items():
                    mlp_arr[write_idx, layer_idx] = _pool_indices(act[0], token_idx).float().numpy().astype(np.float16)

                for layer_idx in range(n_layers):
                    hs = _pool_indices(hidden_states_cpu[layer_idx], token_idx).float().numpy()
                    res_arr[write_idx, layer_idx] = hs.astype(np.float16)

                sentence_records.append({**rec, "sentence_text": sent_text, "parent_index": rec_idx})
                write_idx += 1

            if (rec_idx + 1) % 20 == 0:
                print(
                    f"  Extracted sentences for {rec_idx+1}/{len(records)} records "
                    f"({write_idx} sentences so far, {n_dropped} dropped)"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    remove_hooks(handles)
    print(
        f"Sentence extraction done: {write_idx} sentences from {len(records)} "
        f"records ({n_dropped} dropped -- truncated past max_length={max_length})."
    )

    # Trim unused pre-allocated rows (from dropped sentences) -- a view, not a
    # copy, since it's a slice on the leading axis -- then transpose to match
    # collect_activations's (n_layers, [n_heads,] n_examples, dim) contract.
    mha_arr = mha_arr[:write_idx].transpose(1, 2, 0, 3)  # (n_layers, n_heads, n_sentences, head_dim)
    mlp_arr = mlp_arr[:write_idx].transpose(1, 0, 2)     # (n_layers, n_sentences, hidden_dim)
    res_arr = res_arr[:write_idx].transpose(1, 0, 2)     # (n_layers, n_sentences, hidden_dim)

    return sentence_records, {"mha": mha_arr, "mlp": mlp_arr, "residual": res_arr}


def extract_and_train_all(
    model,
    tokenizer,
    texts: list,
    labels: np.ndarray,
    model_config: dict,
    batch_size: int = 1,
) -> dict:
    """
    Extract activations and train all three probe types.

    Returns dict with keys "mha_accuracy", "mlp_accuracy", "residual_accuracy"
    each mapping to their respective accuracy dicts.
    """
    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]

    print("Extracting activations...")
    activations = collect_activations(model, tokenizer, texts, model_config, batch_size)

    print(f"\nTraining MHA probes ({n_layers} layers × {n_heads} heads)...")
    mha_acc, mha_ci, mha_states, mha_auc, mha_auc_ci = train_mha_probes(activations["mha"], labels, n_layers, n_heads)

    print(f"\nTraining MLP probes ({n_layers} layers)...")
    mlp_acc, mlp_ci, mlp_states, mlp_auc, mlp_auc_ci = train_mlp_probes(activations["mlp"], labels, n_layers)

    print(f"\nTraining residual probes ({n_layers} layers)...")
    res_acc, res_ci, res_states, res_auc, res_auc_ci = train_residual_probes(activations["residual"], labels, n_layers)

    return {
        "mha_accuracy": mha_acc,
        "mha_ci": mha_ci,
        "mha_states": mha_states,
        "mha_auc": mha_auc,
        "mha_auc_ci": mha_auc_ci,
        "mlp_accuracy": mlp_acc,
        "mlp_ci": mlp_ci,
        "mlp_states": mlp_states,
        "mlp_auc": mlp_auc,
        "mlp_auc_ci": mlp_auc_ci,
        "residual_accuracy": res_acc,
        "residual_ci": res_ci,
        "residual_states": res_states,
        "residual_auc": res_auc,
        "residual_auc_ci": res_auc_ci,
        "activations": activations,   # kept for attention analysis; caller should del after use
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_probe_results(results: dict, output_dir: str, model_name: str = ""):
    """
    Save accuracy dicts, probe weights, and projection stds.

    Structure written to output_dir/:
        mha_accuracy.pkl              — {(layer, head): accuracy}
        mlp_accuracy.pkl              — {layer: accuracy}
        residual_accuracy.pkl         — {layer: accuracy}
        {component}_ci.pkl            — {key: (ci_lower, ci_upper)}, if present
        {component}_auc_roc.pkl       — {key: auc_roc}, if present (diagnostic only)
        {component}_auc_roc_ci.pkl    — {key: (auc_ci_lower, auc_ci_upper)}, if present
        mha_probe_weights.pth         — single checkpoint {(layer, head): state_dict}
        mha_projection_stds.pt        — single checkpoint {(layer, head): proj_std}
        probe_metadata.json
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "mha_accuracy.pkl", "wb") as f:
        pickle.dump(results["mha_accuracy"], f)
    with open(out / "mlp_accuracy.pkl", "wb") as f:
        pickle.dump(results["mlp_accuracy"], f)
    with open(out / "residual_accuracy.pkl", "wb") as f:
        pickle.dump(results["residual_accuracy"], f)

    # Save CI dicts
    if "mha_ci" in results:
        with open(out / "mha_ci.pkl", "wb") as f:
            pickle.dump(results["mha_ci"], f)
    if "mlp_ci" in results:
        with open(out / "mlp_ci.pkl", "wb") as f:
            pickle.dump(results["mlp_ci"], f)
    if "residual_ci" in results:
        with open(out / "residual_ci.pkl", "wb") as f:
            pickle.dump(results["residual_ci"], f)

    # Save AUC-ROC dicts (diagnostic only -- direction selection stays accuracy-based)
    for component in ("mha", "mlp", "residual"):
        auc = results.get(f"{component}_auc")
        auc_ci = results.get(f"{component}_auc_ci")
        if auc:
            with open(out / f"{component}_auc_roc.pkl", "wb") as f:
                pickle.dump(auc, f)
        if auc_ci:
            with open(out / f"{component}_auc_roc_ci.pkl", "wb") as f:
                pickle.dump(auc_ci, f)

    # Save probe weights and projection stds as single checkpoints, for
    # whichever components have trained states available. This is what
    # sycophancy_steering.load_steering_vectors reads to steer generation.
    for component in ("mha", "mlp", "residual"):
        states = results.get(f"{component}_states")
        if not states:
            continue
        weights_ckpt = {k: v["model_state"] for k, v in states.items()}
        proj_stds_ckpt = {k: v["proj_std"] for k, v in states.items()}
        torch.save(weights_ckpt, out / f"{component}_probe_weights.pth")
        torch.save(proj_stds_ckpt, out / f"{component}_projection_stds.pt")

    mha_best = max(results["mha_accuracy"].values()) if results["mha_accuracy"] else 0.0
    mlp_best = max(results["mlp_accuracy"].values()) if results["mlp_accuracy"] else 0.0
    res_best = max(results["residual_accuracy"].values()) if results["residual_accuracy"] else 0.0

    mha_best_key = max(results["mha_accuracy"], key=results["mha_accuracy"].get) if results["mha_accuracy"] else None
    mlp_best_key = max(results["mlp_accuracy"], key=results["mlp_accuracy"].get) if results["mlp_accuracy"] else None
    res_best_key = max(results["residual_accuracy"], key=results["residual_accuracy"].get) if results["residual_accuracy"] else None

    # Best CI for each component type
    mha_ci = results.get("mha_ci", {})
    mlp_ci = results.get("mlp_ci", {})
    res_ci = results.get("residual_ci", {})

    mha_best_ci = mha_ci.get(mha_best_key, (None, None)) if mha_best_key else (None, None)
    mlp_best_ci = mlp_ci.get(mlp_best_key, (None, None)) if mlp_best_key else (None, None)
    res_best_ci = res_ci.get(res_best_key, (None, None)) if res_best_key else (None, None)

    # AUC-ROC at the accuracy-selected best key (diagnostic value, not a
    # separately-maximized-by-AUC key -- direction selection stays accuracy-based).
    mha_auc = results.get("mha_auc", {})
    mlp_auc = results.get("mlp_auc", {})
    res_auc = results.get("residual_auc", {})
    mha_best_auc = mha_auc.get(mha_best_key) if mha_best_key else None
    mlp_best_auc = mlp_auc.get(mlp_best_key) if mlp_best_key else None
    res_best_auc = res_auc.get(res_best_key) if res_best_key else None

    metadata = {
        "model_name": model_name,
        "mha_best_accuracy": round(mha_best, 4),
        "mha_best_key": list(mha_best_key) if mha_best_key is not None else None,
        "mha_best_ci": [round(mha_best_ci[0], 4), round(mha_best_ci[1], 4)] if mha_best_ci[0] is not None else None,
        "mha_best_auc": round(mha_best_auc, 4) if mha_best_auc is not None else None,
        "mlp_best_accuracy": round(mlp_best, 4),
        "mlp_best_key": mlp_best_key,
        "mlp_best_ci": [round(mlp_best_ci[0], 4), round(mlp_best_ci[1], 4)] if mlp_best_ci[0] is not None else None,
        "mlp_best_auc": round(mlp_best_auc, 4) if mlp_best_auc is not None else None,
        "residual_best_accuracy": round(res_best, 4),
        "residual_best_key": res_best_key,
        "residual_best_ci": [round(res_best_ci[0], 4), round(res_best_ci[1], 4)] if res_best_ci[0] is not None else None,
        "residual_best_auc": round(res_best_auc, 4) if res_best_auc is not None else None,
    }
    with open(out / "probe_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nProbe results saved to {out}/")
    print(f"  MHA best:      {mha_best:.3f} at {mha_best_key}")
    print(f"  MLP best:      {mlp_best:.3f} at layer {mlp_best_key}")
    print(f"  Residual best: {res_best:.3f} at layer {res_best_key}")

    return metadata


def load_probe_results(probe_dir: str) -> dict:
    """Load saved accuracy/CI/AUC-ROC dicts from probe_dir/."""
    probe_path = Path(probe_dir)
    results = {}

    for key, filename in [
        ("mha_accuracy", "mha_accuracy.pkl"),
        ("mlp_accuracy", "mlp_accuracy.pkl"),
        ("residual_accuracy", "residual_accuracy.pkl"),
        ("mha_ci", "mha_ci.pkl"),
        ("mlp_ci", "mlp_ci.pkl"),
        ("residual_ci", "residual_ci.pkl"),
        ("mha_auc", "mha_auc_roc.pkl"),
        ("mlp_auc", "mlp_auc_roc.pkl"),
        ("residual_auc", "residual_auc_roc.pkl"),
        ("mha_auc_ci", "mha_auc_roc_ci.pkl"),
        ("mlp_auc_ci", "mlp_auc_roc_ci.pkl"),
        ("residual_auc_ci", "residual_auc_roc_ci.pkl"),
    ]:
        path = probe_path / filename
        if path.exists():
            with open(path, "rb") as f:
                results[key] = pickle.load(f)
        else:
            results[key] = {}

    metadata_path = probe_path / "probe_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            results["metadata"] = json.load(f)

    return results
