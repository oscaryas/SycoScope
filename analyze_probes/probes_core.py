"""
Shared core for residual-stream probing: activation extraction, the legacy
prompt_probes readers used by the eval_* scripts, and the sklearn L2
logistic-regression C sweep that every training entry point now uses.

Sections:
  * Span extraction (from the retired probe/get_activations.py): token-position
    spans with the system prompt in context, batched forward pass with a
    right-padding guard. Layer convention: `layer` is a 0-based transformer
    block read at hidden_states[layer + 1] (index 0 is the embedding output).
  * Prompt-probes readers (from the retired probe/train_probes.py): the
    per-cell .npz/_index.jsonl readers, fit_probe, safe_auc, paired_win_rate.
  * Whole-text pooled extraction (from the retired probe/baseline_probes.py and
    probe/mixture_residual_probe_pipeline.py): collect_activations and the lean
    residual-only collect_residual_only, both pooling via _pool (mean over the
    span after the model's answer_token_id, or the whole sequence when the
    model has none).
  * Training: l2_sweep_train (grouped-CV C sweep, refit on all data per C),
    pickle_weights, and the .npz activation-cache format (see save_cache).
"""
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.utils import common  # noqa: E402

BALANCE_METHODS = ("undersample", "upweight", "none")


# ---------------------------------------------------------------------------
# Span extraction (pure helpers first -- no model weights, so tests/ can
# exercise them with only a tokenizer)
# ---------------------------------------------------------------------------


def response_token_span(offsets, chat_prefix_len: int) -> tuple[int, int]:
    """Token span covering the response portion (chars >= chat_prefix_len).

    Offsets rather than len(tokenizer(chat_prefix).input_ids) because BPE can merge
    across the prompt/response boundary; the naive length is then off by one, which
    shifts every activation.
    """
    tok_start = None
    tok_end = 0
    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if start >= chat_prefix_len:
            if tok_start is None:
                tok_start = i
            tok_end = i + 1
    if tok_start is None:
        return len(offsets), len(offsets)
    return tok_start, tok_end


def position_spans(prompt_len: int, resp_end: int, n_first: int = 5) -> dict[str, tuple[int, int]]:
    """Half-open [start, end) token spans per position name."""
    return {
        "last_prompt": (prompt_len - 1, prompt_len),
        "first5": (prompt_len, min(prompt_len + n_first, resp_end)),
        "response": (prompt_len, resp_end),
    }


def resolve_layers(n_layers: int, layers=None, fracs=None) -> list[int]:
    """Explicit block indices, or depth fractions resolved against n_layers."""
    if layers:
        out = sorted(dict.fromkeys(int(x) for x in layers))
    elif fracs:
        out = sorted(dict.fromkeys(int(round(float(f) * n_layers)) for f in fracs))
    else:
        out = sorted(dict.fromkeys(int(round(f * n_layers)) for f in common.DEFAULT_LAYER_FRACS))
    for layer in out:
        if not 0 <= layer < n_layers:
            raise ValueError(f"layer {layer} outside [0, {n_layers}) for this model")
    return out


def act_key(position: str, layer: int) -> str:
    return f"{position}_L{layer:02d}"


def load_acts(run_dir: Path, slug: str, position: str, layer: int) -> np.ndarray:
    """(n_rows, hidden_dim) float32 for one (cell, position, layer)."""
    with np.load(run_dir / "activations" / f"{slug}.npz") as z:
        return z[act_key(position, layer)].astype(np.float32)


def load_index(run_dir: Path, slug: str) -> list[dict]:
    """Row i of this list corresponds to row i of every array for that cell."""
    return common.read_jsonl(run_dir / "activations" / f"{slug}_index.jsonl")


def prepare_records(records: list[dict], tokenizer, args) -> tuple[list[dict], list[dict]]:
    """Tokenize (no weights) to compute spans and apply the length policy.

    Returns (prepared, skip_log).

    An over-length prompt+response is skipped and logged, never truncated -- a
    cut prompt makes last_prompt meaningless.
    """
    from utils.inference import build_chat_prompt

    prepared, skip_log = [], []
    for rec in records:
        response = rec.get("response") or ""
        if rec.get("chat_messages") is not None:
            # Multi-turn OOD records contain only the history BEFORE this answer.
            # Render with the same tokenizer as extraction; never append future turns.
            chat_prefix = tokenizer.apply_chat_template(
                rec["chat_messages"], tokenize=False, add_generation_prompt=True
            )
        else:
            chat_prefix = rec.get("chat_prefix") or build_chat_prompt(
                tokenizer, rec["user_prompt"], rec["system_prompt"]
            )
        full_text = chat_prefix + response

        enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        prompt_len, resp_end = response_token_span(enc["offset_mapping"], len(chat_prefix))
        n_tokens_full = len(enc["input_ids"])

        if n_tokens_full > args.max_length:
            skip_log.append({"example_id": rec["example_id"], "reason": "too_long", "n_tokens": n_tokens_full})
            continue
        if resp_end <= prompt_len:
            skip_log.append({"example_id": rec["example_id"], "reason": "empty_response_span"})
            continue

        spans = position_spans(prompt_len, resp_end)
        first5_ids = enc["input_ids"][spans["first5"][0] : spans["first5"][1]]
        prepared.append(
            {
                "rec": rec,
                "full_text": full_text,
                "prompt_len": prompt_len,
                "resp_end": resp_end,
                "n_tokens_full": n_tokens_full,
                "spans": spans,
                "first5_text": tokenizer.decode(first5_ids),
            }
        )
    return prepared, skip_log


def pool_span(hidden: "np.ndarray", start: int, end: int, average: str = "mean") -> np.ndarray:
    """Mean over [start, end) in float32. Pooling before any downcast matters:
    averaging hundreds of bf16 vectors accumulates real error otherwise.

    average="none" takes the span's final token h[end - 1] instead of a mean
    (identical for last_prompt, whose span is one token)."""
    if average == "none":
        return hidden[end - 1]
    return hidden[start:end].mean(axis=0)


def extract(model, tokenizer, prepared: list[dict], layers: list[int], hidden_dim: int, batch_size: int,
            positions=common.POSITIONS, average: str = "mean"):
    """Returns {(position, layer): (n, hidden_dim) float32} in `prepared` order."""
    import torch

    n = len(prepared)
    out = {
        (pos, layer): np.zeros((n, hidden_dim), dtype=np.float32)
        for pos in positions
        for layer in layers
    }
    # Longest first, so padding waste is concentrated in the first batches
    # rather than spread across all of them.
    order = sorted(range(n), key=lambda i: prepared[i]["n_tokens_full"], reverse=True)
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for bstart in range(0, n, batch_size):
            idxs = order[bstart : bstart + batch_size]
            texts = [prepared[i]["full_text"] for i in idxs]
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            mask = enc["attention_mask"]
            lengths = mask.sum(dim=1).tolist()
            for row, (i, length) in enumerate(zip(idxs, lengths)):
                # Right padding makes the unpadded prefix identical to the
                # single-example tokenization, so precomputed spans index straight
                # into the batched hidden states.
                assert length == prepared[i]["n_tokens_full"], (
                    f"{prepared[i]['rec']['example_id']}: batched length {length} != "
                    f"single-example length {prepared[i]['n_tokens_full']}"
                )
                # The length check above cannot catch left padding -- the mask sums
                # the same either way. Check the mask's shape instead. This is the
                # only guard beyond the caller's padding_side="right" assignment.
                assert mask[row, :length].all() and not mask[row, length:].any(), (
                    f"{prepared[i]['rec']['example_id']}: padding is not right-aligned. "
                    "Absolute token indices would read the wrong tokens -- check that "
                    'tokenizer.padding_side = "right" is still set after loading.'
                )
            enc = {k: v.to(device) for k, v in enc.items()}
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states

            for layer in layers:
                # hidden_states[layer + 1] == output of transformer block `layer`.
                block = hs[layer + 1].float().cpu().numpy()
                for row, i in enumerate(idxs):
                    for pos in positions:
                        s, e = prepared[i]["spans"][pos]
                        out[(pos, layer)][i] = pool_span(block[row], s, e, average)
            print(f"  {min(bstart + batch_size, n)}/{n}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Prompt-probes readers + estimator (legacy run-dir layout, read by eval_*)
# ---------------------------------------------------------------------------

# Degeneracy reasons that justify dropping a row (and its pair partner).
# "truncated", "refusal" and "too_short" are deliberately absent: none corrupts
# the activations, and dropping them would condition the sample on the very
# thing being measured. "too_short" especially -- a brusque system prompt is
# supposed to produce short answers, so on ctrl_politeness it fires on 88 of 200
# non_sycophantic rows and would take their sycophantic partners with them,
# leaving the subset where the instruction worked least well. All stay flagged.
DROP_REASONS = frozenset({"empty", "repetitive"})


def fit_probe(X: np.ndarray, y: np.ndarray, seed: int, C: float, max_iter: int, class_weight=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    # L2 is LogisticRegression's default. Passing penalty="l2" explicitly is
    # deprecated in sklearn 1.8 and removed in 1.10; omitting it is verified
    # bit-identical (coef max abs diff 0.0) and keeps the paper's lambda=1
    # as C=1.0.
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed, class_weight=class_weight)
    clf.fit(scaler.transform(X), y)
    return scaler, clf


def score(scaler, clf, X: np.ndarray) -> np.ndarray:
    """Signed distance from the boundary. Continuous scores, not thresholded
    predictions, because AUC and the paired comparison both need a ranking."""
    return clf.decision_function(scaler.transform(X))


def safe_auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    """None when a split has only one class present, where AUC is undefined."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def paired_win_rate(scores: np.ndarray, y: np.ndarray, prompt_ids: list[str]) -> tuple[float, int]:
    """Fraction of prompts whose label-1 response outscores its label-0 response.

    Preferred over accuracy for transfer because it needs neither a threshold
    nor an intercept: a probe from another cell may have a badly miscalibrated
    bias, shifting every score the same way, which destroys accuracy while
    leaving the ranking intact. Ties count as half.
    """
    by: dict[str, dict[int, float]] = {}
    for s, label, pid in zip(scores, y, prompt_ids):
        by.setdefault(pid, {})[int(label)] = float(s)
    wins = [
        1.0 if d[1] > d[0] else (0.5 if d[1] == d[0] else 0.0)
        for d in by.values()
        if 1 in d and 0 in d
    ]
    return (float(np.mean(wins)) if wins else float("nan")), len(wins)


def load_cell(run_dir: Path, slug: str, position: str, layer: int, drop_degenerate: bool):
    """Returns (X, y, prompt_ids, n_dropped) with pairing preserved.

    Dropping a degenerate row also drops its pair partner, so exact 1:1 pairing
    and exact class balance survive filtering.
    """
    index = load_index(run_dir, slug)
    X = load_acts(run_dir, slug, position, layer)
    assert len(index) == X.shape[0], f"{slug}: index has {len(index)} rows, array has {X.shape[0]}"

    keep = np.ones(len(index), dtype=bool)
    if drop_degenerate:
        bad_prompts = {r["prompt_id"] for r in index if r["degenerate"] in DROP_REASONS}
        keep = np.array([r["prompt_id"] not in bad_prompts for r in index], dtype=bool)
    n_dropped = int((~keep).sum())
    if len(index) and not keep.any():
        reasons = sorted({r["degenerate"] for r in index if r["degenerate"]})
        print(
            f"  WARNING [{slug} {position}_L{layer:02d}]: all {len(index)} rows dropped as "
            f"degenerate (reasons: {reasons}). Nothing left to fit."
        )

    rows = [r for r, k in zip(index, keep) if k]
    y = np.array([r["label"] for r in rows], dtype=int)
    return X[keep], y, [r["prompt_id"] for r in rows], n_dropped


# ---------------------------------------------------------------------------
# Whole-text pooled extraction (baseline recipe)
# ---------------------------------------------------------------------------


def _pool(seq, pos: int, pooling: str):
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
    import torch
    from utils.model_registry import register_hooks, remove_hooks

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


def collect_residual_only(model, tokenizer, texts: list, model_config: dict, max_length: int = 1024, pooling: str = "mean") -> np.ndarray:
    """
    Residual-stream activations only -- no MHA/MLP hooks registered, since
    residual comes directly from output_hidden_states. Same pooling
    convention as collect_activations (mean over the response span after
    the answer_token_id delimiter, or whole sequence if not found) so
    results are comparable to the MHA/MLP-inclusive pipelines.

    Returns (n_layers, n_examples, hidden_dim).
    """
    import torch

    n_layers = model_config["n_layers"]
    answer_token_id = model_config.get("answer_token_id")
    device = next(model.parameters()).device

    all_res = []
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"]
            if str(device) != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)

            if answer_token_id is not None:
                token_list = input_ids[0].tolist()
                positions = [j for j, t in enumerate(token_list) if t == answer_token_id]
                pos = positions[-1] if positions else -1
            else:
                pos = -1

            res_example = np.zeros((n_layers, model_config["hidden_dim"]), dtype=np.float32)
            hidden_states = outputs.hidden_states
            for layer_idx in range(n_layers):
                hs = _pool(hidden_states[layer_idx + 1][0], pos, pooling).cpu().float().numpy().astype(np.float32)
                res_example[layer_idx] = hs
            all_res.append(res_example)

            if (i + 1) % 50 == 0:
                print(f"  Extracted {i+1}/{len(texts)} examples")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    res_arr = np.stack(all_res, axis=0)       # (n_examples, n_layers, hidden_dim)
    return res_arr.transpose(1, 0, 2)         # (n_layers, n_examples, hidden_dim)


def load_model_and_config(model_name: str):
    """(model, tokenizer, model_config) as every residual pipeline loaded them."""
    from utils.model import load_model_and_tokenizer
    from utils.model_registry import get_model_config

    model, tokenizer = load_model_and_tokenizer(model_name)
    model_config = get_model_config(model_name)
    model_config["model_name"] = model_name
    return model, tokenizer, model_config


def select_layers(n_layers: int, layers=None) -> list[int]:
    """--layers as given (validated), or every block when omitted."""
    return resolve_layers(n_layers, layers) if layers else list(range(n_layers))


def add_sweep_args(parser, default_model: str) -> None:
    """CLI shared by the cache_train_* scripts."""
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Blocks to train on (default: all).")
    parser.add_argument("--average", choices=["mean", "none"], default="mean",
                        help="mean: average the response span after answer_token_id (whole text if the model "
                             "has none); none: that single delimiter token (the old pooling='last').")
    parser.add_argument("--max-length", type=int, default=1024, help="Per-pass truncation budget.")
    parser.add_argument("--C-values", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--balance-method", choices=BALANCE_METHODS, default="undersample")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-out", type=Path, default=None, help="Also save the pooled activations (.npz).")
    parser.add_argument("--output", type=Path, required=True, help="Pickled weights path.")


def pooling_for(average: str) -> str:
    return "last" if average == "none" else "mean"


def cache_and_train(args, residual: np.ndarray, y: np.ndarray, groups, layers: list[int], meta: dict, **extra) -> dict:
    """Shared tail of the cache_train_* scripts: residual is (n_layers, n, hidden)
    from collect_residual_only; optionally cache the chosen layers, then sweep."""
    acts = {layer: residual[layer] for layer in layers}
    groups = np.asarray([str(g) for g in groups])
    if args.cache_out:
        save_cache(args.cache_out, acts, y, groups, **extra)
        print(f"cache -> {args.cache_out}")
    return train_and_save(
        acts, np.asarray(y).astype(int), groups, layers, args.C_values, args.output, seed=args.seed,
        balance_method=args.balance_method, n_splits=args.n_splits, max_iter=args.max_iter,
        meta=dict(meta, model=args.model, average=args.average, max_length=args.max_length),
    )


# ---------------------------------------------------------------------------
# Statistics
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


def bootstrap_ci(values: np.ndarray, stat_fn=np.mean, n_bootstrap: int = 1000, alpha: float = 0.05) -> tuple:
    """Bootstrap percentile CI for any statistic over a 1-D array."""
    rng = np.random.default_rng(42)
    boots = [stat_fn(rng.choice(values, size=len(values), replace=True)) for _ in range(n_bootstrap)]
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Training: L2 logistic-regression C sweep
# ---------------------------------------------------------------------------

def balance_indices(y: np.ndarray, balance_method: str, seed: int) -> np.ndarray:
    """Row indices to train on. Ported from the retired torch train_probe:
    "undersample" draws min(n_pos, n_neg) of each class with
    np.random.default_rng(seed) and shuffles; "upweight" and "none" keep every
    row ("upweight" is handled by class_weight="balanced" in the estimator,
    the sklearn analogue of the old pos_weight-scaled BCE loss)."""
    if balance_method not in BALANCE_METHODS:
        raise ValueError(f"balance_method must be one of {BALANCE_METHODS}, got {balance_method!r}")
    if balance_method != "undersample":
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    pos_idx = np.nonzero(y == 1)[0]
    neg_idx = np.nonzero(y == 0)[0]
    n_min = min(len(pos_idx), len(neg_idx))
    keep_idx = np.concatenate([
        rng.choice(pos_idx, size=n_min, replace=False),
        rng.choice(neg_idx, size=n_min, replace=False),
    ])
    rng.shuffle(keep_idx)
    return keep_idx


def l2_sweep_train(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C_values: list[float],
    seed: int = 0,
    balance_method: str = "undersample",
    n_splits: int = 5,
    max_iter: int = 2000,
) -> dict:
    """L2 logistic-regression probe per C value, evaluated with GroupKFold CV.

    The scaler is fit on each training fold only (fitting it on all data leaks
    the test distribution into the features); groups keep every row sharing a
    prompt/row_id on one side of each split. After CV the scaler + model are
    refit on ALL (balanced) rows for each C -- cross-validate to evaluate,
    refit to deploy -- and that refit is what gets stored. direction_raw is
    coef / scale (zero-variance features divided by 1, as StandardScaler does),
    i.e. the weight mapped back to raw activation space.

    Returns {C: {"mean_accuracy", "fold_accuracies", "mean_auc", "fold_aucs",
    "model", "scaler", "direction_raw", "intercept", "n_train", "n_pos",
    "n_neg", "n_folds", "keep_idx"}}.
    """
    from sklearn.model_selection import GroupKFold

    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    if not (len(X) == len(y) == len(groups)):
        raise ValueError(f"length mismatch: X={len(X)} y={len(y)} groups={len(groups)}")

    keep_idx = balance_indices(y, balance_method, seed)
    X, y, groups = X[keep_idx], y[keep_idx], groups[keep_idx]
    if len(np.unique(y)) < 2:
        raise ValueError("only one class present after balancing -- nothing to fit")
    class_weight = "balanced" if balance_method == "upweight" else None

    n_folds = min(n_splits, len(np.unique(groups)))
    if n_folds < 2:
        raise ValueError(f"need >= 2 distinct groups for grouped CV, got {n_folds}")
    folds = list(GroupKFold(n_splits=n_folds).split(X, y, groups))

    results = {}
    for C in C_values:
        fold_accuracies, fold_aucs = [], []
        for train_idx, test_idx in folds:
            if len(np.unique(y[train_idx])) < 2:
                continue
            scaler, clf = fit_probe(X[train_idx], y[train_idx], seed, C, max_iter, class_weight)
            s = score(scaler, clf, X[test_idx])
            fold_accuracies.append(float(((s > 0).astype(int) == y[test_idx]).mean()))
            auc = safe_auc(y[test_idx], s)
            if auc is not None:
                fold_aucs.append(auc)

        scaler, clf = fit_probe(X, y, seed, C, max_iter, class_weight)
        coef = clf.coef_[0].astype(np.float64)
        results[float(C)] = {
            "mean_accuracy": float(np.mean(fold_accuracies)) if fold_accuracies else float("nan"),
            "fold_accuracies": fold_accuracies,
            "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else None,
            "fold_aucs": fold_aucs,
            "model": clf,
            "scaler": scaler,
            "direction_raw": coef / np.where(scaler.scale_ == 0, 1.0, scaler.scale_),
            "intercept": float(clf.intercept_[0]),
            "n_train": int(len(y)),
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
            "n_folds": n_folds,
            "keep_idx": keep_idx,
        }
    return results


def best_C(results: dict) -> float:
    return max(results, key=lambda c: results[c]["mean_accuracy"])


def pickle_weights(path: Path, weights: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(weights, f)


def load_weights(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def train_and_save(acts_by_layer: dict, y: np.ndarray, groups: np.ndarray, layers: list[int], C_values: list[float],
                   output: Path, seed: int = 0, balance_method: str = "undersample", n_splits: int = 5,
                   max_iter: int = 2000, meta: Optional[dict] = None) -> dict:
    """Run l2_sweep_train per layer, pickle {"meta", "layers": {layer: {"best_C",
    "by_C": sweep}}} to `output`, and write a JSON accuracy summary next to it
    (<output>.summary.json). Returns the pickled payload."""
    payload = {"meta": dict(meta or {}, C_values=list(C_values), seed=seed, balance_method=balance_method,
                            n_splits=n_splits, max_iter=max_iter), "layers": {}}
    summary = {"meta": payload["meta"], "layers": {}}
    for layer in layers:
        sweep = l2_sweep_train(acts_by_layer[layer], y, groups, C_values, seed, balance_method, n_splits, max_iter)
        c_star = best_C(sweep)
        payload["layers"][int(layer)] = {"best_C": c_star, "by_C": sweep}
        summary["layers"][str(layer)] = {
            "best_C": c_star,
            "by_C": {str(c): {"mean_accuracy": r["mean_accuracy"], "mean_auc": r["mean_auc"],
                              "fold_accuracies": r["fold_accuracies"]} for c, r in sweep.items()},
            "n_train": sweep[c_star]["n_train"],
        }
        auc = sweep[c_star]["mean_auc"]
        print(f"layer {layer:>2}: best C={c_star:g} acc={sweep[c_star]['mean_accuracy']:.3f} "
              f"auc={(auc if auc is not None else float('nan')):.3f} (n={sweep[c_star]['n_train']})")
    output = Path(output)
    pickle_weights(output, payload)
    output.with_name(output.name + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"weights -> {output}")
    return payload


# ---------------------------------------------------------------------------
# Activation cache (.npz)
# ---------------------------------------------------------------------------


def layer_key(layer: int) -> str:
    return f"L{int(layer):02d}"


def save_cache(path: Path, acts_by_layer: dict, y: np.ndarray, groups, **extra) -> None:
    """Cache format: one .npz with a (n, hidden_dim) float32 array per layer
    under key "L{layer:02d}", plus "y" (n,) int labels, "groups" (n,) str CV
    groups, "layers" (the cached layer indices), and any extra (n,)-aligned
    arrays passed as keyword arguments (e.g. example_id, degenerate)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layers = sorted(int(layer) for layer in acts_by_layer)
    arrays = {layer_key(layer): np.asarray(acts_by_layer[layer], dtype=np.float32) for layer in layers}
    arrays.update(
        y=np.asarray(y).astype(int),
        groups=np.asarray([str(g) for g in groups]),
        layers=np.asarray(layers, dtype=int),
        **{k: np.asarray(v) for k, v in extra.items()},
    )
    np.savez(path, **arrays)


def load_cache(path: Path) -> dict:
    """Inverse of save_cache: {"acts": {layer: X}, "y", "groups", "layers", **extra}."""
    with np.load(path, allow_pickle=False) as z:
        layers = [int(x) for x in z["layers"]]
        out = {"acts": {layer: z[layer_key(layer)] for layer in layers}, "layers": layers}
        for k in z.files:
            if k not in out and not (k.startswith("L") and k[1:].isdigit()):
                out[k] = z[k]
    return out
