#!/usr/bin/env python3
"""
RQ1 Gate 1 -- direction-geometry screen, per RQ1_gate1_cosine_ceilings.md.

Question: do the four sycophancy types (moral, social/validation, praise,
truth-caving) share one underlying representation, or are they distinct?
Computed via DIM (difference-in-means) directions at 3 fixed residual layers
(early/mid/late), a 4x4 cosine matrix between them, split-half reliability
ceilings (how similar are two independent estimates of the SAME type's
direction), disattenuated cosines (correcting the raw matrix for each type's
own estimation noise), and bootstrap CIs on every cross-type cosine.

All four types' data (prompt/response/label) is already cached from earlier
work this session -- no fresh generation or judging needed. Only new compute
is the extraction itself: last-prompt-token residual activations (the
prompt's own last token, BEFORE any response -- leakage-free, and doesn't
need a response at all for this position, unlike the mean-pooled position
used elsewhere in this session's probe work). This model
(meta-llama/Meta-Llama-3-8B-Instruct) has no answer_token_id in the registry,
so this script computes the position directly (tokenize the prompt alone,
take its length) rather than relying on collect_activations' answer_token_id
lookup.

Includes Section 7's confound-control rerun: builds sentiment and generic-
agreement control directions the same way, reports cos(v_t, v_control) for
each type, and recomputes the whole matrix with controls projected out.

Does NOT include Gate 1.5 (causal sufficiency/necessity validation via
steering/ablation) -- explicitly scoped as a follow-up per the plan doc,
since it needs generation, not just forward passes.

Usage:
    python rq1_gate1_geometry.py \
        --output-dir results/rq1_gate1_geometry \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --n-moral 500 --n-social 3027 --n-praise-pos 924 --n-truth 150 \
        --n-split-half 100 --n-bootstrap 1000
"""
import argparse
import json
import random
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

from utils.model import load_model_and_tokenizer, cleanup as cleanup_model
from utils.inference import build_chat_prompt
from sycophancy_model_registry import get_model_config
from sycophancy_dim import compute_dim_direction
from sycophancy_data import load_truthfulqa

LAYERS = [5, 13, 27]
TYPES = ["moral", "social", "praise", "truth"]
CONTROLS = ["sentiment", "agreement"]


# ---------------------------------------------------------------------------
# Last-prompt-token residual extraction -- no response needed at all, since
# this position is BEFORE any response begins.
# ---------------------------------------------------------------------------

def extract_last_token_residual(model, tokenizer, prompts: list, layers: list, max_length: int = 1024) -> dict:
    """Returns {layer: (n_examples, hidden_dim) np.ndarray}, last-token residual
    activation at each requested layer, for each of `prompts` (already
    chat-formatted, no response appended)."""
    device = next(model.parameters()).device
    out = {l: [] for l in layers}
    model.eval()
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            if str(device) != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            for l in layers:
                vec = outputs.hidden_states[l + 1][0, -1].cpu().float().numpy()
                out[l].append(vec)
            if (i + 1) % 100 == 0:
                print(f"    extracted {i+1}/{len(prompts)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return {l: np.stack(v, axis=0) for l, v in out.items()}


# ---------------------------------------------------------------------------
# Per-type data loading -- all reuse already-cached prompt/response/label
# data from earlier this session. Only the moral type needs row_id-pooling
# (average the two sides' activations per conflict into one example).
# ---------------------------------------------------------------------------

def load_moral(tokenizer, n_pairs: int, seed: int) -> tuple:
    """Returns (prompts_og, prompts_flip, row_ids, labels) -- row_ids/labels
    are per-PAIR (not per-side); prompts_og/prompts_flip are parallel lists,
    one entry per pair, chat-formatted, NO response appended."""
    path = SYCOPHANCY_DIR / "results" / "generations" / "AITA-NTA-FLIP_moral_sycophancy_judged.jsonl"
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n_pairs]

    row_ids = [r["row_id"] for r in rows]
    labels = np.array([
        1 if r["original_post_verdict"] == "NTA" and r["flipped_story_verdict"] == "NTA" else 0
        for r in rows
    ], dtype=np.float32)
    prompts_og = [build_chat_prompt(tokenizer, r["original_post_prompt"], system_prompt=None) for r in rows]
    prompts_flip = [build_chat_prompt(tokenizer, r["flipped_story_prompt"], system_prompt=None) for r in rows]
    return prompts_og, prompts_flip, row_ids, labels


def load_social(tokenizer, n: int, seed: int) -> tuple:
    """Returns (prompts, labels) from cached OEQ validation labels."""
    path = SYCOPHANCY_DIR / "results" / "generations" / "social_sycophancy_pooled" / "pooled.jsonl"
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    rows = [r for r in rows if r.get("domain") == "OEQ"]
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]
    prompts = [build_chat_prompt(tokenizer, r["prompt"], system_prompt=None) for r in rows]
    labels = np.array([r["validation"] for r in rows], dtype=np.float32)
    return prompts, labels


def load_praise(tokenizer, n_pos: int, seed: int) -> tuple:
    """Returns (prompts, labels) from cached SyPR generations, n_pos positives
    matched with n_pos randomly-sampled negatives (undersampled to balance)."""
    path = SYCOPHANCY_DIR / "results" / "generations" / "sypr_praise_llama31_full" / "checkpoint.jsonl"
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    rng = random.Random(seed)
    n_pos = min(n_pos, len(pos))
    sampled = rng.sample(pos, n_pos) + rng.sample(neg, n_pos)
    rng.shuffle(sampled)
    prompts = [build_chat_prompt(tokenizer, r["prompt"], system_prompt=None) for r in sampled]
    labels = np.array([r["label"] for r in sampled], dtype=np.float32)
    return prompts, labels


def load_truth(tokenizer, n_questions: int, seed: int) -> tuple:
    """Constructed contrastive pairs: question + correct answer (label=1) vs.
    question + first incorrect answer (label=0), no judge needed, 50/50 by
    construction. Prompt = the WHOLE constructed string (question+answer) --
    unlike the other 3 types there's no separate response to exclude here,
    the appended answer IS the thing being contrasted."""
    dataset = load_truthfulqa(n_questions * 2)  # load extra, filter down after
    rng = random.Random(seed)
    items = list(dataset)
    rng.shuffle(items)
    items = items[:n_questions]

    prompts, labels = [], []
    for ex in items:
        question = ex["question"]
        best_answer = ex["best_answer"]
        incorrect = ex["incorrect_answers"][0]
        prompts.append(build_chat_prompt(tokenizer, question, system_prompt=None) + best_answer)
        labels.append(1)
        prompts.append(build_chat_prompt(tokenizer, question, system_prompt=None) + incorrect)
        labels.append(0)
    return prompts, np.array(labels, dtype=np.float32)


def load_controls(tokenizer, n: int, seed: int) -> dict:
    """Small constructed control pairs (Section 7), same layer/position/model
    as the 4 real types: sentiment (positive vs negative statements) and
    generic agreement (agree vs disagree with a neutral factual claim)."""
    rng = random.Random(seed)

    sentiment_pos_templates = [
        "I had such a wonderful day today, everything went perfectly.",
        "This is the best meal I've ever had, truly delicious.",
        "I'm so happy and grateful for my friends and family.",
        "What a beautiful, sunny morning -- I feel great.",
        "This movie was fantastic, I loved every minute of it.",
    ]
    sentiment_neg_templates = [
        "I had such a terrible day today, everything went wrong.",
        "This is the worst meal I've ever had, truly awful.",
        "I'm so upset and disappointed in my friends and family.",
        "What a gloomy, miserable morning -- I feel awful.",
        "This movie was terrible, I hated every minute of it.",
    ]
    agree_claim_templates = [
        "Water boils at 100 degrees Celsius at sea level.",
        "The Earth orbits the Sun once per year.",
        "Paris is the capital of France.",
        "A triangle has three sides.",
        "The human heart pumps blood through the body.",
    ]

    def expand(templates, n_needed):
        out = []
        i = 0
        while len(out) < n_needed:
            out.append(templates[i % len(templates)])
            i += 1
        return out[:n_needed]

    sentiment_prompts = (
        [build_chat_prompt(tokenizer, t, system_prompt=None) for t in expand(sentiment_pos_templates, n // 2)]
        + [build_chat_prompt(tokenizer, t, system_prompt=None) for t in expand(sentiment_neg_templates, n // 2)]
    )
    sentiment_labels = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.float32)

    agree_prompts, agree_labels = [], []
    for t in expand(agree_claim_templates, n // 2):
        agree_prompts.append(build_chat_prompt(tokenizer, f"{t} I agree with this statement.", system_prompt=None))
        agree_labels.append(1)
        agree_prompts.append(build_chat_prompt(tokenizer, f"{t} I disagree with this statement.", system_prompt=None))
        agree_labels.append(0)
    agree_labels = np.array(agree_labels, dtype=np.float32)

    return {
        "sentiment": (sentiment_prompts, sentiment_labels),
        "agreement": (agree_prompts, agree_labels),
    }


# ---------------------------------------------------------------------------
# Geometry: cosines, split-half ceilings, disattenuation, bootstrap
# ---------------------------------------------------------------------------

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def dim_from_indices(X, y, idx):
    Xs, ys = X[idx], y[idx]
    if len(np.unique(ys)) < 2:
        return None
    return compute_dim_direction(Xs, ys, method="naive")["direction"]


def split_half_ceiling_stratified(X, y, n_splits, seed):
    """Standard split-half: stratify by label so each half keeps its share of
    positives. Returns list of cosines across n_splits random splits."""
    rng = np.random.default_rng(seed)
    n = len(y)
    pos_idx = np.nonzero(y == 1)[0]
    neg_idx = np.nonzero(y == 0)[0]
    cosines = []
    for _ in range(n_splits):
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        half_pos = len(pos_idx) // 2
        half_neg = len(neg_idx) // 2
        idx_a = np.concatenate([pos_idx[:half_pos], neg_idx[:half_neg]])
        idx_b = np.concatenate([pos_idx[half_pos:2 * half_pos], neg_idx[half_neg:2 * half_neg]])
        d_a = dim_from_indices(X, y, idx_a)
        d_b = dim_from_indices(X, y, idx_b)
        if d_a is not None and d_b is not None:
            cosines.append(cosine(d_a, d_b))
    return cosines


def split_half_ceiling_moral(X_pooled, y, row_ids, n_splits, seed):
    """Moral-specific: split by row_id (never let one conflict's pooled
    example be duplicated across halves -- trivially true here since each
    row_id already IS one pooled example, but keep the explicit row_id-aware
    split for clarity/consistency with the plan doc's leakage rule)."""
    return split_half_ceiling_stratified(X_pooled, y, n_splits, seed)


def bootstrap_cross_cosine(X_i, y_i, X_j, y_j, n_boot, seed):
    rng = np.random.default_rng(seed)
    n_i, n_j = len(y_i), len(y_j)
    cosines = []
    for _ in range(n_boot):
        idx_i = rng.choice(n_i, size=n_i, replace=True)
        idx_j = rng.choice(n_j, size=n_j, replace=True)
        d_i = dim_from_indices(X_i, y_i, idx_i)
        d_j = dim_from_indices(X_j, y_j, idx_j)
        if d_i is not None and d_j is not None:
            cosines.append(cosine(d_i, d_j))
    return cosines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=str, default=str(SYCOPHANCY_DIR / "results" / "rq1_gate1_geometry"))
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--n-moral", type=int, default=500)
    parser.add_argument("--n-social", type=int, default=3027)
    parser.add_argument("--n-praise-pos", type=int, default=924)
    parser.add_argument("--n-truth", type=int, default=150)
    parser.add_argument("--n-controls", type=int, default=100)
    parser.add_argument("--n-split-half", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    print(f"Layers: {LAYERS}")

    print("\nLoading per-type data (all from cached labels, no fresh generation/judging)...")
    moral_og_prompts, moral_flip_prompts, moral_row_ids, moral_labels = load_moral(tokenizer, args.n_moral, args.seed)
    social_prompts, social_labels = load_social(tokenizer, args.n_social, args.seed)
    praise_prompts, praise_labels = load_praise(tokenizer, args.n_praise_pos, args.seed)
    truth_prompts, truth_labels = load_truth(tokenizer, args.n_truth, args.seed)
    controls = load_controls(tokenizer, args.n_controls, args.seed)

    print(f"  moral: {len(moral_labels)} pairs ({int(moral_labels.sum())} pos / {int((1-moral_labels).sum())} neg)")
    print(f"  social: {len(social_labels)} ({int(social_labels.sum())} pos / {int((1-social_labels).sum())} neg)")
    print(f"  praise: {len(praise_labels)} ({int(praise_labels.sum())} pos / {int((1-praise_labels).sum())} neg)")
    print(f"  truth: {len(truth_labels)} ({int(truth_labels.sum())} pos / {int((1-truth_labels).sum())} neg)")
    for name, (prompts, labels) in controls.items():
        print(f"  control[{name}]: {len(labels)} ({int(labels.sum())} pos / {int((1-labels).sum())} neg)")

    print(f"\nExtracting last-prompt-token residual activations at layers {LAYERS}...")
    print("  [moral, original-post side]")
    moral_og_acts = extract_last_token_residual(model, tokenizer, moral_og_prompts, LAYERS)
    print("  [moral, flipped-story side]")
    moral_flip_acts = extract_last_token_residual(model, tokenizer, moral_flip_prompts, LAYERS)
    moral_acts = {l: (moral_og_acts[l] + moral_flip_acts[l]) / 2.0 for l in LAYERS}  # row_id-pooled: average both sides

    print("  [social]")
    social_acts = extract_last_token_residual(model, tokenizer, social_prompts, LAYERS)
    print("  [praise]")
    praise_acts = extract_last_token_residual(model, tokenizer, praise_prompts, LAYERS)
    print("  [truth]")
    truth_acts = extract_last_token_residual(model, tokenizer, truth_prompts, LAYERS)

    control_acts = {}
    for name, (prompts, labels) in controls.items():
        print(f"  [control: {name}]")
        control_acts[name] = extract_last_token_residual(model, tokenizer, prompts, LAYERS)

    cleanup_model(model, tokenizer)

    activations = {"moral": moral_acts, "social": social_acts, "praise": praise_acts, "truth": truth_acts}
    labels_by_type = {"moral": moral_labels, "social": social_labels, "praise": praise_labels, "truth": truth_labels}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_layer_results = {}
    for layer in LAYERS:
        print(f"\n{'='*70}\nLayer {layer}\n{'='*70}")

        directions = {}
        for t in TYPES:
            X, y = activations[t][layer], labels_by_type[t]
            result = compute_dim_direction(X, y, method="naive")
            directions[t] = result["direction"]
            print(f"  {t}: cohen_d={result['effect_size']:.3f} auc={result['auc_roc']:.3f} n={len(y)}")

        control_directions = {}
        for name, (_, clabels) in controls.items():
            X = control_acts[name][layer]
            result = compute_dim_direction(X, clabels, method="naive")
            control_directions[name] = result["direction"]

        # Save direction vectors for this layer.
        vec_ckpt = {t: torch.from_numpy(directions[t]).float() for t in TYPES}
        vec_ckpt.update({f"control_{name}": torch.from_numpy(d).float() for name, d in control_directions.items()})
        torch.save(vec_ckpt, out_dir / f"directions_L{layer:02d}.pt")

        # Raw cosine matrix.
        C = {i: {j: cosine(directions[i], directions[j]) for j in TYPES} for i in TYPES}
        print("\n  Cosine matrix:")
        for i in TYPES:
            print(f"    {i:8s} " + " ".join(f"{C[i][j]:+.3f}" for j in TYPES))

        # Split-half ceilings.
        print("\n  Split-half ceilings (r_t)...")
        ceilings = {}
        for t in TYPES:
            X, y = activations[t][layer], labels_by_type[t]
            if t == "moral":
                cosines = split_half_ceiling_moral(X, y, moral_row_ids, args.n_split_half, args.seed)
            else:
                cosines = split_half_ceiling_stratified(X, y, args.n_split_half, args.seed)
            ceilings[t] = {
                "mean": float(np.mean(cosines)) if cosines else None,
                "std": float(np.std(cosines)) if cosines else None,
                "n_valid_splits": len(cosines),
            }
            print(f"    r_{t} = {ceilings[t]['mean']:.3f} (std={ceilings[t]['std']:.3f}, n_valid={ceilings[t]['n_valid_splits']})")

        # Disattenuated cosines.
        C_hat = {}
        for i in TYPES:
            C_hat[i] = {}
            for j in TYPES:
                ri, rj = ceilings[i]["mean"], ceilings[j]["mean"]
                denom = np.sqrt(max(ri, 0) * max(rj, 0)) if ri is not None and rj is not None else None
                C_hat[i][j] = C[i][j] / denom if denom and denom > 1e-6 else None

        # Bootstrap CIs on raw cross-type cosines (upper triangle only, i<j).
        print(f"\n  Bootstrapping cross-type cosine CIs ({args.n_bootstrap} resamples each)...")
        bootstrap_ci = {}
        type_pairs = [(TYPES[a], TYPES[b]) for a in range(len(TYPES)) for b in range(a + 1, len(TYPES))]
        for i, j in type_pairs:
            cosines = bootstrap_cross_cosine(
                activations[i][layer], labels_by_type[i], activations[j][layer], labels_by_type[j],
                args.n_bootstrap, args.seed,
            )
            bootstrap_ci[f"{i}_{j}"] = {
                "mean": float(np.mean(cosines)) if cosines else None,
                "ci_lower_2.5": float(np.percentile(cosines, 2.5)) if cosines else None,
                "ci_upper_97.5": float(np.percentile(cosines, 97.5)) if cosines else None,
                "n_valid": len(cosines),
            }
            print(f"    cos({i},{j}) = {C[i][j]:+.3f}, bootstrap CI=[{bootstrap_ci[f'{i}_{j}']['ci_lower_2.5']:.3f}, "
                  f"{bootstrap_ci[f'{i}_{j}']['ci_upper_97.5']:.3f}]")

        # Section 7: control cosines, then project controls out and recompute C.
        print("\n  Confound check: cos(v_t, v_control)...")
        control_cosines = {t: {name: cosine(directions[t], control_directions[name]) for name in CONTROLS} for t in TYPES}
        for t in TYPES:
            print(f"    {t}: " + ", ".join(f"{name}={control_cosines[t][name]:+.3f}" for name in CONTROLS))

        projected_directions = {}
        for t in TYPES:
            v = directions[t].copy()
            for name in CONTROLS:
                c = control_directions[name]
                v = v - np.dot(v, c) * c
            norm = np.linalg.norm(v)
            projected_directions[t] = v / norm if norm > 1e-8 else v
        C_projected = {i: {j: cosine(projected_directions[i], projected_directions[j]) for j in TYPES} for i in TYPES}
        print("\n  Cosine matrix AFTER projecting out controls:")
        for i in TYPES:
            print(f"    {i:8s} " + " ".join(f"{C_projected[i][j]:+.3f}" for j in TYPES))

        per_layer_results[layer] = {
            "cosine_matrix": C,
            "ceilings": ceilings,
            "disattenuated_cosine_matrix": C_hat,
            "bootstrap_ci": bootstrap_ci,
            "control_cosines": control_cosines,
            "cosine_matrix_after_control_projection": C_projected,
            "n_examples": {t: int(len(labels_by_type[t])) for t in TYPES},
        }

    summary = {
        "model": args.model, "layers": LAYERS, "seed": args.seed,
        "n_split_half": args.n_split_half, "n_bootstrap": args.n_bootstrap,
        "sample_sizes": {
            "moral_pairs": args.n_moral, "social": args.n_social,
            "praise_pos": args.n_praise_pos, "truth_questions": args.n_truth,
            "controls": args.n_controls,
        },
        "results_by_layer": per_layer_results,
    }
    with open(out_dir / "cosine_matrix.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {out_dir / 'cosine_matrix.json'}")
    print(f"Wrote per-layer direction checkpoints to {out_dir}/directions_L{{05,13,27}}.pt")


if __name__ == "__main__":
    main()
