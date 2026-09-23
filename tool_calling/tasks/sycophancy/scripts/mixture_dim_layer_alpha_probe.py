#!/usr/bin/env python3
"""
Small, fast diagnostic companion to mixture_dim_pipeline.py: pick one early /
one mid / one late residual layer, sweep every alpha across a HANDFUL of
held-out prompts (2 sycophantic + 2 non-sycophantic, by default from
distinct categories), and print each individual generation's judge verdict
-- unlike mixture_dim_pipeline.py's steering sweep, which only reports
aggregate rates over the full held-out set, this is meant to be read
directly: "at this layer and this alpha, did THIS specific prompt's
judged-sycophancy verdict flip?"

Reuses --seed/--n-heldout=20 (mixture_dim_pipeline.py's defaults) to
deterministically reproduce the EXACT same held-out set that produced
results/probing/mixture_dim/dim_classification_results.json and
residual_dim_vectors.pt -- both must already exist (run mixture_dim_
pipeline.py --skip-steering first) since this script only loads them, it
does not retrain DIM.

Usage:
    python mixture_dim_layer_alpha_probe.py \
        --dim-dir results/probing/mixture_dim \
        --layers 0,15,31 \
        --alphas 0.001,0.01,1.0,5.0,10.0,50.0 \
        --n-pos 2 --n-neg 2 \
        --output-dir results/probing/mixture_dim/layer_alpha_probe
"""
import argparse
import json
import sys
from pathlib import Path

import anthropic
import torch

HERE = Path(__file__).resolve().parent
SYCOPHANCY_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
for p in (SYCOPHANCY_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.model import load_model_and_tokenizer, cleanup as cleanup_model
from sycophancy_model_registry import get_model_config
from sycophancy_steering import ActivationSteerer
from mixture_dim_pipeline import (
    load_mixture, load_moral_index, load_social_index, load_sypr_index,
    load_ays_mc_index, load_ays_freeform_index, build_heldout_eval_set,
    generate_responses, judge_items, DEFAULT_ALPHAS,
)


def pick_diverse_items(eval_items: list, n_pos: int, n_neg: int) -> list:
    """n_pos label=1 + n_neg label=0 items, preferring distinct categories
    before repeating one -- so a 4-item probe set spans as many of the 4
    domains as possible instead of landing on the same category twice."""
    picked = []
    for label, n in ((1, n_pos), (0, n_neg)):
        candidates = [it for it in eval_items if it["label"] == label]
        seen_categories, chosen = set(), []
        for it in candidates:
            if it["category"] not in seen_categories:
                chosen.append(it)
                seen_categories.add(it["category"])
            if len(chosen) >= n:
                break
        if len(chosen) < n:  # not enough distinct categories -- fill from remaining candidates
            for it in candidates:
                if it not in chosen:
                    chosen.append(it)
                if len(chosen) >= n:
                    break
        picked.extend(chosen[:n])
    return picked


def response_snippet(item: dict, max_chars: int = 200) -> str:
    text = item.get("response") or item.get("turn2_response") or ""
    return text[:max_chars].replace("\n", " ")


def full_generation(item: dict) -> dict:
    """Full (untruncated) generation text for this item, in whatever shape
    its kind produced -- {"response": ...} for single-turn (moral/social/
    sypr), {"turn1_response": ..., "turn2_response": ...} for two-turn
    (are_you_sure mc/freeform)."""
    if item["kind"] == "single":
        return {"response": item.get("response", "")}
    return {"turn1_response": item.get("turn1_response", ""), "turn2_response": item.get("turn2_response", "")}


def prompt_context(item: dict) -> str:
    """Best-available raw/readable prompt text per category, so a saved
    record is self-contained without cross-referencing the held-out set --
    moral has no separately-stored raw prompt (only the chat-formatted
    one), everything else does."""
    if item["category"] == "social":
        return item.get("orig_prompt", "")
    if item["category"] == "sypr":
        return item.get("utterance_text", "")
    if item["category"] == "are_you_sure":
        return item.get("turn1_question", "")
    return item.get("prompt", "")  # moral: chat-formatted (includes special tokens)


def load_checkpoint(path: Path, probe_items: list) -> dict:
    """Returns {"records": [...], "done_layers": set} if a matching
    checkpoint exists, else None. Validated against THIS run's probe_items
    row_indices -- a checkpoint from a different --n-pos/--n-neg/--seed
    draw would silently mix mismatched records otherwise."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("probe_row_indices") != sorted(it["row_index"] for it in probe_items):
        print(f"  Checkpoint at {path} is for a different probe set -- ignoring, starting fresh.")
        return None
    return {"records": data["records"], "done_layers": set(data["done_layers"])}


def save_checkpoint(path: Path, records: list, done_layers: set, probe_items: list) -> None:
    """Atomic write (temp file + rename)."""
    data = {
        "records": records, "done_layers": sorted(done_layers),
        "probe_row_indices": sorted(it["row_index"] for it in probe_items),
    }
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2))
    tmp_path.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mixture", type=str, default=str(SYCOPHANCY_DIR / "results" / "generations" / "sycophancy_mixture" / "mixture.jsonl"))
    parser.add_argument("--moral-judged", type=str, default=str(SYCOPHANCY_DIR / "results" / "generations" / "AITA-NTA-FLIP_moral_sycophancy_judged.jsonl"))
    parser.add_argument("--social-source", type=str, default=str(SYCOPHANCY_DIR / "results" / "generations" / "social_sycophancy_pooled" / "pooled.jsonl"))
    parser.add_argument("--sypr-source", type=str, default=str(SYCOPHANCY_DIR / "results" / "generations" / "sypr_praise_llama31_full" / "checkpoint.jsonl"))
    parser.add_argument("--dim-dir", type=str, default=str(SYCOPHANCY_DIR / "results" / "probing" / "mixture_dim"), help="Dir containing residual_dim_vectors.pt from a prior mixture_dim_pipeline.py --skip-steering run.")
    parser.add_argument("--n-heldout", type=int, default=20, help="Must match the run that produced --dim-dir (same --seed) to reproduce the identical held-out set.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=str, default="0,15,31")
    parser.add_argument("--alphas", type=str, default=",".join(str(a) for a in DEFAULT_ALPHAS))
    parser.add_argument("--n-pos", type=int, default=2)
    parser.add_argument("--n-neg", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--output-dir", type=str, default=str(SYCOPHANCY_DIR / "results" / "probing" / "mixture_dim" / "layer_alpha_probe"))
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    layers = [int(l) for l in args.layers.split(",")]
    alphas = [0.0] + [float(a) for a in args.alphas.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dim_vectors = torch.load(Path(args.dim_dir) / "residual_dim_vectors.pt", map_location="cpu")

    rows = load_mixture(Path(args.mixture))
    print(f"Loaded {len(rows)} mixture rows.")

    print("Loading held-out prompt-reconstruction indices...")
    moral_idx = load_moral_index(Path(args.moral_judged))
    social_idx = load_social_index(Path(args.social_source))
    sypr_idx = load_sypr_index(Path(args.sypr_source))
    ays_mc_idx = load_ays_mc_index()
    ays_ff_idx = load_ays_freeform_index()

    print(f"\nLoading {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    model_config = get_model_config(args.model)
    model_config["model_name"] = args.model

    print(f"\nReproducing the held-out set (n={args.n_heldout}, seed={args.seed})...")
    _, _, eval_items = build_heldout_eval_set(
        rows, args.n_heldout, args.seed, tokenizer, moral_idx, social_idx, sypr_idx, ays_mc_idx, ays_ff_idx,
    )

    probe_items = pick_diverse_items(eval_items, args.n_pos, args.n_neg)
    print(f"\nProbe set ({len(probe_items)} items):")
    for it in probe_items:
        print(f"  category={it['category']:14s} true_label={it['label']} kind={it['kind']}")

    client = anthropic.Anthropic()
    steerer = ActivationSteerer(model, tokenizer, model_config)

    checkpoint_path = output_dir / "layer_alpha_probe_checkpoint.partial.json"
    checkpoint = load_checkpoint(checkpoint_path, probe_items)
    if checkpoint is not None:
        records, done_layers = checkpoint["records"], checkpoint["done_layers"]
        print(f"  Resumed from checkpoint: {len(done_layers)} layer(s) already done.")
    else:
        records, done_layers = [], set()

    for layer in layers:
        if layer in done_layers:
            print(f"  layer={layer} already complete in checkpoint, skipping")
            continue
        vector = dim_vectors[layer]
        for alpha in alphas:
            items_copy = [dict(it) for it in probe_items]
            if alpha != 0.0:
                steerer.attach("residual", layer, vector, alpha)
            try:
                generate_responses(steerer, items_copy, args.max_new_tokens)
            finally:
                steerer.cleanup()
            verdicts = judge_items(client, items_copy)

            print(f"\n--- layer={layer} alpha={alpha} ---")
            for it in items_copy:
                v = verdicts.get(it["row_index"])
                rec = {
                    "layer": layer, "alpha": alpha, "category": it["category"], "true_label": it["label"],
                    "row_index": it["row_index"], "judged_sycophantic": v, "response_snippet": response_snippet(it),
                    "prompt": prompt_context(it), **full_generation(it),
                }
                records.append(rec)
                print(f"  [{it['category']:14s}] true_label={it['label']} judged_sycophantic={v}  \"{rec['response_snippet']}\"")
        done_layers.add(layer)
        save_checkpoint(checkpoint_path, records, done_layers, probe_items)  # checkpoint every completed layer

    cleanup_model(model, tokenizer)

    with open(output_dir / "layer_alpha_probe_results.json", "w") as f:
        json.dump({
            "layers": layers, "alphas": alphas,
            "probe_items": [{"category": it["category"], "label": it["label"], "kind": it["kind"]} for it in probe_items],
            "records": records,
        }, f, indent=2)
    print(f"\nWrote {output_dir / 'layer_alpha_probe_results.json'}")


if __name__ == "__main__":
    main()
