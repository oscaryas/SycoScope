#!/usr/bin/env python3
"""
Extract residual-stream activations at three token positions from one forward
pass per generated response. Stage 2 of the prompt_probes pipeline.

Positions, all extracted with the system prompt in context:

  last_prompt  h[prompt_len - 1]                        the final prompt token, i.e.
                                                        the position that generated
                                                        response token 0. The two
                                                        classes differ here by
                                                        literally different token
                                                        sequences in context, so this
                                                        is a CEILING/manipulation
                                                        check, not a result.
  first5       mean h[prompt_len : prompt_len + 5]      the INTENT measurement. Five
                                                        response tokens are too sparse
                                                        to carry the label, so the
                                                        probe has to read the
                                                        instruction's effect. This is
                                                        the analogue of Natarajan et
                                                        al.'s truncation before a fact
                                                        resolves.
  response     mean h[prompt_len : resp_end]            behaviour, content permitted.

The first5 -> response gap is therefore the content-contamination measurement.
Because the first5-is-content-free premise is load-bearing, the decoded first
five response tokens are stored in the index so it can be checked directly.

Layer convention is sycophancy_probes': `layer` is a 0-based transformer block
read at hidden_states[layer + 1] (index 0 is the embedding output). Note that
SAE/pipeline/cache_activations.py uses the other convention (raw
hidden_states[layer]) -- do not mix them.

Does not reuse sycophancy_probes._pool / collect_activations: those locate the
response via model_config["answer_token_id"], which is None for
Meta-Llama-3-8B-Instruct, silently degrading to a mean over the whole sequence
including the system prompt.

Right padding is verified on every batch. Left padding would shift every real
token and silently move all three spans, and a mask-sum check cannot catch it --
the sum is identical whichever side the padding sits on.

Usage:
    python get_activations.py --run-name main --layers 8 16 24 --batch-size 8
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402


# ---------------------------------------------------------------------------
# Pure helpers -- no model weights, no heavy imports, so tests/ can exercise
# them with only a tokenizer.
# ---------------------------------------------------------------------------


def response_token_span(offsets, chat_prefix_len: int) -> tuple[int, int]:
    """Token span covering the response portion (chars >= chat_prefix_len).

    Copied from SAE/pipeline/cache_activations.py rather than imported: that module
    imports spacy at top level and is a script, not a library. The duplication is
    kept honest by tests/test_prompt_probes_spans.py.

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


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


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
        chat_prefix = build_chat_prompt(tokenizer, rec["user_prompt"], rec["system_prompt"])
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


def pool_span(hidden: "np.ndarray", start: int, end: int) -> np.ndarray:
    """Mean over [start, end) in float32. Pooling before any downcast matters:
    averaging hundreds of bf16 vectors accumulates real error otherwise."""
    return hidden[start:end].mean(axis=0)


def extract(model, tokenizer, prepared: list[dict], layers: list[int], hidden_dim: int, batch_size: int):
    """Returns {(position, layer): (n, hidden_dim) float32} in `prepared` order."""
    import torch

    n = len(prepared)
    out = {
        (pos, layer): np.zeros((n, hidden_dim), dtype=np.float32)
        for pos in common.POSITIONS
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
                # only guard beyond the padding_side="right" assignment in main().
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
                    for pos, (s, e) in prepared[i]["spans"].items():
                        out[(pos, layer)][i] = pool_span(block[row], s, e)
            print(f"  {min(bstart + batch_size, n)}/{n}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model", type=str, default=common.DEFAULT_MODEL)
    common.add_cells_arg(parser)
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Explicit 0-based block indices.")
    parser.add_argument(
        "--layer-fracs",
        type=float,
        nargs="+",
        default=None,
        help="Depth fractions, resolved as round(frac * n_layers). Default 0.25 0.5 0.75.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096,
                        help="Skip prompt+response longer than this. Raised for uncapped generation.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract cells that already have a .npz.")
    args = parser.parse_args()

    run_dir = common.resolve_run_dir(args.run_name, create=False)
    if not run_dir.exists():
        raise SystemExit(f"no such run: {run_dir}")
    gen_dir = run_dir / "generations"
    act_dir = run_dir / "activations"
    act_dir.mkdir(exist_ok=True)

    requested = args.cells or common.all_slugs()
    available = [s for s in requested if (gen_dir / f"{s}.jsonl").exists()]
    missing = [s for s in requested if s not in available]
    if missing:
        print(f"No generations for: {missing} (skipping)")
    if not available:
        raise SystemExit("nothing to extract")

    from utils.model import cleanup as cleanup_model, load_model_and_tokenizer

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    # load_model_and_tokenizer sets padding_side="left" for generation. Absolute
    # token indices require RIGHT padding, or every span points into pad tokens.
    tokenizer.padding_side = "right"

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    layers = resolve_layers(n_layers, args.layers, args.layer_fracs)
    print(f"\nModel has {n_layers} blocks, hidden {hidden_dim}")
    print("Layer resolution (block index -> hidden_states index):")
    for layer in layers:
        print(f"  block {layer:>2} (depth {layer / n_layers:.2f}) -> hidden_states[{layer + 1}]")

    totals = {"n_examples": 0, "n_skipped": 0}
    for slug in available:
        npz_path = act_dir / f"{slug}.npz"
        if npz_path.exists() and not args.overwrite:
            print(f"\n[{slug}] {npz_path.name} exists, skipping (--overwrite to redo)")
            continue

        records = common.read_jsonl(gen_dir / f"{slug}.jsonl")
        prepared, skip_log = prepare_records(records, tokenizer, args)
        print(f"\n[{slug}] {len(prepared)} examples ({len(skip_log)} skipped)")
        if not prepared:
            continue

        arrays = extract(model, tokenizer, prepared, layers, hidden_dim, args.batch_size)
        np.savez(npz_path, **{act_key(pos, layer): arr for (pos, layer), arr in arrays.items()})

        index_rows = []
        for row, p in enumerate(prepared):
            rec = p["rec"]
            index_rows.append(
                {
                    "row": row,
                    "example_id": rec["example_id"],
                    "label": rec["label"],
                    "slug": rec["slug"],
                    "cell": rec["cell"],
                    "pair_type": rec["pair_type"],
                    "polarity": rec["polarity"],
                    "prompt_id": rec["prompt_id"],
                    "prompt_len": p["prompt_len"],
                    "n_response_tokens": p["resp_end"] - p["prompt_len"],
                    "n_first5_tokens": p["spans"]["first5"][1] - p["spans"]["first5"][0],
                    "first5_text": p["first5_text"],
                    "degenerate": rec["degenerate"],
                }
            )
        common.write_jsonl(act_dir / f"{slug}_index.jsonl", index_rows)

        totals["n_examples"] += len(prepared)
        totals["n_skipped"] += len(skip_log)
        if skip_log:
            (act_dir / f"{slug}_skips.json").write_text(json.dumps(skip_log, indent=2), encoding="utf-8")

    # meta.json written last, as the artifact-complete marker.
    meta = {
        "model": args.model,
        "layer_convention": "hidden_states[layer + 1]",
        "layers": layers,
        "layer_fracs": [round(layer / n_layers, 4) for layer in layers],
        "n_layers": n_layers,
        "hidden_size": hidden_dim,
        "positions": list(common.POSITIONS),
        "dtype": "float32",
        "max_length": args.max_length,
        "cells": available,
        "tokenizer_name_or_path": args.model,
        "code_version": common.get_code_version(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script_args": vars(args),
        **totals,
    }
    (act_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    common.write_run_info(run_dir, "get_activations", args, {"layers": layers, **totals})

    cleanup_model(model, tokenizer)
    print(f"\nDone. {totals['n_examples']} examples, {totals['n_skipped']} skipped. -> {act_dir}")


if __name__ == "__main__":
    main()
