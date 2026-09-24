#!/usr/bin/env python3
"""Cache residual-stream activations at one token position from a single
forward pass per generated response (prompt + response, with any system
prompt in context). Generic across evaluations/generation/ outputs; recipes
that need two forward passes averaged per example (moral, aita_nta_flip,
gemma mixture) have their own cache_train_*.py instead.

Positions:
  last_prompt  h[prompt_len - 1]                 manipulation-check ceiling
  first5       h[prompt_len : prompt_len + 5]    intent measurement
  response     h[prompt_len : resp_end]          behaviour measurement
--average mean pools the span; --average none takes its final token.

Prompt resolution per record: chat_messages, else chat_prefix, else
user_prompt + system_prompt through the chat template. --prompt-field uses
another field instead (with --prompt-templated if it is already a rendered
chat prefix, as in the baseline checkpoint.jsonl files).

Over-length records are skipped, never truncated. Right padding is asserted
on every batch. Output format: see probes_core.save_cache.

Usage:
    python -m probing.analyze_probes.cache_activations \\
        --input probing/data/baseline/<model>/sypr/checkpoint.jsonl \\
        --prompt-field prompt --prompt-templated --group-field utterance_text \\
        --model <model> --token-position response --layers 8 16 24 --output cache.npz
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from probing.analyze_probes import probes_core as core  # noqa: E402
from probing.utils import common  # noqa: E402


def normalize_records(records: list[dict], args, tokenizer) -> list[dict]:
    """Fill the fields prepare_records reads (example_id, chat_prefix) from the
    CLI's field choices, without touching records that already carry them."""
    from utils.inference import build_chat_prompt

    out = []
    for i, rec in enumerate(records):
        rec = dict(rec)
        rec.setdefault("example_id", str(i))
        if args.prompt_field:
            prompt = rec[args.prompt_field]
            rec["chat_prefix"] = prompt if args.prompt_templated else build_chat_prompt(
                tokenizer, prompt, rec.get("system_prompt")
            )
        out.append(rec)
    return out


def record_group(rec: dict, group_field: str) -> str:
    for field in (group_field, "prompt_id", "row_id", "example_id"):
        if rec.get(field) is not None:
            return str(rec[field])
    raise KeyError(f"record has none of {group_field!r}/prompt_id/row_id/example_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Generation .jsonl (one record per response).")
    parser.add_argument("--model", required=True)
    parser.add_argument("--token-position", choices=list(common.POSITIONS), default="response")
    parser.add_argument("--average", choices=["mean", "none"], default="mean")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="0-based block indices (read at hidden_states[layer + 1]).")
    parser.add_argument("--layer-fracs", type=float, nargs="+", default=None, help="Depth fractions; default 0.25 0.5 0.75 when --layers is omitted.")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--group-field", default="prompt_id", help="CV group field (falls back to prompt_id/row_id/example_id).")
    parser.add_argument("--prompt-field", default=None)
    parser.add_argument("--prompt-templated", action="store_true", help="--prompt-field already holds the rendered chat prefix.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096, help="Skip prompt+response longer than this.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from utils.model import cleanup as cleanup_model, load_model_and_tokenizer

    records = common.read_jsonl(args.input)
    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    # load_model_and_tokenizer sets padding_side="left" for generation. Absolute
    # token indices require RIGHT padding, or every span points into pad tokens.
    tokenizer.padding_side = "right"

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    layers = core.resolve_layers(n_layers, args.layers, args.layer_fracs)
    for layer in layers:
        print(f"  block {layer:>2} (depth {layer / n_layers:.2f}) -> hidden_states[{layer + 1}]")

    records = normalize_records(records, args, tokenizer)
    prepared, skip_log = core.prepare_records(records, tokenizer, args)
    print(f"{len(prepared)} examples ({len(skip_log)} skipped)")
    if not prepared:
        raise SystemExit("nothing to extract")

    arrays = core.extract(model, tokenizer, prepared, layers, hidden_dim, args.batch_size,
                          positions=(args.token_position,), average=args.average)
    cleanup_model(model, tokenizer)

    recs = [p["rec"] for p in prepared]
    core.save_cache(
        args.output,
        {layer: arrays[(args.token_position, layer)] for layer in layers},
        y=np.array([int(float(r[args.label_field])) for r in recs]),
        groups=[record_group(r, args.group_field) for r in recs],
        example_id=np.array([str(r["example_id"]) for r in recs]),
        degenerate=np.array([str(r.get("degenerate") or "") for r in recs]),
    )
    meta = {
        "input": str(args.input), "model": args.model, "layers": layers,
        "layer_convention": "hidden_states[layer + 1]", "token_position": args.token_position,
        "average": args.average, "n_examples": len(prepared), "n_skipped": len(skip_log),
        "skips": skip_log, "code_version": common.get_code_version(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.output.with_name(args.output.name + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"cache -> {args.output}")


if __name__ == "__main__":
    main()
