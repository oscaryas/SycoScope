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

Prompt resolution per record: --prompt-field if given (with
--prompt-templated if it is already a rendered chat prefix, as in the baseline
checkpoint.jsonl files), else chat_messages, else chat_prefix, else a
multi-turn `messages` list ending in an assistant turn (the generate_*.py
output schema: messages[:-1], system prompt included, is the prompt and the
final assistant turn is the response span), else user_prompt + system_prompt
through the chat template.

Labels (--label-field) are validated before the model is loaded: every row
must carry a 0/1 label, or the run stops with the offending example ids.
--drop-unlabeled skips rows whose label is missing/None instead (logged in
the .meta.json skips).

Over-length records are skipped, never truncated. Right padding is asserted
on every batch. Output format: see probes_core.save_cache.

Usage:
    python -m analyze_probes.cache_activations \\
        --input data/baseline/<model>/sypr/checkpoint.jsonl \\
        --prompt-field prompt --prompt-templated --group-field utterance_text \\
        --model <model> --token-position response --layers 8 16 24 --output cache.npz
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from analyze_probes import probes_core as core  # noqa: E402
from utils import common  # noqa: E402


def parse_label(value) -> int | None:
    """0/1 int from a bool/int/float/numeric-string label; None if missing or
    not binary."""
    if value is None or isinstance(value, (list, dict)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number in (0.0, 1.0) else None


def attach_labels(records: list[dict], label_field: str, drop_unlabeled: bool) -> tuple[list[dict], list[dict]]:
    """Validate/convert every label BEFORE any GPU work. Returns (kept, skip_log);
    raises SystemExit naming the bad rows unless --drop-unlabeled covers them."""
    kept, skip_log, missing, invalid = [], [], [], []
    for i, rec in enumerate(records):
        rec = dict(rec)
        rec.setdefault("example_id", str(i))
        raw = rec.get(label_field)
        label = parse_label(raw)
        if label is not None:
            rec["_label"] = label
            kept.append(rec)
        elif raw is None:
            missing.append(rec["example_id"])
            skip_log.append({"example_id": rec["example_id"], "reason": "unlabeled"})
        else:
            invalid.append((rec["example_id"], raw))
    if invalid:
        raise SystemExit(f"{len(invalid)} rows have a non-binary {label_field!r} (first: {invalid[:5]}); "
                         f"pick another --label-field")
    if missing and not drop_unlabeled:
        raise SystemExit(f"{len(missing)}/{len(records)} rows have no {label_field!r} (missing key or None; "
                         f"first ids: {missing[:5]}). Judge them first, pass the right --label-field, "
                         f"or --drop-unlabeled to skip them.")
    return kept, skip_log


def normalize_records(records: list[dict], args, tokenizer) -> list[dict]:
    """Fill the fields prepare_records reads (example_id, chat_prefix /
    chat_messages + response) from the CLI's field choices and the record's
    own schema."""
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
            rec.pop("chat_messages", None)  # --prompt-field wins over any chat_messages
        elif (rec.get("chat_messages") is None and not rec.get("chat_prefix") and "user_prompt" not in rec
              and rec.get("messages")):
            messages = rec["messages"]
            if messages[-1].get("role") != "assistant":
                raise SystemExit(f"record {rec['example_id']}: `messages` does not end with an assistant turn")
            rec["chat_messages"] = messages[:-1]
            rec["response"] = messages[-1]["content"]
        out.append(rec)
    return out


def record_group(rec: dict, group_field: str) -> str:
    for field in (group_field, "prompt_id", "row_id", "example_id"):
        if rec.get(field) is not None:
            return str(rec[field])
    raise KeyError(f"record has none of {group_field!r}/prompt_id/row_id/example_id")


def extract_records(records: list[dict], model_name: str, layers=None, layer_fracs=None,
                    positions=("response",), average: str = "mean", batch_size: int = 8,
                    max_length: int = 4096, model_path=None, normalize=None):
    """Load the model, compute spans (over-length rows skipped, never
    truncated), run one batched right-padded forward pass over `records`, and
    free the model. Records need example_id, response and one of
    chat_messages / chat_prefix / user_prompt+system_prompt; `normalize`
    (records, tokenizer) -> records may fill those first. Shared with the
    scorer/eval_*.py OOD evaluations.

    Returns (prepared, skip_log, arrays, layers): arrays is
    {(position, layer): (n_prepared, hidden_dim) float32} in `prepared` order."""
    from types import SimpleNamespace

    from utils.model import cleanup as cleanup_model, load_model_and_tokenizer

    print(f"Loading {model_name} ...")
    model, tokenizer = load_model_and_tokenizer(model_path or model_name)
    # load_model_and_tokenizer sets padding_side="left" for generation. Absolute
    # token indices require RIGHT padding, or every span points into pad tokens.
    tokenizer.padding_side = "right"

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    layers = core.resolve_layers(n_layers, layers, layer_fracs)
    for layer in layers:
        print(f"  block {layer:>2} (depth {layer / n_layers:.2f}) -> hidden_states[{layer + 1}]")

    if normalize is not None:
        records = normalize(records, tokenizer)
    prepared, skip_log = core.prepare_records(records, tokenizer, SimpleNamespace(max_length=max_length))
    print(f"{len(prepared)} examples ({len(skip_log)} skipped)")
    if not prepared:
        cleanup_model(model, tokenizer)
        raise SystemExit("nothing to extract")

    arrays = core.extract(model, tokenizer, prepared, layers, hidden_dim, batch_size,
                          positions=tuple(positions), average=average)
    cleanup_model(model, tokenizer)
    return prepared, skip_log, arrays, layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Generation .jsonl (one record per response).")
    parser.add_argument("--model", required=True)
    parser.add_argument("--token-position", choices=list(common.POSITIONS), default="response")
    parser.add_argument("--average", choices=["mean", "none"], default="mean")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="0-based block indices (read at hidden_states[layer + 1]).")
    parser.add_argument("--layer-fracs", type=float, nargs="+", default=None, help="Depth fractions; default 0.25 0.5 0.75 when --layers is omitted.")
    parser.add_argument("--label-field", default="label", help="Binary 0/1 label field (validated before the model loads).")
    parser.add_argument("--drop-unlabeled", action="store_true", help="Skip rows whose label is missing/None instead of failing.")
    parser.add_argument("--group-field", default="prompt_id", help="CV group field (falls back to prompt_id/row_id/example_id).")
    parser.add_argument("--prompt-field", default=None)
    parser.add_argument("--prompt-templated", action="store_true", help="--prompt-field already holds the rendered chat prefix.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096, help="Skip prompt+response longer than this.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records, label_skips = attach_labels(common.read_jsonl(args.input), args.label_field, args.drop_unlabeled)
    if label_skips:
        print(f"dropping {len(label_skips)} unlabeled rows (--drop-unlabeled)")
    if not records:
        raise SystemExit("no labeled rows")
    prepared, skip_log, arrays, layers = extract_records(
        records, args.model, args.layers, args.layer_fracs, positions=(args.token_position,),
        average=args.average, batch_size=args.batch_size, max_length=args.max_length,
        normalize=lambda recs, tokenizer: normalize_records(recs, args, tokenizer),
    )
    skip_log = label_skips + skip_log

    recs = [p["rec"] for p in prepared]
    core.save_cache(
        args.output,
        {layer: arrays[(args.token_position, layer)] for layer in layers},
        y=np.array([r["_label"] for r in recs], dtype=np.int64),
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
