#!/usr/bin/env python3
"""Extract reusable activation caches at an explicit conversational token span."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SYCOPHANCY_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = SYCOPHANCY_DIR.parents[2]
for candidate in (REPO_ROOT, SYCOPHANCY_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from pipeline_scripts.cache import save_activation_cache  # noqa: E402
from pipeline_scripts.datasets import normalize_records  # noqa: E402
from sycophancy_model_registry import get_model_config, register_hooks, remove_hooks  # noqa: E402


def _render_with_spans(tokenizer, messages: list[dict]) -> tuple[str, list[dict]]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    spans = []
    cursor = 0
    for ordinal, message in enumerate(messages):
        content = str(message["content"])
        start = rendered.find(content, cursor)
        if start < 0:
            raise ValueError(
                f"chat template did not preserve content for {message['role']} message {ordinal + 1}"
            )
        end = start + len(content)
        spans.append({"role": message["role"], "start": start, "end": end})
        cursor = end
    return rendered, spans


def _select_turn(spans: list[dict], role: str, turn: str, record: dict) -> dict:
    candidates = [span for span in spans if span["role"] == role]
    if not candidates:
        raise ValueError(f"record {record['id']} has no {role} turns")
    if turn == "final":
        return candidates[-1]
    if turn == "first_flip":
        if role != "assistant":
            raise ValueError("first_flip is valid only with --token-role assistant")
        first_flip = record.get("first_flip_turn")
        index = len(candidates) - 1 if first_flip in (None, 0) else int(first_flip) - 1
    else:
        index = int(turn) - 1
    if index < 0 or index >= len(candidates):
        raise ValueError(
            f"record {record['id']} has {len(candidates)} {role} turns; requested {turn}"
        )
    return candidates[index]


def select_token_indices(
    tokenizer,
    input_ids: torch.Tensor,
    offsets: list[tuple[int, int]],
    spans: list[dict],
    record: dict,
    role: str,
    turn: str,
    reduction: str,
    token_index: int | None,
    answer_token_id: int | None,
) -> list[int]:
    if role == "sequence":
        indices = [i for i, (start, end) in enumerate(offsets) if end > start]
    elif role == "delimiter":
        if answer_token_id is None:
            raise ValueError("model has no validated answer/delimiter token")
        indices = [i for i, token_id in enumerate(input_ids.tolist()) if token_id == answer_token_id]
    else:
        selected = _select_turn(spans, role, turn, record)
        indices = [
            i for i, (start, end) in enumerate(offsets)
            if end > start and start >= selected["start"] and end <= selected["end"]
        ]
    if not indices:
        raise ValueError(f"record {record['id']} selected no tokens (possibly truncated)")
    if reduction == "mean":
        return indices
    if reduction == "last":
        return [indices[-1]]
    if token_index is None:
        raise ValueError("--token-index is required when --token-reduction=index")
    resolved = token_index if token_index >= 0 else len(indices) + token_index
    if resolved < 0 or resolved >= len(indices):
        raise IndexError(
            f"record {record['id']} selected {len(indices)} tokens; index {token_index} is out of bounds"
        )
    return [indices[resolved]]


def _pool(sequence: torch.Tensor, indices: list[int]) -> np.ndarray:
    index = torch.tensor(indices, dtype=torch.long, device=sequence.device)
    return sequence.index_select(0, index).mean(dim=0).detach().cpu().float().numpy()


def extract(args) -> None:
    from utils.model import cleanup, load_model_and_tokenizer
    records = normalize_records(Path(args.input), args.dataset_type, args.label_field)
    if not records:
        raise ValueError("input produced no labeled records")
    components = [item.strip() for item in args.components.split(",") if item.strip()]
    invalid = set(components) - {"residual", "mlp", "mha"}
    if invalid:
        raise ValueError(f"invalid components: {sorted(invalid)}")

    model, tokenizer = load_model_and_tokenizer(args.model, dtype=args.dtype)
    config = get_model_config(args.model)
    n_layers = config["n_layers"]
    output: dict[str, list[np.ndarray]] = {component: [] for component in components}
    handles = []
    activation_store = None
    if "mha" in components or "mlp" in components:
        handles, activation_store = register_hooks(model, config)

    try:
        model.eval()
        for record_index, record in enumerate(records):
            if record.get("messages") is not None:
                rendered, spans = _render_with_spans(tokenizer, record["messages"])
            elif record.get("text") and args.token_role == "sequence":
                rendered, spans = record["text"], []
            else:
                raise ValueError(
                    f"record {record['id']} lacks messages; only --token-role=sequence is possible"
                )
            if activation_store is not None:
                activation_store["mha"].clear()
                activation_store["mlp"].clear()
            encoded = tokenizer(
                rendered, return_tensors="pt", add_special_tokens=False,
                truncation=True, max_length=args.max_length, return_offsets_mapping=True,
            )
            offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")[0].tolist()]
            input_ids_cpu = encoded["input_ids"][0].cpu()
            indices = select_token_indices(
                tokenizer, input_ids_cpu, offsets, spans, record, args.token_role,
                args.turn, args.token_reduction, args.token_index, config.get("answer_token_id"),
            )
            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                result = model(**encoded, output_hidden_states=True)

            if "residual" in components:
                if len(result.hidden_states) != n_layers + 1:
                    raise RuntimeError(
                        f"model returned {len(result.hidden_states)} hidden states; expected {n_layers + 1}"
                    )
                output["residual"].append(np.stack([
                    _pool(result.hidden_states[layer + 1][0], indices) for layer in range(n_layers)
                ]))
            if "mlp" in components:
                if set(activation_store["mlp"]) != set(range(n_layers)):
                    raise RuntimeError("MLP hook validation failed: not every layer produced an activation")
                output["mlp"].append(np.stack([
                    _pool(activation_store["mlp"][layer][0], indices) for layer in range(n_layers)
                ]))
            if "mha" in components:
                if set(activation_store["mha"]) != set(range(n_layers)):
                    raise RuntimeError("MHA hook validation failed: not every layer produced an activation")
                per_layer = []
                for layer in range(n_layers):
                    vector = _pool(activation_store["mha"][layer][0], indices)
                    expected = config["n_heads"] * config["head_dim"]
                    if vector.size != expected:
                        raise RuntimeError(
                            f"MHA layer {layer} width {vector.size} does not match registry width {expected}"
                        )
                    per_layer.append(vector.reshape(config["n_heads"], config["head_dim"]))
                output["mha"].append(np.stack(per_layer))
            if (record_index + 1) % 25 == 0:
                print(f"Extracted {record_index + 1}/{len(records)}")
    finally:
        if handles:
            remove_hooks(handles)
        cleanup(model, tokenizer)

    arrays = {}
    for component, examples in output.items():
        stacked = np.stack(examples, axis=0)
        if component == "mha":
            arrays[component] = stacked.transpose(1, 2, 0, 3)
        else:
            arrays[component] = stacked.transpose(1, 0, 2)
    metadata = {
        "format_version": 1, "model": args.model, "dataset_type": args.dataset_type,
        "components": components, "n_examples": len(records), "max_length": args.max_length,
        "token_selector": {
            "role": args.token_role, "turn": args.turn,
            "reduction": args.token_reduction, "index": args.token_index,
        },
        "model_config": {key: config[key] for key in (
            "n_layers", "n_heads", "hidden_dim", "head_dim", "mha_hook", "mlp_hook"
        )},
    }
    save_activation_cache(Path(args.output_dir), metadata, records, arrays)
    print(f"Wrote activation cache to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset-type", required=True)
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--components", default="residual")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--token-role", choices=["sequence", "user", "assistant", "delimiter"], default="assistant")
    parser.add_argument("--turn", default="final", help="1-based role turn, final, or first_flip")
    parser.add_argument("--token-reduction", choices=["mean", "last", "index"], default="mean")
    parser.add_argument("--token-index", type=int)
    return parser


def main() -> None:
    extract(build_parser().parse_args())


if __name__ == "__main__":
    main()
