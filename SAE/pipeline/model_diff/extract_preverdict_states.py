"""Extract matched story-end and pre-verdict states from plain AITA prompts.

No SAE is loaded on the GPU host. Frozen SAEs are applied locally after the
base/chat state matrices are copied and prompt-token hashes are checked.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from score_verdict_decisions import FORMAT_ORDERS, prompt_text, records


def contexts(manifest, formats, limit):
    rows = list(records(manifest))
    if limit is not None:
        rows = rows[:limit]
    if len({x["response_id"] for x in rows}) != len(rows):
        raise ValueError("duplicate response_id in manifest")
    return [(row, fmt) for row in rows for fmt in formats]


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    formats = tuple(x.strip() for x in args.formats.split(",") if x.strip())
    layers = tuple(int(x.strip()) for x in args.layers.split(",") if x.strip())
    if not formats or len(formats) != len(set(formats)) or any(x not in FORMAT_ORDERS for x in formats):
        raise ValueError("invalid prompt formats")
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("invalid hidden-state layers")
    todo = contexts(args.manifest, formats, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("fast tokenizer required for story-end offsets")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).eval().cuda()
    if max(layers) >= len(model.model.layers) + 1:
        raise ValueError("hidden-state layer exceeds model")
    shape = (len(todo), len(layers), 2, model.config.hidden_size)
    args.out_states.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.out_states.exists() and args.out_meta.exists():
        states = np.load(args.out_states, mmap_mode="r+")
        if states.shape != shape or states.dtype != np.float16:
            raise ValueError("existing state matrix shape/dtype differs")
        done = {x["state_row"]: x for x in records(args.out_meta)}
        if len(done) != sum(1 for _ in records(args.out_meta)):
            raise ValueError("duplicate rows in existing metadata")
    else:
        states = np.lib.format.open_memmap(args.out_states, mode="w+", dtype=np.float16, shape=shape)
        done = {}
    with args.out_meta.open("a" if done else "w") as meta, torch.inference_mode():
        for i, (row, fmt) in enumerate(todo):
            prompt = prompt_text(row["prompt"], fmt)
            enc = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
            ids, offsets = enc["input_ids"], enc["offset_mapping"]
            if len(ids) > args.max_tokens:
                raise ValueError(f"prompt exceeds max_tokens: {row['response_id']} {fmt} {len(ids)}")
            token_hash = hashlib.sha256(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()
            if i in done:
                old = done[i]
                if (old["response_id"], old["example_order"], old["prompt_ids_sha256"]) != (
                        row["response_id"], fmt, token_hash):
                    raise ValueError(f"resume row mismatch at {i}")
                continue
            story_end_char = len(prompt) - len("\nVerdict:")
            story_tokens = [j for j, (a, b) in enumerate(offsets)
                            if b > a and a < story_end_char and b <= story_end_char]
            if not story_tokens:
                raise ValueError(f"story has no tokens: {row['response_id']}")
            story_end_token = story_tokens[-1]
            crosses_story_end = any(a < story_end_char < b for a, b in offsets if b > a)
            x = torch.tensor([ids], dtype=torch.long, device="cuda")
            output = model(x, output_hidden_states=True)
            for j, layer in enumerate(layers):
                h = output.hidden_states[layer][0]
                states[i, j, 0] = h[story_end_token].float().cpu().numpy().astype(np.float16)
                states[i, j, 1] = h[-1].float().cpu().numpy().astype(np.float16)
            states.flush()
            z = {k: row[k] for k in ("response_id", "pair_id", "stance", "truth")}
            z.update({"model_kind": args.model_kind, "example_order": fmt,
                      "state_row": i, "layers": layers, "positions": ("story_end", "pre_verdict"),
                      "prompt_ids_sha256": token_hash, "prompt_tokens": len(ids),
                      "story_end_token": story_end_token, "pre_verdict_token": len(ids)-1,
                      "crosses_story_end": crosses_story_end})
            meta.write(json.dumps(z) + "\n")
            meta.flush()
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                print(args.model_kind, i+1, len(todo), fmt, row["response_id"], flush=True)
            del x, output
    print("completed", args.model_kind, len(todo), shape, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--tokenizer-path", type=Path, required=True)
    p.add_argument("--model-kind", choices=("base", "chat"), required=True)
    p.add_argument("--out-states", type=Path, required=True)
    p.add_argument("--out-meta", type=Path, required=True)
    p.add_argument("--formats", default="yta_nta,nta_yta,four_yta_last,four_nta_last")
    p.add_argument("--layers", default="14,23")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--resume", action="store_true")
    run(p.parse_args())
