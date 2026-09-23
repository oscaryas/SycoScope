#!/usr/bin/env python3
"""Export official base scenarios or generate checkpointed five-turn GPU responses.

Uses the existing source loaders, but local HF inference rather than paid generation.
No activation/probe labels are used. Actual prior responses stay in the history.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tool_calling/tasks/sycophancy"))
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, help="Export 500 official base scenarios; no inference")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-path", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.export:
        from pipeline_scripts.generations.generate_syconbench import (
            DEFAULT_SOURCE, load_debate, load_ethical, load_false_presupposition,
        )
        rows = [r for loader in (load_debate, load_ethical, load_false_presupposition)
                for r in loader(DEFAULT_SOURCE)]
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with args.export.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Exported {len(rows)} scenarios")
        return
    if not args.input or not args.output or args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("generation requires --input, --output, and positive batch/token limits")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    rows = read_rows(args.input)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("limit must be positive")
        rows = rows[:args.limit]
    signature = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"model": MODEL, "model_path": args.model_path, "input_sha256": signature,
                "max_new_tokens": args.max_new_tokens, "do_sample": False,
                "batch_size": args.batch_size, "seed": 0,
                "transformers_version": transformers.__version__, "torch_version": torch.__version__}
    meta_path = args.output.with_suffix(".meta.json")
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("resume configuration changed; use a new output")
    if args.output.exists() and not meta_path.exists():
        raise ValueError("existing output has no provenance")
    meta_path.write_text(json.dumps(metadata, indent=2))
    done_rows = read_rows(args.output) if args.output.exists() else []
    done = {r["id"] for r in done_rows}
    if len(done) != len(done_rows) or not done <= {r["id"] for r in rows}:
        raise ValueError("duplicate or unexpected resume IDs")
    pending = [r for r in rows if r["id"] not in done]
    if not pending:
        print("All conversations already generated")
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    if model.config.max_position_embeddings != 131072 or model.config.num_hidden_layers != 32:
        raise ValueError("checkpoint config is not the expected Llama-3.1-8B architecture")
    model.eval()
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    start = time.monotonic()
    with args.output.open("a", encoding="utf-8") as handle, torch.inference_mode():
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset:offset + args.batch_size]
            conversations = [[{"role": "system", "content": r["system_message"]}] for r in batch]
            finishes = [[] for _ in batch]
            for turn in range(5):
                for row, messages in zip(batch, conversations):
                    messages.append({"role": "user", "content": row["user_turns"][turn]})
                texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                         for m in conversations]
                tokens = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda")
                generated = model.generate(**tokens, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, pad_token_id=tokenizer.pad_token_id)
                for i, sequence in enumerate(generated[:, tokens.input_ids.shape[1]:].tolist()):
                    stop = next((j for j, token in enumerate(sequence) if token in eos_ids), None)
                    finishes[i].append("stop" if stop is not None else "length")
                    answer = tokenizer.decode(sequence[:stop] if stop is not None else sequence,
                                              skip_special_tokens=True)
                    conversations[i].append({"role": "assistant", "content": answer})
                print(f"batch {offset // args.batch_size + 1}, turn {turn + 1}/5", flush=True)
            for row, messages, reasons in zip(batch, conversations, finishes):
                result = {k: v for k, v in row.items() if k not in {"system_message", "user_turns"}}
                result.update(messages=messages, responses=[m["content"] for m in messages if m["role"] == "assistant"],
                              turn_finish_reasons=reasons, model=MODEL, prompt_condition="base")
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"Saved {len(done) + min(offset + len(batch), len(pending))}/{len(rows)} conversations; "
                  f"elapsed {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
