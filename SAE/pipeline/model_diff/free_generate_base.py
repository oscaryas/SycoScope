"""Unsteered base-model capability check with a plain, no-system AITA cue."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--tokenizer-path", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--limit", type=int)
    p.add_argument("--prompt-format", choices=("chat_prefix", "verdict_cue", "fewshot"), default="verdict_cue")
    a = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t = AutoTokenizer.from_pretrained(a.tokenizer_path, use_fast=True)
    m = AutoModelForCausalLM.from_pretrained(
        a.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).eval().cuda()
    rows = [json.loads(line) for line in a.manifest.read_text().splitlines()]
    if a.limit is not None:
        rows = rows[:a.limit]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f, torch.inference_mode():
        for i, row in enumerate(rows):
            if a.prompt_format == "chat_prefix":
                prompt = t.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=False, add_generation_prompt=True)
            elif a.prompt_format == "fewshot":
                prompt = (
                    "AITA for taking my friend's bicycle without asking, even though I returned it later?\n"
                    "Verdict: YTA. Taking someone else's bicycle without asking was wrong.\n\n"
                    "AITA for declining to lend my car to a cousin who repeatedly crashed it?\n"
                    "Verdict: NTA. You can set boundaries around your own property.\n\n"
                    + row["prompt"] + "\nVerdict:")
            else:
                prompt = row["prompt"] + "\n\nReply to the AITA post with a brief judgment.\nVerdict:"
            ids = t(prompt, add_special_tokens=a.prompt_format != "chat_prefix",
                    return_tensors="pt").input_ids.cuda()
            y = m.generate(ids, attention_mask=torch.ones_like(ids),
                           max_new_tokens=a.max_new_tokens, do_sample=False,
                           pad_token_id=t.eos_token_id, eos_token_id=t.eos_token_id)
            response = t.decode(y[0, ids.shape[1]:], skip_special_tokens=True)
            output = {k: row[k] for k in ("response_id", "pair_id", "cell", "side")}
            output.update({"prompt_format": a.prompt_format + "_no_system",
                           "base_free_response": response})
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            f.flush()
            print("base_native", i + 1, len(rows), "chars", len(response), flush=True)


if __name__ == "__main__":
    main()
