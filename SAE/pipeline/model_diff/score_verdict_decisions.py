"""Score the first NTA/YTA verdict under controlled plain-text formats.

No system message or chat template is used. Both official models receive the
same Instruct-tokenizer IDs. Candidate likelihoods cover the whole two-token
verdict acronym, and greedy four-token continuations check format validity.
The default preserves the original two-example orders; optional controls
include a bare verdict cue and two counterbalanced four-example orders.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


EXAMPLES = {
    "yta": ("AITA for taking my friend's bicycle without asking, even though I returned it later?\n"
            "Verdict: YTA. Taking someone else's bicycle without asking was wrong.\n\n"),
    "nta": ("AITA for declining to lend my car to a cousin who repeatedly crashed it?\n"
            "Verdict: NTA. You can set boundaries around your own property.\n\n"),
    "yta2": ("AITA for promising to pick up my coworker from the airport and then knowingly leaving them stranded?\n"
             "Verdict: YTA. A deliberate broken commitment left them without the ride you promised.\n\n"),
    "nta2": ("AITA for refusing to falsify a coworker's timesheet, even though they asked me to help?\n"
             "Verdict: NTA. You do not have to lie on their behalf.\n\n"),
}
ORDERS = ("yta_nta", "nta_yta")
FORMAT_ORDERS = {
    "yta_nta": ("yta", "nta"),
    "nta_yta": ("nta", "yta"),
    "plain": (),
    "four_yta_last": ("yta", "nta", "nta2", "yta2"),
    "four_nta_last": ("nta", "yta", "yta2", "nta2"),
}
VERDICTS = ("NTA", "YTA")


def prompt_text(story, order):
    return "".join(EXAMPLES[x] for x in FORMAT_ORDERS[order]) + story + "\nVerdict:"


def records(path):
    for line in Path(path).read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def verdict_start(text):
    match = re.match(r"\s*(NTA|YTA|ESH|NAH)\b", text)
    return match.group(1) if match else "OTHER"


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    candidates = {v: tokenizer.encode(" " + v, add_special_tokens=False) for v in VERDICTS}
    if len(candidates["NTA"]) != len(candidates["YTA"]):
        raise ValueError("verdict token lengths differ")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).eval().cuda()
    rows = list(records(args.manifest))
    orders = tuple(x.strip() for x in args.formats.split(",") if x.strip())
    if not orders or len(orders) != len(set(orders)) or any(x not in FORMAT_ORDERS for x in orders):
        raise ValueError("invalid or duplicate prompt formats")
    if args.limit:
        rows = rows[:args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {(x["response_id"], x["example_order"]) for x in records(args.out)} if args.resume and args.out.exists() else set()
    with args.out.open("a" if args.resume else "w") as file, torch.inference_mode():
        for index, row in enumerate(rows):
            for order in orders:
                if (row["response_id"], order) in done:
                    continue
                prompt = prompt_text(row["prompt"], order)
                prefix = tokenizer.encode(prompt, add_special_tokens=True)
                logprobs = {}
                for verdict in VERDICTS:
                    full = tokenizer.encode(prompt + " " + verdict, add_special_tokens=True)
                    if full[:len(prefix)] != prefix or full[len(prefix):] != candidates[verdict]:
                        raise ValueError("candidate tokenization changed prefix")
                    x = torch.tensor([full], device="cuda")
                    output = model(x)
                    lp = torch.log_softmax(output.logits[0, len(prefix)-1:-1].float(), dim=-1)
                    target = x[0, len(prefix):]
                    logprobs[verdict] = float(lp.gather(-1, target[:, None]).sum().cpu())
                    del x, output, lp
                x = torch.tensor([prefix], device="cuda")
                generated = model.generate(
                    x, attention_mask=torch.ones_like(x), max_new_tokens=4,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id)
                continuation = tokenizer.decode(generated[0, len(prefix):], skip_special_tokens=True)
                margin = logprobs["NTA"] - logprobs["YTA"]
                result = {k: row[k] for k in (
                    "response_id", "pair_id", "side", "stance", "truth", "asserted", "warranted")}
                result.update({"model_kind": args.model_kind, "example_order": order,
                               "prompt_ids_sha256": hashlib.sha256(
                                   torch.tensor(prefix, dtype=torch.int32).numpy().tobytes()).hexdigest(),
                               "prompt_tokens": len(prefix),
                               "candidate_ids": candidates,
                               "nta_logprob": logprobs["NTA"],
                               "yta_logprob": logprobs["YTA"],
                               "nta_minus_yta_logodds": margin,
                               "candidate_choice": "NTA" if margin > 0 else "YTA",
                               "greedy_first_verdict": verdict_start(continuation),
                               "greedy_first_four_tokens": continuation})
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()
                print(args.model_kind, index + 1, len(rows), order,
                      row["response_id"], result["greedy_first_verdict"], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--tokenizer-path", type=Path, required=True)
    p.add_argument("--model-kind", choices=("base", "chat"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--formats", default=",".join(ORDERS),
                   help="Comma-separated prompt formats; default preserves the original two orders")
    run(p.parse_args())


if __name__ == "__main__":
    main()
