"""Teacher-force existing AITA flip responses through matched Llama-3 models.

Commands: manifest, extract, summarize. Both passes use the instruct tokenizer
and its exact chat-template text; output hashes enforce token-level pairing.
"""
import argparse
import hashlib
import json
from pathlib import Path


def records(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def manifest(judged, out, per_cell):
    counts = {"both_nta": 0, "mixed": 0}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("w") as f:
        for x in records(judged):
            a, b = x["original_post_verdict"], x["flipped_story_verdict"]
            cell = "both_nta" if a == b == "NTA" else "mixed" if (a == "NTA") != (b == "NTA") else None
            if cell is None or counts[cell] >= per_cell:
                continue
            if not all(x.get(prefix + suffix) for prefix in ("original_post", "flipped_story")
                       for suffix in ("_prompt", "_response")):
                continue
            counts[cell] += 1
            for side, prefix in (("original", "original_post"), ("flipped", "flipped_story")):
                row = {"response_id": f"AITA-NTA-FLIP__{x['row_id']}__{prefix}__0",
                       "pair_id": str(x["row_id"]), "cell": cell, "side": side,
                       "verdict": x[prefix + "_verdict"],
                       "prompt": x[prefix + "_prompt"], "response": x[prefix + "_response"]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if all(v >= per_cell for v in counts.values()):
                break
    if any(v < per_cell for v in counts.values()):
        raise ValueError(f"insufficient examples: {counts}")
    print(counts)


def token_span(offsets, a, b):
    hits = [i for i, (x, y) in enumerate(offsets) if x != y and y > a and x < b]
    return (hits[0], hits[-1] + 1) if hits else None


def response_start(offsets, prefix_len):
    # Excludes a BPE token crossing the prompt/response boundary.
    return next((i for i, (a, b) in enumerate(offsets) if a >= prefix_len and b > a), len(offsets))


def sentence_spans(text, prefix_len, offsets, nlp):
    for sent in nlp(text).sents:
        raw = sent.text
        a = prefix_len + sent.start_char + len(raw) - len(raw.lstrip())
        b = prefix_len + sent.end_char - (len(raw) - len(raw.rstrip()))
        if a < b:
            span = token_span(offsets, a, b)
            if span:
                yield span[0], span[1], text[a-prefix_len:b-prefix_len]


def sae_encoder(path):
    import torch
    cfg = json.loads((path / "config.json").read_text())
    weights = torch.load(path / "sae.pt", map_location="cpu", weights_only=True)
    if "state_dict" in weights:
        weights = weights["state_dict"]
    W, be, bd = (weights[k].float() for k in ("W_enc", "b_enc", "b_dec"))
    assert W.shape == (cfg["n_latents"], cfg["d_in"])
    def encode(states):
        z = torch.relu((torch.stack(states) - bd) @ W.T + be)
        val, ind = torch.topk(z, cfg["k"], dim=-1)
        return torch.zeros_like(z).scatter_(-1, ind, val).tolist()
    return cfg["layer"], encode


def extract(args):
    import numpy as np
    import spacy
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = list(records(args.manifest))
    if args.limit is not None:
        rows = rows[:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    assert tokenizer.is_fast
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    sae = sae_encoder(args.sae_dir) if args.sae_dir and args.model_kind == "chat" else None
    layer = sae[0] if sae else 14
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True).eval().cuda()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    states_dir = Path(args.out_dir) / (args.model_kind + "_states")
    states_dir.mkdir(exist_ok=True)
    output = Path(args.out_dir) / (args.model_kind + ".jsonl")
    completed = {x["response_id"] for x in records(output)} if args.resume and output.exists() else set()
    with output.open("a" if args.resume else "w") as f, torch.inference_mode():
        for idx, row in enumerate(rows):
            if row["response_id"] in completed:
                continue
            prefix = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True)
            enc = tokenizer(prefix + row["response"], add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = enc["input_ids"], enc["offset_mapping"]
            start = response_start(offsets, len(prefix))
            if start >= len(ids) or len(ids) > args.max_tokens:
                print("skip", row["response_id"], len(ids), flush=True)
                continue
            x = torch.tensor([ids], device="cuda")
            out = model(x, output_hidden_states=True)
            # Logit at position t-1 predicts the observed token at t.
            lp = torch.log_softmax(out.logits[0, start-1:-1].float(), dim=-1)
            target = x[0, start:]
            forced = lp.gather(-1, target[:, None]).squeeze(-1)
            greedy = lp.argmax(-1)
            hidden = out.hidden_states[layer][0].float()
            spans = list(sentence_spans(row["response"], len(prefix), offsets, nlp))
            states = [hidden[a:b].mean(0).cpu() for a, b, _ in spans]
            state_name = hashlib.sha256(row["response_id"].encode()).hexdigest() + ".npz"
            np.savez_compressed(
                states_dir / state_name,
                sentence_states=np.stack([s.numpy() for s in states]).astype(np.float16)
                if states else np.empty((0, hidden.shape[-1]), dtype=np.float16),
                pre_response_state=hidden[start-1].cpu().numpy().astype(np.float16),
                early_response_state=hidden[start:min(start+32, len(ids))].mean(0).cpu().numpy().astype(np.float16))
            features = sae[1](states) if sae and states else None
            sentences = []
            for j, (a, b, text) in enumerate(spans):
                lo, hi = max(a, start)-start, b-start
                if lo >= hi:
                    continue
                sentences.append({"text": text, "token_start": a, "token_end": b,
                                  "mean_forced_logprob": float(forced[lo:hi].mean().cpu()),
                                  "sae_activations": features[j] if features else None})
            result = {k: row[k] for k in ("response_id", "pair_id", "cell", "side", "verdict")}
            result.update({"model_kind": args.model_kind, "model_id": model.config._name_or_path,
                           "token_ids_sha256": hashlib.sha256(
                               torch.tensor(ids, dtype=torch.int32).numpy().tobytes()).hexdigest(),
                           "n_tokens": len(ids), "response_start": start,
                           "mean_forced_logprob": float(forced.mean().cpu()),
                           "greedy_match_fraction": float((greedy == target).float().mean().cpu()),
                           "states_file": str(states_dir / state_name),
                           "states_layer": layer,
                           "sentences": sentences})
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(args.model_kind, idx+1, len(rows), row["response_id"], len(ids), flush=True)
            del x, out, lp, hidden
            torch.cuda.empty_cache()


def summarize(chat, base):
    c = {x["response_id"]: x for x in records(chat)}
    b = {x["response_id"]: x for x in records(base)}
    matched = []
    for rid in sorted(c.keys() & b.keys()):
        x, y = c[rid], b[rid]
        if (x["token_ids_sha256"], x["response_start"], x["n_tokens"]) != (
                y["token_ids_sha256"], y["response_start"], y["n_tokens"]):
            raise ValueError("token mismatch: " + rid)
        matched.append({"response_id": rid, "cell": x["cell"], "side": x["side"],
                        "chat_minus_base_logprob": x["mean_forced_logprob"]-y["mean_forced_logprob"],
                        "chat_greedy": x["greedy_match_fraction"],
                        "base_greedy": y["greedy_match_fraction"]})
    print(json.dumps({"matched": len(matched), "rows": matched}, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--judged", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--per-cell", type=int, default=4)
    e = sub.add_parser("extract")
    e.add_argument("--manifest", type=Path, required=True)
    e.add_argument("--model-path", type=Path, required=True)
    e.add_argument("--tokenizer-path", type=Path, required=True)
    e.add_argument("--model-kind", choices=("chat", "base"), required=True)
    e.add_argument("--out-dir", type=Path, required=True)
    e.add_argument("--sae-dir", type=Path)
    e.add_argument("--max-tokens", type=int, default=4096)
    e.add_argument("--limit", type=int)
    e.add_argument("--resume", action="store_true",
                   help="append missing response IDs after an interrupted pass")
    s = sub.add_parser("summarize")
    s.add_argument("--chat", type=Path, required=True)
    s.add_argument("--base", type=Path, required=True)
    a = p.parse_args()
    if a.cmd == "manifest":
        manifest(a.judged, a.out, a.per_cell)
    elif a.cmd == "extract":
        extract(a)
    else:
        summarize(a.chat, a.base)


if __name__ == "__main__":
    main()
