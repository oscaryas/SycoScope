"""
Cache mean-pooled per-sentence hidden states from SAE/results/*.jsonl generations
to a frozen artifact (sentence table + per-layer float16 memmaps). SAE training
(SAE/sae/) reads only this artifact and never touches the model.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import inference
from utils.model import DEFAULT_MODEL, cleanup, load_model_and_tokenizer

DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "activations"
# 6 evenly-spaced non-boundary indices into the 33-entry hidden_states tuple
# (index 0 = embeddings, 32 = final layer) for Llama-3-8B's 32 transformer blocks.
DEFAULT_LAYERS = [round(i / 7 * 32) for i in range(1, 7)]


# --------------------------------------------------------------------------
# Shared helpers (Phase 1, Phase 3, and both validation gates all call these,
# so the gates actually exercise the production code path).
# --------------------------------------------------------------------------


def make_response_id(dataset: str, row_id, prompt_col: str, sample_idx: int) -> str:
    return f"{dataset}__{row_id}__{prompt_col}__{sample_idx}"


def build_full_text(tokenizer, prompt: str, response: str) -> tuple[str, str]:
    chat_prefix = inference.build_chat_prompt(tokenizer, prompt, system_prompt=None)
    return chat_prefix, chat_prefix + response


def char_span_to_token_span(offsets, char_start: int, char_end: int) -> tuple[int, int] | None:
    tok_start = None
    tok_end = None
    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if end > char_start and start < char_end:
            if tok_start is None:
                tok_start = i
            tok_end = i + 1
    if tok_start is None:
        return None
    return tok_start, tok_end


def response_token_span(offsets, chat_prefix_len: int) -> tuple[int, int]:
    """Token span covering the response portion (chars >= chat_prefix_len) of full_text."""
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


def normalize_ws(s: str) -> str:
    return " ".join(s.strip().split())


def load_tokenizer_only(model_name: str):
    # Deliberately not utils.model.load_model_and_tokenizer: Phase 1 only needs
    # the tokenizer, and loading the 8B model's weights just to segment text
    # would waste minutes and several GB for no reason.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_spacy(model_name: str):
    nlp = spacy.load(model_name, exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
    nlp.add_pipe("sentencizer")
    return nlp


def iter_input_records(input_dir: Path):
    for path in sorted(input_dir.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_response_texts(input_dir: Path) -> dict[str, tuple[str, str]]:
    texts = {}
    for rec in iter_input_records(input_dir):
        response_id = make_response_id(rec["dataset"], rec["row_id"], rec["prompt_col"], rec["sample_idx"])
        texts[response_id] = (rec["prompt"], rec["response"])
    return texts


def hash_input_files(input_dir: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(input_dir.glob("*.jsonl")):
        hasher.update(path.name.encode("utf-8"))
        hasher.update(hashlib.sha256(path.read_bytes()).digest())
    return hasher.hexdigest()


def get_code_version() -> str:
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL) != 0
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# Phase 1 — build & freeze the sentence table (CPU/tokenizer only, no model weights).
# --------------------------------------------------------------------------


def build_sentence_table(records, tokenizer, nlp, max_length: int, skip_log: list):
    """
    
    """
    
    prepared = []
    for rec in records: # rec is sample dict
        response = rec.get("response") or ""
        response_id = make_response_id(rec["dataset"], rec["row_id"], rec["prompt_col"], rec["sample_idx"]) #return f"{dataset}__{row_id}__{prompt_col}__{sample_idx}"
        if not response.strip():
            skip_log.append({"response_id": response_id, "reason": "empty_response"})
            continue
        chat_prefix, full_text = build_full_text(tokenizer, rec["prompt"], response) #build the full prompt + repsonse sequence; adds resposne to inference.build_chat_prompt
        n_tokens = len(tokenizer(full_text, add_special_tokens=False)["input_ids"]) 
        if n_tokens > max_length: # 
            skip_log.append({"response_id": response_id, "reason": "too_long", "detail": n_tokens})
            continue
        prepared.append(
            {
                "response_id": response_id,
                "dataset": rec["dataset"],
                "row_id": str(rec["row_id"]),
                "prompt_col": rec["prompt_col"],
                "sample_idx": rec["sample_idx"],
                "model": rec["model"],
                "response": response,
                "chat_prefix": chat_prefix,
                "full_text": full_text,
                "n_tokens_full_text": n_tokens,
            }
        )

    print(f"Phase 1: segmenting {len(prepared)} responses with spaCy ...")
    docs = nlp.pipe((p["response"] for p in prepared), batch_size=256) # generator of doc objects

    sentence_rows = []
    response_rows = []
    global_idx = 0
    response_idx = 0
    for p, doc in tqdm(zip(prepared, docs), total=len(prepared), desc="tokenize+span"):
        enc = tokenizer(p["full_text"], add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        chat_prefix_len = len(p["chat_prefix"])

        sent_rows = []
        for sent in doc.sents: # for every sentence in doc
            text = sent.text
            if not text.strip(): #blank string
                skip_log.append({"response_id": p["response_id"], "reason": "blank_sentence"})
                continue
            char_start, char_end = sent.start_char, sent.end_char
            # turn char-span into token start - end spans. tokenizer provides token spans. also offset based on input
            span = char_span_to_token_span(offsets, chat_prefix_len + char_start, chat_prefix_len + char_end)
            if span is None:
                skip_log.append(
                    {"response_id": p["response_id"], "reason": "degenerate_token_span", "detail": text[:80]}
                )
                continue
            tok_start, tok_end = span
            sent_rows.append(
                {
                    "response_id": p["response_id"],
                    "dataset": p["dataset"],
                    "row_id": p["row_id"],
                    "prompt_col": p["prompt_col"],
                    "sample_idx": p["sample_idx"],
                    "sent_idx": len(sent_rows),
                    "text": text,
                    "char_start": char_start,
                    "char_end": char_end,
                    "tok_start": tok_start,
                    "tok_end": tok_end,
                    "n_tokens": tok_end - tok_start,
                }
            )

        if not sent_rows:
            skip_log.append({"response_id": p["response_id"], "reason": "no_valid_sentences"})
            continue

        resp_tok_start, resp_tok_end = response_token_span(offsets, chat_prefix_len)

        for row in sent_rows:
            row["global_idx"] = global_idx
            row["response_idx"] = response_idx
            sentence_rows.append(row)
            global_idx += 1

        response_rows.append(
            {
                "response_idx": response_idx,
                "response_id": p["response_id"],
                "dataset": p["dataset"],
                "row_id": p["row_id"],
                "prompt_col": p["prompt_col"],
                "sample_idx": p["sample_idx"],
                "model": p["model"],
                "n_sentences": len(sent_rows),
                "n_tokens_full_text": p["n_tokens_full_text"],
                "resp_tok_start": resp_tok_start,
                "resp_tok_end": resp_tok_end,
            }
        )
        response_idx += 1
    #sentence rows collates sentences, re
    return sentence_rows, response_rows


def reset_output_dir(output_dir: Path):
    patterns = [
        "sentences.parquet",
        "responses.parquet",
        "_manifest.json",
        "meta.json",
        "skipped.jsonl",
        ".done_response_ids.txt",
        "layer_*.npy",
        "response_means_layer_*.npy",
    ]
    for pattern in patterns:
        for p in output_dir.glob(pattern):
            p.unlink()


def load_or_build_sentence_table(args, output_dir: Path, tokenizer, nlp):
    manifest_path = output_dir / "_manifest.json"
    sentences_path = output_dir / "sentences.parquet"
    responses_path = output_dir / "responses.parquet"
    input_dir = Path(args.input)
    input_hash = hash_input_files(input_dir)

    if args.force:
        reset_output_dir(output_dir)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["input_hash"] != input_hash
            or manifest["model"] != args.model
            or manifest["max_length"] != args.max_length
            or manifest["limit"] != args.limit
        ):
            raise RuntimeError(
                f"Existing artifact at {output_dir} was built with a different input/model/max_length/limit. "
                "Re-run with --force to discard it and rebuild."
            )
        print("Phase 1: sentence table already frozen, reusing.")
        return (
            pd.read_parquet(sentences_path),
            pd.read_parquet(responses_path),
            manifest["n_skip_events"],
            input_hash,
        )

    print("Phase 1: building sentence table ...")
    records = list(iter_input_records(input_dir))
    if args.limit:
        records = records[: args.limit]

    skip_log = []
    sentence_rows, response_rows = build_sentence_table(records, tokenizer, nlp, args.max_length, skip_log)

    sentences_df = pd.DataFrame(sentence_rows)
    responses_df = pd.DataFrame(response_rows)
    sentences_df.to_parquet(sentences_path, index=False)
    responses_df.to_parquet(responses_path, index=False)
    with open(output_dir / "skipped.jsonl", "w", encoding="utf-8") as f:
        for entry in skip_log:
            f.write(json.dumps(entry) + "\n")


    manifest = {
        "input_hash": input_hash,
        "model": args.model,
        "max_length": args.max_length,
        "limit": args.limit,
        "n_sentences": len(sentences_df),
        "n_responses": len(responses_df),
        "n_skip_events": len(skip_log),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Phase 1: {len(sentences_df)} sentences from {len(responses_df)} responses "
        f"({len(skip_log)} skip events) frozen to {sentences_path.name} / {responses_path.name}"
    )
    return sentences_df, responses_df, len(skip_log), input_hash


# --------------------------------------------------------------------------
# Phase 2 — preallocate memmaps.
# --------------------------------------------------------------------------


def open_or_create_memmap(path: Path, shape: tuple[int, int]) -> np.memmap:
    if path.exists():
        arr = np.lib.format.open_memmap(path, mode="r+")
        if tuple(arr.shape) != shape:
            raise ValueError(f"shape mismatch resuming {path}: {arr.shape} vs expected {shape}")
        return arr
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)


# --------------------------------------------------------------------------
# Phase 3 — batched forward loop, resume-safe.
# --------------------------------------------------------------------------


def run_forward_pass(args, sentences_df, responses_df, response_texts, output_dir: Path, hidden_size: int):
    n_sentences = len(sentences_df)
    n_responses = len(responses_df)

    layer_arrays = {
        layer: open_or_create_memmap(output_dir / f"layer_{layer:02d}.npy", (n_sentences, hidden_size))
        for layer in args.layers
    }
    response_mean_arrays = {
        layer: open_or_create_memmap(output_dir / f"response_means_layer_{layer:02d}.npy", (n_responses, hidden_size))
        for layer in args.layers
    }

    done_path = output_dir / ".done_response_ids.txt"
    done = set(done_path.read_text(encoding="utf-8").splitlines()) if done_path.exists() else set()

    pending_df = responses_df[~responses_df["response_id"].isin(done)].sort_values(
        "n_tokens_full_text", ascending=False
    )

    if pending_df.empty:
        print("Phase 3: nothing to do, all responses already cached.")
        for arr in {**layer_arrays, **response_mean_arrays}.values():
            arr.flush()
        return

    sent_by_response: dict[str, list[tuple[int, int, int]]] = {}
    for row in sentences_df.itertuples(index=False):
        sent_by_response.setdefault(row.response_id, []).append((row.global_idx, row.tok_start, row.tok_end))

    print(f"Loading model {args.model} for Phase 3 ...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    tokenizer.padding_side = "right"

    pending_records = list(pending_df.itertuples(index=False))
    newly_done: list[str] = []
    batches_since_flush = 0
    n_batches = (len(pending_records) + args.batch_size - 1) // args.batch_size

    with tqdm(total=len(pending_records), desc="cache", unit="response") as progress:
        for batch_i, batch in enumerate(inference.iter_batches(pending_records, args.batch_size)):
            full_texts = []
            for row in batch:
                prompt, response = response_texts[row.response_id]
                _, full_text = build_full_text(tokenizer, prompt, response)
                full_texts.append(full_text)

            inputs = tokenizer(
                full_texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=False)

            for layer in args.layers: # get sent activation for each layer
                h = out.hidden_states[layer]
                sent_idx_buf, sent_vec_buf = [], []
                resp_idx_buf, resp_vec_buf = [], []
                for i, row in enumerate(batch):
                    for global_idx, tok_start, tok_end in sent_by_response.get(row.response_id, []):
                        sent_idx_buf.append(global_idx)
                        sent_vec_buf.append(h[i, tok_start:tok_end, :].float().mean(dim=0))
                    resp_idx_buf.append(row.response_idx)
                    resp_vec_buf.append(h[i, row.resp_tok_start : row.resp_tok_end, :].float().mean(dim=0))
                if sent_idx_buf:
                    layer_arrays[layer][np.array(sent_idx_buf)] = torch.stack(sent_vec_buf).half().cpu().numpy()
                response_mean_arrays[layer][np.array(resp_idx_buf)] = (
                    torch.stack(resp_vec_buf).half().cpu().numpy()
                )

            del out
            newly_done.extend(row.response_id for row in batch)
            batches_since_flush += 1
            progress.update(len(batch))

            is_last_batch = batch_i == n_batches - 1
            if batches_since_flush >= args.flush_every or is_last_batch:
                for arr in {**layer_arrays, **response_mean_arrays}.values():
                    arr.flush()
                with open(done_path, "a", encoding="utf-8") as f:
                    f.write("\n".join(newly_done) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                newly_done = []
                batches_since_flush = 0
                torch.cuda.empty_cache()

    cleanup(model, tokenizer)


# --------------------------------------------------------------------------
# Phase 4 — meta.json (written last = artifact-complete marker).
# --------------------------------------------------------------------------


def write_meta(output_dir: Path, args, n_sentences: int, n_responses: int, n_skip_events: int, input_hash: str, hidden_size: int):
    meta = {
        "model": args.model,
        "layers": args.layers,
        "hidden_size": hidden_size,
        "pooling": "mean",
        "dtype": "float16",
        "n_sentences": n_sentences,
        "n_responses": n_responses,
        "n_skip_events": n_skip_events,
        "max_length": args.max_length,
        "input_files": sorted(p.name for p in Path(args.input).glob("*.jsonl")),
        "input_hash": input_hash,
        "code_version": get_code_version(),
        "tokenizer_name_or_path": args.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_script_args": vars(args),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# Validation gates — both reuse the exact helpers above, run via --validate-only.
# --------------------------------------------------------------------------


def validate_token_spans(sentences_df, response_texts, tokenizer, n: int = 20, seed: int = 0) -> dict:
    sample = sentences_df.sample(min(n, len(sentences_df)), random_state=seed)
    mismatches = []
    for row in sample.itertuples():
        prompt, response = response_texts[row.response_id]
        chat_prefix, full_text = build_full_text(tokenizer, prompt, response)
        enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        decoded = tokenizer.decode(enc["input_ids"][row.tok_start : row.tok_end], skip_special_tokens=True)
        if normalize_ws(decoded) != normalize_ws(row.text):
            mismatches.append({"response_id": row.response_id, "expected": row.text, "decoded": decoded})
    pass_rate = 1 - len(mismatches) / len(sample)
    return {"pass_rate": pass_rate, "n": len(sample), "mismatches": mismatches, "passed": pass_rate >= 0.95}


def validate_batched_vs_single(model, tokenizer, responses_df, response_texts, layers, n: int = 5, seed: int = 0) -> dict:
    sample = responses_df.sample(min(n, len(responses_df)), random_state=seed)
    rows = list(sample.itertuples())
    full_texts = []
    for row in rows:
        prompt, response = response_texts[row.response_id]
        _, full_text = build_full_text(tokenizer, prompt, response)
        full_texts.append(full_text)

    batch_inputs = tokenizer(full_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        batch_out = model(**batch_inputs, output_hidden_states=True, use_cache=False)

    max_diff_by_layer = {}
    for layer in layers:
        h_batch = batch_out.hidden_states[layer]
        max_diff = 0.0
        for i, row in enumerate(rows):
            batched_vec = h_batch[i, row.resp_tok_start : row.resp_tok_end, :].float().mean(dim=0)
            single_inputs = tokenizer([full_texts[i]], return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                single_out = model(**single_inputs, output_hidden_states=True, use_cache=False)
            single_vec = single_out.hidden_states[layer][0, row.resp_tok_start : row.resp_tok_end, :].float().mean(dim=0)
            max_diff = max(max_diff, (batched_vec - single_vec).abs().max().item())
        max_diff_by_layer[layer] = max_diff

    return {"max_diff_by_layer": max_diff_by_layer, "passed": all(d <= 1e-2 for d in max_diff_by_layer.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT_DIR), help="Dir globbed for *.jsonl generations")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS, help="hidden_states indices to cache")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192, help="full chat_prefix+response token cap; over = skip")
    parser.add_argument("--limit", type=int, default=None, help="Cap total responses processed (for smoke testing)")
    parser.add_argument("--flush-every", type=int, default=20, help="batches between memmap flush + done-set fsync")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--force", action="store_true", help="Discard any existing partial/complete artifact and rebuild")
    parser.add_argument("--validate-only", action="store_true", help="Run Phase 1 + validation gates only, then exit")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed for validation gates")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer_only(args.model)
    nlp = load_spacy(args.spacy_model)

    sentences_df, responses_df, n_skip_events, input_hash = load_or_build_sentence_table(
        args, output_dir, tokenizer, nlp
    )
    response_texts = load_response_texts(Path(args.input))

    if args.validate_only:
        print("Validation gate (a): token-span round-trip ...")
        gate_a = validate_token_spans(sentences_df, response_texts, tokenizer, n=20, seed=args.seed)
        print(f"  pass_rate={gate_a['pass_rate']:.2%} ({gate_a['n']} sampled) -> {'PASS' if gate_a['passed'] else 'FAIL'}")
        for m in gate_a["mismatches"]:
            print(f"    MISMATCH {m['response_id']}: expected={m['expected']!r} decoded={m['decoded']!r}")

        print(f"Loading model {args.model} for validation gate (b) ...")
        model, val_tokenizer = load_model_and_tokenizer(args.model)
        val_tokenizer.padding_side = "right"
        gate_b = validate_batched_vs_single(model, val_tokenizer, responses_df, response_texts, args.layers, n=5, seed=args.seed)
        for layer, diff in gate_b["max_diff_by_layer"].items():
            print(f"  layer {layer}: max_abs_diff={diff:.4g}")
        print(f"Gate (b): {'PASS' if gate_b['passed'] else 'FAIL'}")
        cleanup(model, val_tokenizer)

        overall = gate_a["passed"] and gate_b["passed"]
        print(f"\nOverall: {'PASS' if overall else 'FAIL'}")
        sys.exit(0 if overall else 1)

    hidden_size = AutoConfig.from_pretrained(args.model).hidden_size
    run_forward_pass(args, sentences_df, responses_df, response_texts, output_dir, hidden_size)

    done_path = output_dir / ".done_response_ids.txt"
    done = set(done_path.read_text(encoding="utf-8").splitlines()) if done_path.exists() else set()
    if set(responses_df["response_id"]) <= done:
        write_meta(output_dir, args, len(sentences_df), len(responses_df), n_skip_events, input_hash, hidden_size)
        print(f"Artifact complete: {output_dir}")
    else:
        remaining = len(set(responses_df["response_id"]) - done)
        print(
            f"Artifact incomplete: {remaining} response(s) not yet cached. "
            "meta.json not written; rerun the same command to resume."
        )


if __name__ == "__main__":
    main()
