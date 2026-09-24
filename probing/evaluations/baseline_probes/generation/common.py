from __future__ import annotations

import os
import subprocess
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from probing.utils.probes_common import json_dump

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds; exponential backoff


def decoding_metadata(args) -> dict:
    return {
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature) if args.do_sample else None,
        "top_p": float(args.top_p) if args.do_sample else None,
        "max_new_tokens": int(args.max_new_tokens),
        "num_return_sequences": 1,
    }


def add_generation_args(parser, default_max_new_tokens: int) -> None:
    parser.add_argument("--model", required=True, help="OpenRouter model slug, e.g. meta-llama/llama-3.1-8b-instruct")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-workers", type=int, default=16, help="Concurrent OpenRouter requests")
    parser.add_argument("--max-new-tokens", type=int, default=default_max_new_tokens)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Optional smoke-test limit; default is the complete dataset")
    parser.add_argument("--do-sample", action="store_true", help="Opt in to stochastic decoding")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)


REQUEST_TIMEOUT_SECONDS = 60.0


def _openrouter_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (add it to .env or export it before running)")
    # The openai SDK's default per-request timeout is 10 minutes -- far too
    # long to sit on a genuinely hung connection before _call_with_retry's
    # retry/backoff even gets a chance to run. A occasional hang (confirmed:
    # a 1591-row run had 16 requests sit ESTABLISHED with zero completions
    # for 10+ minutes while a fresh diagnostic call to the same model
    # completed in 6.4s) should fail fast into the retry loop instead.
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)


def _call_with_retry(client, messages: list[dict], args) -> dict:
    from openai import APIConnectionError, APIStatusError, APITimeoutError

    kwargs = {
        "max_tokens": args.max_new_tokens,
        # Reasoning models (e.g. Qwen3.8) otherwise spend the whole max_tokens
        # budget on hidden chain-of-thought before ever emitting `content` --
        # confirmed via a raw diagnostic call: a 3.4k-char AITA prompt hit
        # finish_reason="length" with 564 reasoning tokens and content=None
        # at max_tokens=512. We want each model's direct behavioral response
        # (comparable across models, matching how the non-reasoning models in
        # this study were generated via plain greedy decoding with no
        # reasoning phase at all), not chain-of-thought, so reasoning is
        # disabled for every call. OpenRouter no-ops this for models that
        # don't support reasoning.
        "extra_body": {"reasoning": {"enabled": False}},
    }
    if args.do_sample:
        kwargs.update(temperature=args.temperature, top_p=args.top_p)
    else:
        kwargs.update(temperature=0.0)

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(model=args.model, messages=messages, **kwargs)
            choice = completion.choices[0]
            return {
                "content": (choice.message.content or "").strip(),
                "finish_reason": choice.finish_reason,
                # Captured even though reasoning is requested disabled above --
                # some providers/models only partially honor that, so this is
                # worth keeping rather than trusting the disable flag silently.
                "reasoning": getattr(choice.message, "reasoning", None) or "",
            }
        except (APIConnectionError, APITimeoutError) as error:
            print(f"RETRY: attempt {attempt+1}/{MAX_RETRIES} for model={args.model} hit {type(error).__name__}: {error}", flush=True)
        except APIStatusError as error:
            print(f"RETRY: attempt {attempt+1}/{MAX_RETRIES} for model={args.model} hit APIStatusError {error.status_code}", flush=True)
            if error.status_code not in (429, 500, 502, 503, 529):
                raise
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BASE_DELAY * (2**attempt))
    raise RuntimeError(f"OpenRouter request failed after {MAX_RETRIES} attempts for model={args.model}")


def generate_via_openrouter(messages_list: list[list[dict]], args) -> list[str]:
    """Generate one response per conversation via OpenRouter's OpenAI-compatible
    chat completions API. args.model is an OpenRouter model slug (e.g.
    "meta-llama/llama-3.1-8b-instruct"), not an HF repo id -- no local model or
    tokenizer is loaded. Concurrency (args.max_workers) replaces local GPU
    batching since these are independent HTTP requests, not one batched
    forward pass. do_sample=False maps to temperature=0.0 (deterministic
    request, not a guaranteed-identical-to-greedy-decoding result -- provider
    determinism varies).

    Returns just the response strings for call-site convenience; per-row
    finish_reason (needed to know whether a response was cut off by
    max_new_tokens rather than finishing naturally) is available via
    generate_via_openrouter_with_finish_reasons for callers that want it."""
    return [r["content"] for r in generate_via_openrouter_with_finish_reasons(messages_list, args)]


def generate_via_openrouter_with_finish_reasons(messages_list: list[list[dict]], args) -> list[dict]:
    """Same as generate_via_openrouter but returns [{"content", "finish_reason", "reasoning"}, ...].
    finish_reason == "length" means the response was truncated by max_new_tokens,
    not a natural stop -- prints a truncation-count warning since a silently
    truncated response (e.g. a verdict explanation cut off mid-sentence) can
    confuse downstream judging. reasoning is the model's chain-of-thought text
    when the provider returns one despite the disable-reasoning request in
    _call_with_retry -- normally empty, but captured (not just checked for
    leakage into `content` after the fact) so partial non-compliance is
    visible rather than silently dropped."""
    client = _openrouter_client()
    results: list[dict] = [{"content": "", "finish_reason": None, "reasoning": ""}] * len(messages_list)
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(_call_with_retry, client, messages, args): index
            for index, messages in enumerate(messages_list)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as error:
                # A single permanently-failed row (retries exhausted, or a
                # non-retryable APIStatusError) must not take down the whole
                # run -- there's no incremental checkpointing here, so an
                # unhandled exception mid-run would silently discard every
                # already-completed row along with it. Record the failure on
                # this row only and keep going; write_jsonl still gets a
                # complete file, callers can filter on finish_reason=="error".
                print(f"FAILED row {index}: {type(error).__name__}: {error}", flush=True)
                # "request_failed", not "error" -- some providers legitimately
                # return finish_reason="error" with real content attached
                # (seen in practice), so this needs its own distinct marker
                # for a row that produced no content at all.
                results[index] = {"content": "", "finish_reason": "request_failed", "reasoning": ""}
            n_done += 1
            if n_done % 25 == 0 or n_done == len(messages_list):
                print(f"Generated {n_done}/{len(messages_list)}")
    n_truncated = sum(1 for r in results if r["finish_reason"] == "length")
    if n_truncated:
        print(f"WARNING: {n_truncated}/{len(results)} responses were truncated by --max-new-tokens "
              f"({args.max_new_tokens}) rather than finishing naturally -- consider raising it.")
    n_with_reasoning = sum(1 for r in results if r["reasoning"])
    if n_with_reasoning:
        print(f"NOTE: {n_with_reasoning}/{len(results)} responses returned a non-empty reasoning trace "
              f"despite reasoning being requested disabled -- saved in each row's \"reasoning\" field.")
    return results


def source_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_metadata(output: Path, args, dataset_type: str, source: dict, extra: dict | None = None) -> None:
    rows = []
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    sample_ids = [str(row.get("row_id", row.get("id", index))) for index, row in enumerate(rows)]
    prompt_payload = [row.get("messages", row.get("prompt", row.get("question"))) for row in rows]
    digest = lambda value: hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "format_version": 1,
        "model": args.model,
        "dataset_type": dataset_type,
        "source": source,
        "decoding": decoding_metadata(args),
        "seed": args.seed,
        "output": str(output),
        "baseline_compatibility": {
            "n_rows": len(rows),
            "sample_ids_sha256": digest(sample_ids),
            "prompt_payload_sha256": digest(prompt_payload),
        },
    }
    if extra:
        metadata.update(extra)
    json_dump(output.with_suffix(".metadata.json"), metadata)
