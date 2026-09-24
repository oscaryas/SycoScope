"""Shared generation plumbing for every probing/evaluations/generation/generate_*.py script.

Two interchangeable backends, selected per run with --backend:
  - openrouter: concurrent chat-completions requests (generate_openrouter)
  - local:      a locally loaded HF model, greedy batched decoding via
                utils.inference.generate_from_rendered (generate_local)

Both take a list of chat conversations (each a list of {"role", "content"}
dicts) and return one {"content", "finish_reason", "reasoning"} dict per
conversation, so dataset scripts never branch on the backend except to pick
the generator (make_generator).

--system-prompt names a contrastive cell slug from probing.utils.common
(e.g. ctrl_affiliation); its text is read from
probing/data/system_prompt/sycophancy_probe_prompt_pairs.json, polarity chosen
with --system-prompt-polarity. --system-prompt-text passes a literal system
prompt instead (e.g. Nemotron's "detailed thinking on").
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from probing.utils.probes_common import json_dump

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds; exponential backoff


def decoding_metadata(args) -> dict:
    return {
        "backend": getattr(args, "backend", "openrouter"),
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature) if args.do_sample else None,
        "top_p": float(args.top_p) if args.do_sample else None,
        "max_new_tokens": int(args.max_new_tokens),
        "num_return_sequences": 1,
    }


def add_generation_args(parser, default_max_new_tokens: int) -> None:
    parser.add_argument("--model", required=True,
                        help="OpenRouter model slug (e.g. meta-llama/llama-3.1-8b-instruct) for --backend openrouter, "
                             "HF repo id or local path for --backend local")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-workers", type=int, default=16, help="Concurrent OpenRouter requests")
    parser.add_argument("--max-new-tokens", type=int, default=default_max_new_tokens)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Optional smoke-test limit; default is the complete dataset")
    parser.add_argument("--do-sample", action="store_true", help="Opt in to stochastic decoding (openrouter backend only)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)


def add_backend_args(parser, default_backend: str = "openrouter", system_prompt: bool = True) -> None:
    """--backend {local,openrouter}, --batch-size (local), and unless
    system_prompt=False the --system-prompt / --system-prompt-polarity /
    --system-prompt-text trio."""
    parser.add_argument("--backend", choices=["local", "openrouter"], default=default_backend)
    parser.add_argument("--batch-size", type=int, default=16, help="Local backend generation batch size")
    if not system_prompt:
        return
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--system-prompt",
        default=None,
        metavar="CELL_SLUG",
        help="Cell slug from probing.utils.common.CELL_SLUGS whose prompt text is prepended as a system "
             "message (text from sycophancy_probe_prompt_pairs.json); omit for none",
    )
    group.add_argument("--system-prompt-text", default=None,
                       help="Literal system prompt text, e.g. 'detailed thinking on' for Nemotron")
    parser.add_argument("--system-prompt-polarity", choices=["sycophantic", "non_sycophantic"],
                        default="sycophantic", help="Which side of the --system-prompt cell's pair to use")


def resolve_system_prompt(args) -> str | None:
    """The actual system-prompt TEXT for this run, or None."""
    literal = getattr(args, "system_prompt_text", None)
    if literal:
        return literal
    slug = getattr(args, "system_prompt", None)
    if slug is None:
        return None
    return system_prompt_for_cell(slug, getattr(args, "system_prompt_polarity", "sycophantic"))


def system_prompt_for_cell(slug: str, polarity: str = "sycophantic") -> str | None:
    from probing.utils import common

    if slug == common.NEUTRAL_SLUG:
        return None
    if polarity not in common.POLARITIES:
        raise ValueError(f"unknown polarity {polarity!r}; valid: {list(common.POLARITIES)}")
    by_slug = {pair["slug"]: pair for pair in common.load_prompt_pairs()}
    if slug not in by_slug:
        raise ValueError(f"unknown cell slug {slug!r}; valid: {sorted(by_slug)} (+ {common.NEUTRAL_SLUG!r})")
    return by_slug[slug][polarity]


def system_prompt_metadata(args) -> dict:
    return {
        "system_prompt_cell": getattr(args, "system_prompt", None),
        "system_prompt_polarity": getattr(args, "system_prompt_polarity", None) if getattr(args, "system_prompt", None) else None,
        "system_prompt": resolve_system_prompt(args),
    }


def with_system_prompt(messages: list[dict], system_prompt: str | None) -> list[dict]:
    """Prepend system_prompt (if any) to a conversation. A conversation that
    already opens with its own system message (e.g. SycoNBench's debate stance)
    gets the two joined into ONE system message, cell prompt first, since most
    chat templates accept only a single leading system turn."""
    if not system_prompt:
        return list(messages)
    if messages and messages[0]["role"] == "system":
        merged = {"role": "system", "content": system_prompt + "\n\n" + messages[0]["content"]}
        return [merged] + list(messages[1:])
    return [{"role": "system", "content": system_prompt}] + list(messages)


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


def generate_openrouter(messages_list: list[list[dict]], args) -> list[dict]:
    """Generate one response per conversation via OpenRouter's OpenAI-compatible
    chat completions API; returns [{"content", "finish_reason", "reasoning"}, ...].
    (Formerly generate_via_openrouter_with_finish_reasons -- same behavior.)

    args.model is an OpenRouter model slug (e.g. "meta-llama/llama-3.1-8b-instruct"),
    not an HF repo id. Concurrency (args.max_workers) replaces local GPU
    batching. do_sample=False maps to temperature=0.0 (deterministic request,
    not a guaranteed-identical-to-greedy-decoding result -- provider
    determinism varies).

    finish_reason == "length" means the response was truncated by
    max_new_tokens, not a natural stop -- prints a truncation-count warning
    since a silently truncated response can confuse downstream judging.
    reasoning is the model's chain-of-thought text when the provider returns
    one despite the disable-reasoning request -- normally empty, but captured
    so partial non-compliance is visible rather than silently dropped."""
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
                # run. Record the failure on this row only and keep going.
                # "request_failed", not "error" -- some providers legitimately
                # return finish_reason="error" with real content attached.
                print(f"FAILED row {index}: {type(error).__name__}: {error}", flush=True)
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


def local_device_map():
    """cuda -> "auto", else mps, else cpu (same choice the steering generators make)."""
    import torch

    if torch.cuda.is_available():
        return "auto"
    if torch.backends.mps.is_available():
        return {"": "mps"}
    return {"": "cpu"}


def load_local_model(model_name: str):
    from utils.model import load_model_and_tokenizer

    device_map = local_device_map()
    print(f"Loading {model_name} (device_map={device_map})...")
    return load_model_and_tokenizer(model_name, device_map=device_map)


def generate_local(messages_list: list[list[dict]], model, tokenizer, args) -> list[dict]:
    """Local-model counterpart of generate_openrouter: renders each
    conversation with the model's own chat template and decodes greedily via
    utils.inference.generate_from_rendered. Same return shape as
    generate_openrouter; finish_reason is "length" for a cap-hit (incomplete)
    response and "stop" otherwise."""
    from utils.inference import generate_from_rendered

    if getattr(args, "do_sample", False):
        raise ValueError("--do-sample is not supported by --backend local (generate_from_rendered is greedy)")
    rendered = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_list
    ]
    responses, truncated = generate_from_rendered(
        model, tokenizer, rendered, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )
    return [
        {"content": response, "finish_reason": "length" if trunc else "stop", "reasoning": ""}
        for response, trunc in zip(responses, truncated)
    ]


def make_generator(args):
    """messages_list -> list[{"content", "finish_reason", "reasoning"}] for
    args.backend. The local model is loaded lazily on first call and reused."""
    if args.backend == "openrouter":
        return lambda messages_list: generate_openrouter(messages_list, args)
    if args.do_sample:
        raise ValueError("--do-sample is not supported by --backend local (generate_from_rendered is greedy)")
    state: dict = {}

    def _generate(messages_list: list[list[dict]]) -> list[dict]:
        if not messages_list:
            return []
        if "model" not in state:
            state["model"], state["tokenizer"] = load_local_model(args.model)
        return generate_local(messages_list, state["model"], state["tokenizer"], args)

    return _generate


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
        **system_prompt_metadata(args),
        "baseline_compatibility": {
            "n_rows": len(rows),
            "sample_ids_sha256": digest(sample_ids),
            "prompt_payload_sha256": digest(prompt_payload),
        },
    }
    if extra:
        metadata.update(extra)
    json_dump(output.with_suffix(".metadata.json"), metadata)
