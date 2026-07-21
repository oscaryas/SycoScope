"""
Minimal single-shot Anthropic client for LLM-judge tasks (e.g.
SAE/pipeline/label_clusters.py). Not tool_calling/tool_loop.py's agentic
tool-calling loop -- these are one-shot "send a prompt, parse JSON back"
calls, a different shape entirely.

Requires ANTHROPIC_API_KEY set in the environment (read implicitly by the
anthropic SDK -- no key is passed explicitly here).
"""

import json
import time

DEFAULT_JUDGE_MODEL = "claude-sonnet-5"


def call_judge(
    prompt: str,
    system: str | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    max_retries: int = 3,
    max_tokens: int = 1024,
) -> str:
    import anthropic

    client = anthropic.Anthropic()
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(**kwargs)
            return "".join(block.text for block in response.content if block.type == "text")
        except anthropic.APIStatusError as e:
            if 400 <= e.status_code < 500:
                raise
            last_error = e
            time.sleep(2**attempt)
        except (anthropic.APIConnectionError, anthropic.RateLimitError) as e:
            last_error = e
            time.sleep(2**attempt)
    raise last_error


def parse_json_response(text: str) -> dict:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No JSON object found in judge response: {text!r}")
