import inspect
import json
from pathlib import Path
from typing import Literal, get_args, get_origin, get_type_hints

SKILLS_DIR: Path | None = None


def set_skills_dir(path: Path):
    global SKILLS_DIR
    SKILLS_DIR = path


def load_skill(skill_name: str) -> str:
    """Load detailed instructions for a named step. Returns the full skill text."""
    if SKILLS_DIR is None:
        return "error: skills directory not configured"
    skill_file = SKILLS_DIR / f"{skill_name}.md"
    if not skill_file.exists():
        available = [p.stem for p in SKILLS_DIR.glob("*.md")]
        return f"error: skill '{skill_name}' not found. Available: {available}"
    return skill_file.read_text()


def _hint_to_json_schema(hint) -> dict:
    origin = get_origin(hint)
    if origin is Literal:
        return {"type": "string", "enum": list(get_args(hint))}
    if hint is int:
        return {"type": "integer"}
    if hint is float:
        return {"type": "number"}
    if hint is bool:
        return {"type": "boolean"}
    return {"type": "string"}


def build_tool_schemas(fns: list) -> list:
    tools = []
    for fn in fns:
        sig = inspect.signature(fn)
        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}
        properties = {}
        required = []
        for name, param in sig.parameters.items():
            properties[name] = _hint_to_json_schema(hints.get(name, str))
            if param.default is inspect.Parameter.empty:
                required.append(name)
        tools.append({
            "name": fn.__name__,
            "description": (fn.__doc__ or "").strip(),
            "input_schema": {"type": "object", "properties": properties, "required": required},
        })
    return tools


MAX_OUTPUT_CHARS = 8000


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        suffix = f"\n...[truncated, {len(text)} chars total]"
        return text[:MAX_OUTPUT_CHARS - len(suffix)] + suffix
    return text


def _run_tool(name: str, args: dict, registry: dict) -> str:
    if name not in registry:
        return f"error: unknown tool '{name}'"
    try:
        result = registry[name](**args)
        if result is None:
            return "ok"
        return _truncate(json.dumps(result) if isinstance(result, (dict, list)) else str(result))
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def _run_anthropic_loop(system_prompt, user_prompt, schemas, registry, model, max_iterations):
    import anthropic
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_prompt}]

    for _ in range(max_iterations):
        try:
            response = client.messages.create(
                model=model, max_tokens=4096,
                system=system_prompt, tools=schemas, messages=messages,
            )
        except Exception as e:
            return f"error: API call failed: {type(e).__name__}: {e}"
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = _run_tool(block.name, block.input, registry)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            break  # model signalled tool_use but sent no tool_use blocks

    return "error: max_iterations reached"


def _to_openai_schemas(anthropic_schemas: list) -> list:
    return [
        {"type": "function", "function": {
            "name": s["name"],
            "description": s["description"],
            "parameters": s["input_schema"],
        }}
        for s in anthropic_schemas
    ]


def _run_openai_loop(system_prompt, user_prompt, schemas, registry, model, max_iterations):
    import openai
    client = openai.OpenAI()
    openai_schemas = _to_openai_schemas(schemas)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for _ in range(max_iterations):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages,
                tools=openai_schemas if openai_schemas else None,
            )
        except Exception as e:
            return f"error: API call failed: {type(e).__name__}: {e}"
        choice = response.choices[0]
        msg = choice.message

        if choice.finish_reason == "stop" or not msg.tool_calls:
            return msg.content or ""

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = _run_tool(tc.function.name, args, registry)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "error: max_iterations reached"


def run_tool_loop(
    system_prompt: str,
    user_prompt: str,
    tools: dict,
    model: str,
    max_iterations: int = 50,
) -> str:
    all_fns = [load_skill] + list(tools.values())
    schemas = build_tool_schemas(all_fns)
    registry = {"load_skill": load_skill, **tools}

    if model.startswith("claude-"):
        return _run_anthropic_loop(system_prompt, user_prompt, schemas, registry, model, max_iterations)
    elif model.startswith("gpt-"):
        return _run_openai_loop(system_prompt, user_prompt, schemas, registry, model, max_iterations)
    else:
        return f"error: unknown model prefix in '{model}'. Expected 'claude-' or 'gpt-'."
