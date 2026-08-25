"""
Tokenization and batched-generation helpers for instruct models.
"""

import torch


def build_chat_prompt(tokenizer, user_message: str, system_prompt: str | None = None) -> str:
    """Wrap a raw user message in the model's chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_chat_prompt_multiturn(tokenizer, messages: list, system_prompt: str | None = None) -> str:
    """Like build_chat_prompt, but for a pre-built multi-turn message list
    (e.g. SyPR's persona-calibration history + final utterance) rather than
    a single user string. system_prompt, when given, is prepended as a system
    message (e.g. Nemotron's "detailed thinking on" reasoning toggle)."""
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + list(messages)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def iter_batches(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def resolve_terminators(model, tokenizer) -> list[int]:
    """Model-agnostic end-of-turn terminator list: the UNION of the
    checkpoint's own generation_config eos ids (authoritative per family --
    e.g. Gemma ends turns with <turn|>/<end_of_turn>, which is NOT
    tokenizer.eos_token, so replacing that list would run every generation to
    max_new_tokens), tokenizer.eos_token, and Llama-3's <|eot_id|> (older
    Llama-3 checkpoints ship a generation_config listing only
    <|end_of_text|>)."""
    gen_cfg_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    ids = list(gen_cfg_eos) if isinstance(gen_cfg_eos, (list, tuple)) else ([gen_cfg_eos] if gen_cfg_eos is not None else [])
    ids.append(tokenizer.eos_token_id)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id not in (None, tokenizer.unk_token_id):
        ids.append(eot_id)
    return sorted({i for i in ids if isinstance(i, int)})


def strip_reasoning(text: str) -> str:
    """Remove visible reasoning wrappers from a generated response before
    MECHANICAL answer parsing (LLM judges should keep the full text):
    - Qwen3-style <think>...</think>: keep only text after the LAST closing
      tag. An OPENED but unclosed block means generation was truncated
      mid-think -- return "" so parsers report no answer instead of matching
      letters/words inside the reasoning.
    - Gemma-4-style channels: keep text after the last <channel|> marker
      when the markers survived decoding."""
    if "<think>" in text and "</think>" not in text:
        return ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    if "<channel|>" in text:
        text = text.rsplit("<channel|>", 1)[-1]
    return text.strip()


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    system_prompt: str | None = None,
    max_new_tokens: int = 512,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> list[str]:
    """
    Apply the chat template to each prompt, tokenize as a left-padded batch,
    generate, and return only the newly generated text per example.
    """
    chat_texts = [build_chat_prompt(tokenizer, p, system_prompt) for p in prompts]
    # apply_chat_template already renders the BOS token into the string, so
    # add_special_tokens=False here avoids a doubled <|begin_of_text|>.
    inputs = tokenizer(
        chat_texts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(model.device)
    terminators = resolve_terminators(model, tokenizer)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:

        gen_kwargs.update(temperature=temperature, top_p=top_p)

    with torch.no_grad():
        output_ids = model.generate(**inputs, eos_token_id=terminators, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    responses = []
    for i in range(output_ids.shape[0]):
        gen_ids = output_ids[i, input_len:]
        responses.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return responses
