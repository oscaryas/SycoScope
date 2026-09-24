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
    a single user string."""
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + list(messages)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def iter_batches(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


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


def resolve_terminators(model, tokenizer) -> list[int]:
    """Model-agnostic end-of-turn terminator list: the UNION of the
    checkpoint's own generation_config eos ids (authoritative per family --
    e.g. Gemma ends turns with <end_of_turn>, which is NOT
    tokenizer.eos_token, so replacing that list would run every generation to
    max_new_tokens), tokenizer.eos_token, and Llama-3's <|eot_id|> (older
    Llama-3 checkpoints ship a generation_config listing only
    <|end_of_text|>)."""
    gen_cfg_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    ids = list(gen_cfg_eos) if isinstance(gen_cfg_eos, (list, tuple)) else ([gen_cfg_eos] if gen_cfg_eos is not None else [])
    ids.append(tokenizer.eos_token_id)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id not in (None, getattr(tokenizer, "unk_token_id", None)):
        ids.append(eot_id)
    return sorted({i for i in ids if isinstance(i, int)})


def generate_from_rendered(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 150,
    batch_size: int = 8,
) -> tuple[list[str], list[bool]]:
    """Greedy, batched generation from prompts that are ALREADY chat-rendered
    (e.g. from build_chat_prompt / build_chat_prompt_multiturn), including
    their own BOS -- tokenized with add_special_tokens=False so it isn't
    doubled. Left-padded and chunked by batch_size for throughput; requires
    tokenizer.padding_side == "left" (right-padding would corrupt position
    ids for every prompt but the longest in a chunk under a causal LM).

    Returns (responses, truncated): truncated[i] is True when prompt i's
    generation hit max_new_tokens without emitting a terminator or pad
    token -- i.e. that response is incomplete.
    """
    if tokenizer.padding_side != "left":
        raise ValueError(
            "generate_from_rendered requires tokenizer.padding_side == 'left' for "
            "correct batched causal-LM generation; got 'right'."
        )
    terminators = resolve_terminators(model, tokenizer)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    responses: list[str] = []
    truncated: list[bool] = []
    for chunk in iter_batches(prompts, batch_size):
        # Prompts here are ALREADY chat-rendered, so their tail carries the
        # model's own generation-prompt suffix (e.g. Llama-3's
        # <|start_header_id|>assistant<|end_header_id|>\n\n or Gemma's
        # <start_of_turn>model). HF's default truncation_side is "right",
        # which would silently drop that suffix for an over-long prompt and
        # make the model continue the wrong turn. Force left-truncation for
        # this call only, and always restore the tokenizer's original
        # setting afterward so this function doesn't leave the caller's
        # tokenizer object mutated.
        original_truncation_side = getattr(tokenizer, "truncation_side", "right")
        tokenizer.truncation_side = "left"
        try:
            inputs = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024,
                add_special_tokens=False,
            )
        finally:
            tokenizer.truncation_side = original_truncation_side
        # A row's attention_mask sums to its own post-truncation token count
        # (left-padding is 0s, real tokens are 1s). Any row sitting exactly
        # at max_length either needed truncation or is a very rare prompt
        # that happens to be exactly max_length tokens long -- either way,
        # flagging it as "possibly truncated" costs nothing extra (no second
        # tokenizer call) and errs toward visibility.
        n_at_cap = int((inputs["attention_mask"].sum(dim=1) == 1024).sum())
        if n_at_cap:
            print(
                f"  WARNING: {n_at_cap}/{len(chunk)} prompts in this batch reached "
                "max_length=1024 and were LEFT-truncated (earliest tokens dropped so each "
                "prompt's own generation-prompt suffix is preserved)."
            )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=terminators,
                pad_token_id=pad_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for i in range(output_ids.shape[0]):
            new_tokens = output_ids[i, input_len:]
            is_trunc = (
                bool(len(new_tokens))
                and int(new_tokens[-1]) not in terminators
                and int(new_tokens[-1]) != pad_id
            )
            truncated.append(is_trunc)
            responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    n_truncated = sum(truncated)
    if n_truncated:
        print(
            f"  WARNING: {n_truncated}/{len(prompts)} generations hit the max_new_tokens="
            f"{max_new_tokens} cap without an end-of-turn token -- those responses are "
            "INCOMPLETE (thinking models need a much larger budget)."
        )
    return responses, truncated


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
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=pad_id,
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
