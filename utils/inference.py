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


def iter_batches(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


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
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

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
