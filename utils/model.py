"""
Model loading helpers.
"""

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    dtype: str = "bfloat16",
    device_map: str = "auto",
):
    """
    Load a causal LM and its tokenizer. Pads left and falls back to eos_token
    for pad_token, since Llama-family tokenizers don't define one by default
    and left-padding is required for batched generation.
    """
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device_map
    )
    model.eval()

    return model, tokenizer


def cleanup(model=None, tokenizer=None):
    """Free GPU memory after generation is done."""
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
