#!/usr/bin/env python3
"""
Model registry for sycophancy probing.

Add a new model by inserting a dict entry — no code changes needed elsewhere.
Unsupported models raise ValueError naming the missing field.
"""
import re
import torch


MODELS = {
    "google/gemma-3-12b-it": {
        "family": "gemma",
        "n_layers": 46,
        "n_heads": 16,
        "hidden_dim": 5120,
        "mlp_dim": 10240,
        "head_dim": 320,         # hidden_dim // n_heads
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": "text_config",   # Gemma nests arch info under model.config.text_config
        "answer_token_id": 105,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "family": "llama",
        "n_layers": 32,
        "n_heads": 32,
        "hidden_dim": 4096,
        "mlp_dim": 8192,
        "head_dim": 128,         # hidden_dim // n_heads
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": None,
        "answer_token_id": 128007,
    },
    # -------------------------------------------------------------------------
    # Small models — for smoke tests and fast iteration
    # -------------------------------------------------------------------------
    "HuggingFaceTB/SmolLM2-135M": {
        "family": "llama",
        "n_layers": 30,
        "n_heads": 9,
        "hidden_dim": 576,
        "mlp_dim": 1536,
        "head_dim": 64,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": None,
        "answer_token_id": 1,
    },
    "HuggingFaceTB/SmolLM2-360M": {
        "family": "llama",
        "n_layers": 32,
        "n_heads": 15,
        "hidden_dim": 960,
        "mlp_dim": 2560,
        "head_dim": 64,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": None,
        "answer_token_id": 1,
    },
    "HuggingFaceTB/SmolLM2-1.7B": {
        "family": "llama",
        "n_layers": 24,
        "n_heads": 32,
        "hidden_dim": 2048,
        "mlp_dim": 8192,
        "head_dim": 64,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": None,
        "answer_token_id": 1,
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "family": "qwen",
        "n_layers": 24,
        "n_heads": 14,
        "hidden_dim": 896,
        "mlp_dim": 4864,
        "head_dim": 64,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": None,
        "answer_token_id": 151644,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "family": "qwen",
        "n_layers": 28,
        "n_heads": 12,
        "hidden_dim": 1536,
        "mlp_dim": 8960,
        "head_dim": 128,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": None,
        "answer_token_id": 151644,
    },
    "google/gemma-3-1b-it": {
        "family": "gemma",
        "n_layers": 18,
        "n_heads": 8,
        "hidden_dim": 1152,
        "mlp_dim": 6912,
        "head_dim": 256,         # Gemma uses head_dim=256 independent of hidden_dim
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": "text_config",
        "answer_token_id": 105,
    },
    "google/gemma-3-4b-it": {
        "family": "gemma",
        "n_layers": 34,
        "n_heads": 8,
        "hidden_dim": 2560,
        "mlp_dim": 10240,
        "head_dim": 256,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": "text_config",
        "answer_token_id": 105,
    },
    # -------------------------------------------------------------------------
    # Qwen3 — modern reasoning-capable models
    # -------------------------------------------------------------------------
    "Qwen/Qwen3-0.6B": {
        "family": "qwen3",
        "n_layers": 28,
        "n_heads": 16,
        "hidden_dim": 1024,
        "mlp_dim": 3072,
        "head_dim": 64,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": None,
        "answer_token_id": 151644,
    },
    "Qwen/Qwen3-1.7B": {
        "family": "qwen3",
        "n_layers": 28,
        "n_heads": 16,
        "hidden_dim": 2048,
        "mlp_dim": 8192,
        "head_dim": 128,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": None,
        "answer_token_id": 151644,
    },
    "Qwen/Qwen3-4B": {
        "family": "qwen3",
        "n_layers": 36,
        "n_heads": 32,
        "hidden_dim": 2560,
        "mlp_dim": 9216,
        "head_dim": 80,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": None,
        "answer_token_id": 151644,
    },
    "Qwen/Qwen3-8B": {
        "family": "qwen3",
        "n_layers": 36,
        "n_heads": 32,
        "hidden_dim": 4096,
        "mlp_dim": 14336,
        "head_dim": 128,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": None,
        "answer_token_id": 151644,
    },
    "Qwen/Qwen3-14B": {
        "family": "qwen3",
        "n_layers": 40,
        "n_heads": 40,
        "hidden_dim": 5120,
        "mlp_dim": 17920,
        "head_dim": 128,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": None,
        "answer_token_id": 151644,
    },
    "Qwen/Qwen3-32B": {
        "family": "qwen3",
        "n_layers": 64,
        "n_heads": 64,
        "hidden_dim": 5120,
        "mlp_dim": 25600,
        "head_dim": 80,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": None,
        "answer_token_id": 151644,
    },
    # -------------------------------------------------------------------------
    # Opt-in larger models (not part of default reproduction matrix)
    # -------------------------------------------------------------------------
    "google/gemma-3-27b-it": {
        "family": "gemma",
        "n_layers": 62,
        "n_heads": 32,
        "hidden_dim": 7168,
        "mlp_dim": 14336,
        "head_dim": 224,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": "text_config",
        "answer_token_id": 105,
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "family": "llama",
        "n_layers": 80,
        "n_heads": 64,
        "hidden_dim": 8192,
        "mlp_dim": 28672,
        "head_dim": 128,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": None,
        "answer_token_id": 128007,
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "family": "qwen",
        "n_layers": 64,
        "n_heads": 40,
        "hidden_dim": 5120,
        "mlp_dim": 13696,
        "head_dim": 128,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "config_key": None,
        "answer_token_id": 151644,
    },
}

REQUIRED_FIELDS = {"family", "n_layers", "n_heads", "hidden_dim", "mlp_dim",
                   "head_dim", "mha_hook", "mlp_hook", "answer_token_id"}


def get_model_config(model_name: str) -> dict:
    """Return architecture config for model_name, or raise with a clear message."""
    if model_name in MODELS:
        return dict(MODELS[model_name])

    # Try to auto-detect from HuggingFace config for unknown models
    try:
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_name)
        config_source = getattr(hf_config, "text_config", hf_config)

        n_layers = getattr(config_source, "num_hidden_layers", None)
        n_heads = getattr(config_source, "num_attention_heads", None)
        hidden_dim = getattr(config_source, "hidden_size", None)
        mlp_dim = getattr(config_source, "intermediate_size", None)

        if all(v is not None for v in [n_layers, n_heads, hidden_dim, mlp_dim]):
            print(f"WARNING: {model_name} not in registry — auto-detected config. "
                  "Hook paths assumed as self_attn.o_proj / mlp.down_proj. "
                  "Verify these match the actual module names or add to MODELS.")
            return {
                "family": "unknown",
                "n_layers": n_layers,
                "n_heads": n_heads,
                "hidden_dim": hidden_dim,
                "mlp_dim": mlp_dim,
                "head_dim": hidden_dim // n_heads,
                "mha_hook": "self_attn.o_proj",
                "mlp_hook": "mlp.down_proj",
                "config_key": None,
                "answer_token_id": None,
            }
    except Exception:
        pass

    raise ValueError(
        f"Model '{model_name}' is not in the sycophancy registry and could not be "
        f"auto-detected. Add it to MODELS in sycophancy_model_registry.py with fields: "
        f"{sorted(REQUIRED_FIELDS)}"
    )


def _extract_layer_idx(module_name: str) -> int:
    """Parse layer index from a module path like 'model.layers.3.self_attn.o_proj'."""
    match = re.search(r"\.(\d+)\.", module_name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract layer index from module name: {module_name}")


def register_hooks(model, model_config: dict):
    """
    Attach forward hooks to MHA (input to o_proj) and MLP (output of down_proj).

    Returns:
        handles: list of hook handles — call handle.remove() to clean up
        activation_store: dict with keys "mha" and "mlp", each a dict mapping
                          layer_idx -> tensor collected on the last forward pass
    """
    activation_store = {"mha": {}, "mlp": {}}
    handles = []
    mha_suffix = model_config["mha_hook"]
    mlp_suffix = model_config["mlp_hook"]

    for name, module in model.named_modules():
        if name.endswith(mha_suffix):
            layer_idx = _extract_layer_idx(name)

            def mha_pre_hook(m, inp, li=layer_idx):
                # inp[0] shape: (batch, seq_len, n_heads * head_dim)
                activation_store["mha"][li] = inp[0].detach().cpu()

            handles.append(module.register_forward_pre_hook(mha_pre_hook))

        elif name.endswith(mlp_suffix):
            layer_idx = _extract_layer_idx(name)

            def mlp_hook(m, inp, out, li=layer_idx):
                # out shape: (batch, seq_len, hidden_dim)
                activation_store["mlp"][li] = out.detach().cpu()

            handles.append(module.register_forward_hook(mlp_hook))

    if not handles:
        raise RuntimeError(
            f"No modules matched hook paths '{mha_suffix}' or '{mlp_suffix}'. "
            "Check the module names in this model match the registry entries."
        )

    return handles, activation_store


def remove_hooks(handles: list):
    """Remove all registered hooks. Always call this after extraction."""
    for h in handles:
        h.remove()
    handles.clear()
