#!/usr/bin/env python3
"""
Conservative GPU memory management for sycophancy probing.

Default policy: bfloat16, batch_size=1, device_map=auto, OOM retry with
smaller batch, explicit cleanup between models.
"""
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Optional


def load_model_conservatively(model_name: str, dtype_str: str = "auto"):
    """
    Load a HuggingFace causal LM with conservative memory settings.

    Returns (model, tokenizer, dtype_used).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Choose dtype: prefer bfloat16 if the GPU supports it
    if dtype_str == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            dtype_name = "bfloat16"
        elif torch.cuda.is_available():
            dtype = torch.float16
            dtype_name = "float16"
        else:
            dtype = torch.float32
            dtype_name = "float32"
    else:
        dtype = getattr(torch, dtype_str)
        dtype_name = dtype_str

    print(f"Loading {model_name} with dtype={dtype_name}, device_map=auto, low_cpu_mem_usage=True")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.eval()

    return model, tokenizer, dtype_name


def clear_gpu_memory():
    """Synchronously clear GPU memory caches."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_memory_diagnostics(
    model_name: str,
    dtype: str,
    batch_size: int,
    oom_retries: int = 0,
    memory_policy: str = "conservative",
) -> dict:
    """Return a dict describing current GPU memory state."""
    import torch

    diagnostics = {
        "model_name": model_name,
        "memory_policy": memory_policy,
        "dtype": dtype,
        "batch_size": batch_size,
        "oom_retries": oom_retries,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        diagnostics["gpu_name"] = props.name
        diagnostics["total_vram_gb"] = round(props.total_memory / 1e9, 1)
        diagnostics["peak_allocated_gb"] = round(
            torch.cuda.max_memory_allocated() / 1e9, 2
        )
        diagnostics["peak_reserved_gb"] = round(
            torch.cuda.max_memory_reserved() / 1e9, 2
        )
        torch.cuda.reset_peak_memory_stats()
    else:
        diagnostics["gpu_name"] = "CPU"
        diagnostics["total_vram_gb"] = 0
        diagnostics["peak_allocated_gb"] = 0
        diagnostics["peak_reserved_gb"] = 0

    return diagnostics


def safe_cleanup(
    model=None,
    tokenizer=None,
    hook_handles: Optional[list] = None,
    job_dir: Optional[str] = None,
):
    """
    Delete model references, clear GPU caches, remove hook handles, and
    optionally clean up model shards copied into the job workspace.

    NEVER deletes HuggingFace cache or any path outside job_dir.
    """
    import torch

    # Remove hooks first so no dangling references to model internals
    if hook_handles:
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        hook_handles.clear()

    # Delete model and tokenizer references
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer

    clear_gpu_memory()

    # Remove only job-local model shards (never shared cache)
    if job_dir is not None:
        _clean_job_shards(job_dir)


def _clean_job_shards(job_dir: str):
    """
    Delete *.safetensors files under job_dir/model_shards/ only.

    Guards against broad paths. Refuses to operate on home, cache, or root.
    """
    job_path = Path(job_dir).resolve()
    shards_dir = job_path / "model_shards"

    # Safety: refuse if job_dir looks like a shared or system path
    forbidden_prefixes = [
        Path("/").resolve(),
        Path.home().resolve(),
        Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser().resolve(),
        Path(os.environ.get("AUTOINTERP_HF_HOME", "~/.cache/huggingface")).expanduser().resolve(),
    ]
    for forbidden in forbidden_prefixes:
        if job_path == forbidden:
            print(f"WARNING: Refusing to clean shards in protected path: {job_path}")
            return

    if not shards_dir.exists():
        return

    count = 0
    for shard in shards_dir.glob("*.safetensors"):
        shard.unlink()
        count += 1

    if count:
        print(f"Cleaned {count} model shard(s) from {shards_dir}")
