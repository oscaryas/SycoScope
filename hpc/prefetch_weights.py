#!/usr/bin/env python3
"""One-time login-node weight staging for the SycoScope HPC generation jobs.

Run this on the login node (NOT inside a GPU allocation -- it's a plain
network download, no GPU needed) before submitting any generation jobs on
either cluster. It stages each model's weights into a shared HF_HOME on
persistent, non-purged storage, so the GPU jobs themselves can run with
HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 and never touch the network --
required on Trillium (compute nodes have no internet at all) and strongly
recommended on CSCS Clariden too (compute-node internet exists but is a
shared, rate-limit-risk NAT egress that CSCS asks users not to hammer with
bulk pulls).

Usage:
    # CSCS Clariden (login node, inside a `git clone` of this repo):
    export HF_HOME=/capstor/store/<tenant>/<group>/huggingface
    source ~/.sycoscope_secrets.env   # for HF_TOKEN
    python hpc/prefetch_weights.py

    # Compute Canada Trillium (login node):
    export HF_HOME=$PROJECT/huggingface
    source ~/.sycoscope_secrets.env
    python hpc/prefetch_weights.py

    # Stage only specific models:
    python hpc/prefetch_weights.py --model Qwen/Qwen3-8B --model google/gemma-4-12B-it

Requires only `huggingface_hub` (not the full pipeline's torch/transformers
stack) -- safe to run with the system Python on a login node, or
`pip install --user huggingface_hub`, if the container isn't set up yet.
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download

# Exact model IDs used by the sycophancy generation pipeline -- kept in sync
# with the MODELS registry in
# tool_calling/tasks/sycophancy/sycophancy_model_registry.py. Add/remove
# entries here as the model matrix changes; this list is intentionally not
# imported from the registry module itself so this script has no dependency
# on torch/transformers being installed on the login node.
DEFAULT_MODELS = [
    "Qwen/Qwen3-8B",
    "google/gemma-4-12B-it",
    "Qwen/Qwen3.8-27B",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="Model repo id to stage (repeatable). Defaults to all models in DEFAULT_MODELS.",
    )
    args = parser.parse_args()
    models = args.models or DEFAULT_MODELS

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        print(
            "HF_HOME is not set -- refusing to guess. Export it to the persistent "
            "storage path your generation jobs will use (e.g. "
            "/capstor/store/<tenant>/<group>/huggingface on CSCS, "
            "$PROJECT/huggingface on Trillium) before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "Warning: HF_TOKEN is not set. Gated/private model repos will fail to "
            "download; public repos will still work.",
            file=sys.stderr,
        )

    print(f"Staging {len(models)} model(s) into HF_HOME={hf_home}")
    for model in models:
        print(f"  -> {model}")
        snapshot_download(
            repo_id=model,
            token=token,
            # Skip files a generation job never touches (original checkpoint
            # formats when safetensors are present, training artifacts).
            ignore_patterns=["*.bin", "*.pt", "*.h5", "*.msgpack", "*.onnx", "original/*"],
        )
    print("Done. Generation jobs can now run with HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1.")


if __name__ == "__main__":
    main()
