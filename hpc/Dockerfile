# Runtime image for the SycoScope sycophancy-generation pipeline on SLURM/HPC.
#
# Deliberately does NOT bake in the SycoScope repo checkout -- the repo is
# mounted/bound live from a git clone on each cluster's persistent filesystem
# (see hpc/README.md), so editing tool_calling/tasks/sycophancy/scripts/*.py
# (e.g. tuning --generation-batch-size) never requires rebuilding this image.
#
# Consumed two different ways:
#   - CSCS Clariden: referenced directly via a registry URL in the EDF
#     `image =` field (Container Engine auto-pulls + caches it); see
#     hpc/cscs/generate.edf.toml.
#   - Compute Canada Trillium: built into a .sif off-cluster
#     (`apptainer build sycoscope.sif docker://<registry>/sycoscope:latest`,
#     or `apptainer build --remote` to avoid needing local root) and
#     transferred once via rsync/scp -- Trillium's compute nodes have no
#     internet, so nothing pulls from a registry on the cluster itself.
#
# Build:
#   docker build -t <registry>/sycoscope:latest -f hpc/Dockerfile .
#   docker push <registry>/sycoscope:latest
#
# Versions below are pinned to exactly what's resolved in the repo's
# uv.lock as of this writing (see hpc/README.md if the lockfile has since
# moved -- keep these in sync with `uv export --frozen` output).

FROM python:3.13-slim

# --nv (Apptainer) / GH200 NVIDIA driver passthrough both expect a CUDA
# userspace matching the cluster's driver; torch's own bundled CUDA runtime
# (installed via pip below, no separate system CUDA toolkit needed) handles
# this — this base image intentionally stays driver-agnostic. If a cluster's
# `--nv`/GH200 passthrough reports a CUDA-version mismatch, pin an
# explicit `--index-url https://download.pytorch.org/whl/cu1XX` matching
# that cluster's driver instead of the default PyPI wheel below.

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Exact versions pinned from uv.lock (torch/transformers/accelerate/anthropic/
# huggingface-hub are the only ones actually imported by the generation +
# judge scripts at runtime; the rest of pyproject.toml's dependency list --
# datasets, spacy, scikit-learn, pyarrow, matplotlib -- is not needed here).
RUN pip install --no-cache-dir \
        torch==2.12.1 \
        transformers==5.13.0 \
        accelerate==1.14.0 \
        anthropic==0.116.0 \
        "huggingface_hub[cli]==1.22.0" \
        pandas==3.0.3 \
        numpy==2.5.1 \
        tqdm==4.68.3

# No CMD/ENTRYPOINT: the sbatch scripts invoke `python <script>.py ...`
# explicitly against the live-mounted repo checkout.
