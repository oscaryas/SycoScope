#!/bin/bash
# Submit the full model x dataset generation matrix as independent SLURM
# jobs on one cluster. 3 models x 4 datasets = 12 jobs -- small enough that
# plain named `sbatch` submissions are simpler than a job array.
#
# Usage:
#   hpc/submit_matrix.sh cscs
#   hpc/submit_matrix.sh trillium
#
#   # Override scale/models/datasets, e.g. for a smoke test:
#   N=8 BATCH_SIZE=4 MODELS="Qwen/Qwen3-8B" DATASETS="ss" hpc/submit_matrix.sh cscs
#
# Run from the repo root on the target cluster's login node, inside the
# persistent `git clone` described in hpc/README.md.

set -euo pipefail

CLUSTER="${1:?Usage: hpc/submit_matrix.sh <cscs|trillium>}"

case "${CLUSTER}" in
    cscs)
        SBATCH_SCRIPT="hpc/cscs/submit_generate.sbatch"
        ;;
    trillium)
        SBATCH_SCRIPT="hpc/trillium/submit_generate.sbatch"
        ;;
    *)
        echo "Unknown cluster '${CLUSTER}' -- expected 'cscs' or 'trillium'" >&2
        exit 1
        ;;
esac

# Matches the MODELS registry entries actually exercised this session (see
# tool_calling/tasks/sycophancy/sycophancy_model_registry.py) and
# hpc/prefetch_weights.py's DEFAULT_MODELS.
MODELS="${MODELS:-Qwen/Qwen3-8B
google/gemma-4-12B-it
Qwen/Qwen3.8-27B}"

DATASETS="${DATASETS:-oeq
ss
aita_yta
aita_nta_flip}"

N="${N:-2000}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

echo "Submitting on ${CLUSTER} via ${SBATCH_SCRIPT}: N=${N} SEED=${SEED} BATCH_SIZE=${BATCH_SIZE}"

while IFS= read -r MODEL; do
    [ -z "${MODEL}" ] && continue
    while IFS= read -r DATASET; do
        [ -z "${DATASET}" ] && continue
        JOB_NAME="$(echo "${MODEL}" | tr '/' '__')-${DATASET}"
        echo "  sbatch --job-name=${JOB_NAME} ${SBATCH_SCRIPT}  (${MODEL} / ${DATASET})"
        sbatch \
            --job-name="${JOB_NAME}" \
            --export=ALL,MODEL="${MODEL}",DATASET="${DATASET}",N="${N}",SEED="${SEED}",BATCH_SIZE="${BATCH_SIZE}",MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
            "${SBATCH_SCRIPT}"
    done <<< "${DATASETS}"
done <<< "${MODELS}"
