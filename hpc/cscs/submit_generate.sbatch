#!/bin/bash
# CSCS Clariden (GH200 vCluster) generation job for one (model, dataset)
# pair. Submitted directly for a smoke test, or via hpc/submit_matrix.sh for
# the full model x dataset matrix.
#
# Usage (manual smoke test):
#   sbatch --export=ALL,MODEL="Qwen/Qwen3-8B",DATASET=ss,N=8,BATCH_SIZE=4 \
#       hpc/cscs/submit_generate.sbatch
#
# Requires (see hpc/README.md for one-time setup):
#   - ~/.sycoscope_secrets.env (chmod 600) exporting HF_TOKEN, ANTHROPIC_API_KEY
#   - hpc/prefetch_weights.py already run for $MODEL, staged at the HF_HOME
#     path set in hpc/cscs/generate.edf.toml
#   - a persistent `git clone` of this repo at
#     /capstor/store/<tenant>/<group>/sycoscope-repo (matches
#     generate.edf.toml's `workdir` -- deliberately NOT $SCRATCH, which is
#     purge-swept every 14 days on Clariden and would silently delete
#     in-progress checkpoints)
#   - REGISTRY_PLACEHOLDER in generate.edf.toml replaced with the pushed
#     image (see hpc/Dockerfile)
#
# Resume/checkpoint: social_generate.py / moral_generate.py resume by
# counting lines already in --out for the same --seed -- resubmitting this
# exact script (same MODEL/DATASET/OUT/N/SEED) after a time-limit kill is
# safe. This script self-resubmits automatically (see bottom) when
# summary.json hasn't appeared yet, capped at MAX_RESUBMITS to avoid a
# runaway loop if something is actually broken rather than just slow.

#SBATCH --job-name=sycoscope-gen
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/slurm-%x-%j.out

set -euo pipefail

# --- required env (validate early, fail loudly rather than silently) -------
: "${MODEL:?Set MODEL, e.g. --export=ALL,MODEL=Qwen/Qwen3-8B}"
: "${DATASET:?Set DATASET, e.g. --export=ALL,DATASET=oeq}"

N="${N:-2000}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_RESUBMITS="${MAX_RESUBMITS:-20}"

# Persistent storage -- NOT $SCRATCH (purge-swept every 14 days on
# Clariden's iopsstor). Must match generate.edf.toml's `workdir`.
REPO_DIR="/capstor/store/CHANGE_ME_TENANT/CHANGE_ME_GROUP/sycoscope-repo"
SCRIPTS_DIR="${REPO_DIR}/tool_calling/tasks/sycophancy/scripts"
MODEL_SLUG="${MODEL//\//__}"
RESULTS_DIR="${REPO_DIR}/tool_calling/tasks/sycophancy/results/generations/multimodel/${MODEL_SLUG}/${DATASET}"
OUT="${RESULTS_DIR}/checkpoint.jsonl"
SUMMARY="${RESULTS_DIR}/summary.json"
RESUBMIT_COUNT_FILE="${RESULTS_DIR}/.resubmit_count"

mkdir -p "${RESULTS_DIR}" logs

# --- secrets (never printed/committed; see hpc/README.md) ------------------
source "${HOME}/.sycoscope_secrets.env"

# --- pick the right generation script for this dataset ---------------------
case "${DATASET}" in
    oeq|ss|aita_yta)
        SCRIPT="social_generate.py"
        ;;
    aita_nta_flip|aita_nta_og)
        SCRIPT="moral_generate.py"
        ;;
    *)
        echo "Unknown DATASET '${DATASET}' -- expected one of: oeq ss aita_yta aita_nta_flip aita_nta_og" >&2
        exit 1
        ;;
esac

echo "[$(date -Iseconds)] job ${SLURM_JOB_ID}: ${MODEL} / ${DATASET} -> ${OUT}"

srun -ul --environment="${REPO_DIR}/hpc/cscs/generate.edf.toml" bash -c "
    cd '${SCRIPTS_DIR}' &&
    python ${SCRIPT} \
        --dataset '${DATASET}' \
        --model '${MODEL}' \
        --seed ${SEED} \
        --generation-batch-size ${BATCH_SIZE} \
        --max-new-tokens ${MAX_NEW_TOKENS} \
        --out '${OUT}' \
        --n ${N}
"

# --- self-resubmit if not yet finished --------------------------------------
if [ -f "${SUMMARY}" ]; then
    echo "[$(date -Iseconds)] ${MODEL} / ${DATASET} complete (summary.json present)."
    rm -f "${RESUBMIT_COUNT_FILE}"
else
    COUNT=$(cat "${RESUBMIT_COUNT_FILE}" 2>/dev/null || echo 0)
    COUNT=$((COUNT + 1))
    if [ "${COUNT}" -ge "${MAX_RESUBMITS}" ]; then
        echo "[$(date -Iseconds)] Hit MAX_RESUBMITS=${MAX_RESUBMITS} without finishing -- not resubmitting. Investigate manually (checkpoint at ${OUT})." >&2
        exit 1
    fi
    echo "${COUNT}" > "${RESUBMIT_COUNT_FILE}"
    echo "[$(date -Iseconds)] Not yet finished (resubmit ${COUNT}/${MAX_RESUBMITS}) -- queuing continuation."
    sbatch --dependency=afterany:"${SLURM_JOB_ID}" \
        --export=ALL,MODEL="${MODEL}",DATASET="${DATASET}",N="${N}",SEED="${SEED}",BATCH_SIZE="${BATCH_SIZE}",MAX_NEW_TOKENS="${MAX_NEW_TOKENS}",MAX_RESUBMITS="${MAX_RESUBMITS}" \
        "${REPO_DIR}/hpc/cscs/submit_generate.sbatch"
fi
