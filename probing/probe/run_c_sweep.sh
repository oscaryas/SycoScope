#!/bin/bash
# Run the 5-value C sweep for the llama31_5k_subset initial exploration.
# Each C value gets its own run-dir symlinking the shared activations/
# generations/prompt_split.json (built already), so train_probes.py runs
# unmodified per C with zero code changes.
set -e
cd "$(dirname "$0")"

CELLS="general_baseline pv_implicit pe_implicit ctrl_pure_sycophancy ctrl_obsequiousness"
BASE=../../prompt_probes/results/llama31_5k_subset

for C in 0.01 0.1 1.0 10.0 100.0; do
  RUN="llama31_5k_subset_C${C}"
  echo "=== C=$C -> $RUN ==="
  python3 train_probes.py --run-name "$RUN" --cells $CELLS --C "$C"
done
