#!/bin/bash
# Score both OOD targets (AITA via eval_moral.py, cross-cell via eval_cross_cell.py)
# against each of the 5 C-sweep probe sets. Extraction for both targets was done
# ONCE into the canonical llama31_5k_subset run-dir; this symlinks those cached
# activations into each C-dir so scoring is CPU-only and doesn't re-extract.
set -e
cd "$(dirname "$0")"

BASE=../../prompt_probes/results/llama31_5k_subset
AITA=../../tool_calling/tasks/sycophancy/results/generations/openrouter_llama31

for C in 0.01 0.1 1.0 10.0 100.0; do
  RUN="../../prompt_probes/results/llama31_5k_subset_C${C}"
  ln -sfn "../llama31_5k_subset/eval_moral" "$RUN/eval_moral"
  ln -sfn "../llama31_5k_subset/eval_cross_cell" "$RUN/eval_cross_cell"

  echo "=== C=$C: eval_moral (AITA) ==="
  python3 eval_moral.py --run-name "llama31_5k_subset_C${C}" \
    --yta-judged "$AITA/aita_yta/verdict_judged.jsonl" \
    --og-judged "$AITA/aita_nta_og/verdict_judged.jsonl" \
    --flip-judged "$AITA/aita_nta_flip/judged.jsonl"

  echo "=== C=$C: eval_cross_cell ==="
  python3 eval_cross_cell.py --run-name "llama31_5k_subset_C${C}"
done
