"""Model set — the secondary researcher knob.

Each model entry needs:
  model_id    — HuggingFace model identifier (used by vLLM in pipeline.sh)
  path        — local subdirectory under PHASE_H_DATA_ROOT for outputs
  layer       — extraction layer (Lu canonical = N/2 of total layers)
  hidden_dim  — residual stream dimensionality
  color       — matplotlib hex color for radar / scatter overlays
  display     — pretty name for plot legends

To run with different models:
  cp configs/models.py configs/my_models.py  # edit
  PHASE_H_MODELS=configs/my_models.py python bootstrap_radar.py

`pipeline.sh <tag>` reads ALL of these from this file (single source of truth),
so layer / hidden_dim / paths can never drift between pipeline and analysis.
"""

MODELS: dict[str, dict] = {
    "phi": {
        "model_id":   "microsoft/Phi-3.5-mini-instruct",
        "path":       "phi-3.5-mini",
        "layer":      16,
        "hidden_dim": 3072,
        "color":      "#1f77b4",
        "display":    "Phi-3.5-mini",
    },
    "llama": {
        "model_id":   "meta-llama/Llama-3.2-3B-Instruct",
        "path":       "llama-3.2-3b",
        "layer":      14,
        "hidden_dim": 3072,
        "color":      "#d62728",
        "display":    "Llama-3.2-3B",
    },
    "qwen": {
        "model_id":   "Qwen/Qwen2.5-3B-Instruct",
        "path":       "qwen2.5-3b",
        "layer":      18,
        "hidden_dim": 2048,
        "color":      "#2ca02c",
        "display":    "Qwen-2.5-3B",
    },
}
