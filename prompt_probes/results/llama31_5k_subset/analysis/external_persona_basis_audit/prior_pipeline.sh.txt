#!/bin/bash
# Run the assistant-axis pipeline (no judge) for one model from configs/models.py.
#
# Usage:
#   ./pipeline.sh <model_tag>
#
#   <model_tag> is a key in configs/models.py (e.g. "phi", "llama", "qwen").
#   All parameters (HF model_id, output dir, layer, hidden_dim) are read from
#   that config — single source of truth, no risk of layer mismatch.
#
# Example:
#   ./pipeline.sh phi
#
# Output goes to $PHASE_H_DATA_ROOT/<MODELS[tag]['path']>/.
#
# Steps (no judge):
#   1. vLLM generation: 276 roles × 1200 rollouts → responses/<role>.jsonl
#   2. Activation extraction at layer L → activations/<role>.pt
#   3. Unfiltered role vectors (mean over all rollouts) → vectors/<role>.pt
#   4. Lu-style assistant axis (mean(default) − mean(roles)) → axis.pt
#   5. Default activation vector → default.pt
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_tag>" >&2
    echo "       <model_tag> must be a key in configs/models.py" >&2
    exit 2
fi

TAG="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

# Read all params from configs/models.py via _common.py (honors PHASE_H_MODELS
# and PHASE_H_DATA_ROOT env vars).  Validates the tag exists.
read MODEL_ID OUTPUT LAYER HIDDEN_DIM <<<"$(
    cd "$REPO_ROOT" && "$PYTHON" -c "
import sys
from _common import MODELS, model_dir
tag = '$TAG'
if tag not in MODELS:
    sys.exit(f'Unknown model tag {tag!r}; configured tags: {list(MODELS)}')
m = MODELS[tag]
print(m['model_id'], model_dir(tag), m['layer'], m['hidden_dim'])
"
)"

# Locate the assistant-axis library (must be pip-installed or on PYTHONPATH).
PIPELINE_DIR="$("$PYTHON" -c 'import os, assistant_axis; print(os.path.dirname(os.path.dirname(assistant_axis.__file__)) + "/pipeline")')"
ROLES_DIR="$("$PYTHON" -c 'import os, assistant_axis; print(os.path.dirname(os.path.dirname(assistant_axis.__file__)) + "/data/roles/instructions")')"
QUESTIONS_FILE="$("$PYTHON" -c 'import os, assistant_axis; print(os.path.dirname(os.path.dirname(assistant_axis.__file__)) + "/data/extraction_questions.jsonl")')"

mkdir -p "$OUTPUT"

echo "=== Pipeline: tag=$TAG ==="
echo "  model_id:   $MODEL_ID"
echo "  output:     $OUTPUT"
echo "  layer:      $LAYER"
echo "  hidden_dim: $HIDDEN_DIM"
echo ""

echo "=== Step 1/5: Generate responses (vLLM) ==="
"$PYTHON" "$PIPELINE_DIR/1_generate.py" \
    --model "$MODEL_ID" \
    --roles_dir "$ROLES_DIR" \
    --questions_file "$QUESTIONS_FILE" \
    --output_dir "$OUTPUT/responses" \
    --question_count 240 \
    --max_tokens 512

echo "=== Step 2/5: Extract activations at layer $LAYER ==="
"$PYTHON" "$PIPELINE_DIR/2_activations.py" \
    --model "$MODEL_ID" \
    --responses_dir "$OUTPUT/responses" \
    --output_dir "$OUTPUT/activations" \
    --layers "$LAYER" \
    --batch_size 32

echo "=== Step 3/5: Compute unfiltered role vectors (no judge) ==="
"$PYTHON" - "$OUTPUT/activations" "$OUTPUT/vectors" <<'PYEOF'
import sys
import torch
from pathlib import Path

act_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

n = 0
for af in sorted(act_dir.glob("*.pt")):
    role = af.stem
    activations = torch.load(af, map_location="cpu", weights_only=False)
    stacked = torch.stack([v.squeeze(0) if v.ndim > 1 else v for v in activations.values()])
    mean_vec = stacked.mean(dim=0)
    torch.save({"vector": mean_vec, "type": "mean", "role": role}, out_dir / f"{role}.pt")
    n += 1
print(f"Wrote {n} unfiltered role vectors → {out_dir}")
PYEOF

echo "=== Step 4/5: Compute Lu-style assistant axis ==="
# Inline (instead of upstream 5_axis.py) because upstream classifies vectors
# by the saved 'type' field: type='mean' → default, type='pos_3' → role.
# Our Step 3 saves every role with type='mean' (no judge → no pos_3 vectors),
# so upstream sees 276 defaults and 0 roles and errors out. We instead key off
# the literal role name 'default' to separate the two pools.
"$PYTHON" - "$OUTPUT/vectors" "$OUTPUT/axis.pt" <<'PYEOF'
import sys
import torch
from pathlib import Path

vec_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
default_vec = None
role_vecs = []
for vf in sorted(vec_dir.glob("*.pt")):
    data = torch.load(vf, map_location="cpu", weights_only=False)
    v = data["vector"] if isinstance(data, dict) else data
    v = v.squeeze() if v.ndim > 1 else v
    if vf.stem == "default":
        default_vec = v
    else:
        role_vecs.append(v)
if default_vec is None:
    sys.exit("FATAL: no vectors/default.pt found")
if not role_vecs:
    sys.exit("FATAL: no role vectors found")
axis = default_vec - torch.stack(role_vecs).mean(dim=0)
torch.save(axis, out_path)
print(f"Wrote axis {tuple(axis.shape)}, norm={axis.norm().item():.4f}, n_roles={len(role_vecs)} → {out_path}")
PYEOF

echo "=== Step 5/5: Save default activation vector ==="
"$PYTHON" - "$OUTPUT/activations/default.pt" "$OUTPUT/default.pt" <<'PYEOF'
import sys
import torch
from pathlib import Path

act_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
activations = torch.load(act_path, map_location="cpu", weights_only=False)
default_vec = torch.stack([v.squeeze(0) if v.ndim > 1 else v for v in activations.values()]).mean(dim=0)
torch.save(default_vec, out_path)
print(f"Wrote default vector {tuple(default_vec.shape)} → {out_path}")
PYEOF

echo ""
echo "=== Done. Outputs in $OUTPUT ==="
echo "  responses/    role response JSONL files"
echo "  activations/  per-role activation dicts"
echo "  vectors/      per-role unfiltered mean vectors"
echo "  axis.pt       Lu-style assistant axis"
echo "  default.pt    default activation vector"
