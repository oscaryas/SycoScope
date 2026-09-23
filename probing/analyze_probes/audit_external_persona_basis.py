#!/usr/bin/env python3
"""Fetch and audit a pinned external persona basis; never execute its source.

This audit deliberately does not project cached activations when the basis's
extraction-layer mapping is not established. Original data remain untouched.
"""
import hashlib
import io
import json
import sys
from pathlib import Path
import urllib.request

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.utils import common


REVISION = "98ae1b599e8ed1ae6952d83167ea4e4ba42111ce"
BASIS_SHA = "327d15905d1840ff4c1501e8522d3b9948282fc69f8aeb02d2b1b7a5fe5eb09f"
STEER_REV = "c8f975e88b4b4c366befaa59342c4405d7705def"
PRIOR_REV = "b94fd93ff532cd51edf93c7b7a3e10c1b526d086"
BASE = f"https://huggingface.co/datasets/pandaman007/steered-persona-space/resolve/{REVISION}/"


def validate_basis(basis):
    p = np.column_stack([basis[f"pc{i}"] for i in (1, 2, 3)]).astype(np.float64)
    mean = np.asarray(basis["role_mean"], dtype=np.float64)
    if p.shape != (4096, 3) or mean.shape != (4096,):
        raise ValueError("Unexpected basis dimensions")
    if not np.isfinite(p).all() or not np.isfinite(mean).all():
        raise ValueError("Nonfinite basis")
    np.testing.assert_allclose(p.T @ p, np.eye(3), atol=1e-6)
    return p, mean


def main():
    out = common.RESULTS_DIR / "llama31_5k_subset/analysis/external_persona_basis_audit"
    out.mkdir(parents=True, exist_ok=True)
    sources = {
        "llama8b_basis.pt": BASE + "pca_basis/llama8b_basis.pt",
        "dataset_README.md": BASE + "README.md",
        "dataset_models.py.txt": BASE + "configs/models.py",
        "compute_baseline_pca.py.txt": f"https://raw.githubusercontent.com/pauljCherian/steered-persona-space/{STEER_REV}/scripts/compute_baseline_pca.py",
        "baseline_loader.py.txt": f"https://raw.githubusercontent.com/pauljCherian/steered-persona-space/{STEER_REV}/_common.py",
        "prior_pipeline.sh.txt": f"https://raw.githubusercontent.com/pauljCherian/persona-space-comparison/{PRIOR_REV}/pipeline.sh",
        "prior_models.py.txt": f"https://raw.githubusercontent.com/pauljCherian/persona-space-comparison/{PRIOR_REV}/configs/models.py",
    }
    inventory = {}
    for name, url in sources.items():
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if name == "llama8b_basis.pt" and digest != BASIS_SHA:
            raise ValueError("Pinned PCA artifact hash mismatch")
        target = out / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f"Refuse to overwrite different audit input: {target}")
        target.write_bytes(data)
        inventory[name] = {"url": url, "sha256": digest, "bytes": len(data)}

    # The pinned artifact contains ordinary NumPy arrays. Allow only those
    # constructors with PyTorch's restricted loader, never weights_only=False.
    with torch.serialization.safe_globals([
        np._core.multiarray._reconstruct, np.ndarray, np.dtype,
        np.dtypes.Float32DType, np.dtypes.Float64DType,
    ]):
        basis = torch.load(io.BytesIO((out / "llama8b_basis.pt").read_bytes()),
                           map_location="cpu", weights_only=True)
    p, mean = validate_basis(basis)
    np.savez_compressed(out / "basis_numeric_only.npz", components=p, role_mean=mean)
    result = {
        "status": "blocked_unverified_extraction_layer",
        "projection_analysis_run": False,
        "model_id_documented": "meta-llama/Llama-3.1-8B-Instruct",
        "checkpoint_revision_documented": None,
        "source_layer_index": None,
        "compatible_local_activation_key": None,
        "source_pooling_documented": "mean over assistant response tokens",
        "basis_shape": list(p.shape),
        "orthonormality_max_abs_error": float(np.max(np.abs(p.T @ p - np.eye(3)))),
        "role_count": int(basis["n_roles_used"]),
        "role_variance_fraction_per_pc": list(basis["var_frac_top3"]),
        "role_variance_fraction_top3": float(basis["cum_var_top3"]),
        "pc1_cosine_with_source_default_minus_role_axis": float(basis["cos_pc1_lu"]),
        "basis_source_label": basis["source"],
        "sources": inventory,
        "blockers": [
            "The basis artifact has no extraction layer or model revision metadata.",
            "The layer=16 in the steered-project config describes its new runner; its baseline loader reads precomputed vectors without selecting or checking a layer.",
            "The public prior-project configuration covers different model sizes and does not contain the Llama-3.1-8B run configuration.",
        ],
        "method_limitations": [
            "Only three PCs are supplied, capturing 51.45% of source role-vector variance; not a full persona-space basis.",
            "The public prior pipeline omits the role-expression judge/filter, unlike Lu et al.; the exact Llama-8B run logs are unavailable in the audited materials.",
            "The supplied PCA basis was selected by its authors for stronger PC1 alignment with the Assistant Axis; it is not a neutral estimate of every aspect of personality.",
            "The public alpha=0 role vectors in the steering archive use last-token pooling, so they are not replacements for the response-mean role vectors that produced this basis.",
        ],
    }
    (out / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    report = """# External Lu-style persona basis audit

**Status: numerically valid, but extraction-layer provenance is unresolved. No projected variance result has been computed.**

## Confirmed

- Public artifact targets `meta-llama/Llama-3.1-8B-Instruct` according to its model configuration.
- Pinned dataset revision: `98ae1b599e8ed1ae6952d83167ea4e4ba42111ce`.
- PCA artifact SHA256: `327d15905d1840ff4c1501e8522d3b9948282fc69f8aeb02d2b1b7a5fe5eb09f`.
- Three finite orthonormal 4096-dimensional PCs; role-mean vector also has 4096 dimensions.
- 275 source roles. Variance fractions: PC1 27.58%, PC2 16.83%, PC3 7.04%; total 51.45% **of source role-vector variance**, not of our response variance.
- Artifact metadata and PCA construction source specify mean-over-assistant-token extraction. This is the appropriate pooling family for our response-mean analysis.
- PC1 cosine with the source default-minus-role axis is 0.818; PC1 is not identical to that axis.
- Loaded using PyTorch's restricted weights-only loader with standard NumPy array constructors explicitly allowed. External source files were read, not executed. A numeric-only NPZ is retained for future analysis.

## Why the analysis is paused

The artifact does not record the layer at which its source role vectors were extracted. The new steering runner's configuration says layer 16, but the PCA builder reads vectors from a separate prior project without selecting or checking a layer. The available prior-project configuration does not include this Llama-3.1-8B run. Thus we cannot distinguish `hidden_states[16]` (our block 15, not cached) from block 16 output (our `response_L16`), or independently establish the original extraction depth. Dimensionality alone cannot settle this.

The surrounding project documents an off-by-one steering-layer error. Its public alpha=0 role-vector cloud also uses last-token extraction, not the response-mean extraction used for this basis. Neither resolves the source-layer ambiguity.

The prior pipeline explicitly omits the role-expression judge/filter. This is an independent **Lu-style adaptation**, not a verified exact replication of Lu et al. The original model commit, actual Llama-8B extraction configuration, and role-filtering logs are not established by the audited materials.

## Required next step

Obtain the author's extraction command/log specifying model revision, zero-based block index (or `hidden_states` index), response-token span and role filtering; ideally obtain the original response-mean role vectors. Alternatively, construct a new independently elicited Lu-style basis on our model with documented extraction settings. Do not choose a layer by whichever produces the strongest system-prompt variance.

After matching the layer, use the frozen 400-question × 40-system-condition cohort. Project responses and repeat the system/question/interaction decomposition, reporting both variance shares inside the three-PC subspace and the fraction of full activation variance captured. Include random three-dimensional subspaces and question-bootstrap uncertainty. Do not reuse a single middle-layer basis as if it were a validated basis for all layers.

## Sources

- [Pinned dataset card](https://huggingface.co/datasets/pandaman007/steered-persona-space/blob/98ae1b599e8ed1ae6952d83167ea4e4ba42111ce/README.md)
- [PCA construction](https://github.com/pauljCherian/steered-persona-space/blob/c8f975e88b4b4c366befaa59342c4405d7705def/scripts/compute_baseline_pca.py)
- [Baseline loader](https://github.com/pauljCherian/steered-persona-space/blob/c8f975e88b4b4c366befaa59342c4405d7705def/_common.py)
- [Prior extraction pipeline](https://github.com/pauljCherian/persona-space-comparison/blob/b94fd93ff532cd51edf93c7b7a3e10c1b526d086/pipeline.sh)

The original activation-variance results and the running GPU experiment were not changed.
"""
    (out / "AUDIT.md").write_text(report)
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}, indent=2))
    print(f"Saved audit: {out / 'AUDIT.md'}")


if __name__ == "__main__":
    main()
