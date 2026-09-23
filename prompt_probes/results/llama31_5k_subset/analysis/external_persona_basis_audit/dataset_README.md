---
license: apache-2.0
task_categories:
  - feature-extraction
tags:
  - activation-steering
  - persona-vectors
  - interpretability
  - mechanistic-interpretability
  - llama
  - qwen
size_categories:
  - 1K<n<10K
---

# Steered Persona Space

*How activation steering reshapes a language model's character cloud.*

We apply activation-steering hooks at the canonical mid-layer (N/2) of three open chat models and measure how the model's **default-assistant point** — its neutral self-identity — moves through a baseline-derived principal-component space of 275 character role-playing activations.

![all-models default trajectory](headline_figures/all_models_default_pc1_pc2.png)

Each colored marker is a steered model's default-assistant point projected into the baseline model's PC1×PC2 plane. Marker size scales with α (steering strength). Gray dots are the baseline role cloud for spatial context. **The same neutral assistant prompt sends the model to systematically different places under different trait vectors, and barely moves under magnitude-matched random vectors.**

---

## Findings (3 bullets, each with its own figure)

### 1. Direction matters more than magnitude
![trait vs control](headline_figures/trait_vs_control_displacement.png)

At α=2 every trait vector shifts the persona cloud by ‖δ‖ ≈ 5–7 in 4096-dim residual space. `random_unit_s42` — a Gaussian vector rescaled to match the trait vectors' norm — sits at the same ‖α·v‖ but moves the cloud only a fraction as much. **Steering is direction-specific, not just magnitude-driven.**

### 2. Different traits send the default in distinguishable directions
![per-model trajectories](headline_figures/default_assistant_trajectory_llama8b.png)

Evil, sycophantic, hallucinating, and humorous each carve a distinct path through PC1×PC2×PC3 as α grows. The trajectories aren't collinear — meaning the four "persona vectors" don't share one master "steering direction." See per-model PNGs in `headline_figures/` for qwen7b and dolphin8b.

### 3. Clean dose-response within the coherent regime (α ≤ 2)
![dose-response](headline_figures/dose_response_curves.png)

Per-role ‖steered − α=0‖ grows monotonically with α for every (model, trait). At α=4 the model often falls off-manifold (incoherent degeneration) — see sample responses in `responses_sample/` for qualitative confirmation. **α=1–2 is the scientifically useful regime; α=4 is the manifold-edge stress test.**

---

## Experimental design

| Axis | Values |
|---|---|
| Models | llama8b · qwen7b · dolphin8b |
| Traits | evil · sycophantic · hallucinating · humorous |
| Controls | random_unit_s42 · random_unit_s43 · random_unit_s44 |
| α magnitudes | [0.5, 1.0, 2.0, 4.0, 8.0] (plus α=0 baseline) |
| Roles × prompts × questions | 276 · 5 · 16 = 22,080 generations per condition |

Per condition we save: the 277 per-role activation means (`role_vectors/{model}/{vector}/alpha_{a}/*.pt`), the per-condition default-assistant activation (`default.pt`), and a representative response (doctor) in JSONL.

Currently in this release: **57 complete conditions** (llama8b: 19, qwen7b: 19, dolphin8b: 19).

---

## Folder map

| Path | What's inside |
|---|---|
| `headline_figures/*.png` | 6 hero plots — start here |
| `summary/default_positions.csv` | One row per (model, vector, α) — PC coords + along/perp decomposition |
| `summary/displacement_table.csv` | Quantitative summary across all conditions |
| `summary/cosine_matrix.csv` | Pairwise cosines between every steering vector |
| `summary/trait_vector_stats.csv` | Per-model trait + control norms, hidden_dim, layer |
| `configs/` | `models.py` + `steering_vectors.py` + `alphas.py` from the source repo |
| `steering_vectors/{model}/{name}.pt` | The 30 steering directions (12 trait + 18 control) |
| `pca_basis/{model}_basis.pt` | Baseline-derived PC1/PC2/PC3 + role_mean per model |
| `projections/{model}/{vector}/alpha_{a}.npz` | Each condition's roles projected into baseline PC1-3 |
| `role_vectors/{model}/{vector}/alpha_{a}/*.pt` | Per-role activations (276 + default), the "persona clouds" |
| `responses_sample/{model}/{vector}/alpha_{a}/doctor.jsonl` | 1 role's 80 rollouts per condition for qualitative inspection |

**Deliberately excluded:** the raw per-rollout activations (~280 GB) and the full responses corpus (~50 GB). Both are reproducible from `steering_vectors/` + the source repo if you need them.

---

## Load + use

```python
# A. Load a steering vector
from huggingface_hub import hf_hub_download
import torch
p = hf_hub_download(repo_id="pandaman007/steered-persona-space", repo_type="dataset",
                    filename="steering_vectors/llama8b/evil.pt")
v_evil = torch.load(p, weights_only=False)["vector_at_canonical_layer"]   # (4096,)

# B. Load the PCA basis and project a custom activation
basis = torch.load(hf_hub_download("pandaman007/steered-persona-space", repo_type="dataset",
                                    filename="pca_basis/llama8b_basis.pt"), weights_only=False)
import numpy as np
P = np.stack([basis["pc1"], basis["pc2"], basis["pc3"]], axis=1)   # (4096, 3)
my_pc = (my_activation - basis["role_mean"]) @ P                    # (3,)

# C. Reproduce the hero figure from default_positions.csv
import pandas as pd, matplotlib.pyplot as plt
csv = hf_hub_download("pandaman007/steered-persona-space", repo_type="dataset",
                      filename="summary/default_positions.csv")
df = pd.read_csv(csv)
for (model, vec), g in df.groupby(["model", "vector"]):
    plt.plot(g["pc1"], g["pc2"], marker="o", label=f"{model}/{vec}")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(fontsize=6); plt.show()
```

---

## Caveats (read these)

These are real and worth understanding before drawing strong quantitative conclusions:

1. **Off-by-one layer.** The persona-vectors paper builds `v` from `hidden_states[L]` (= output of `model.model.layers[L-1]`) and applies it back at the same layer. Our runner hooks `model.model.layers[L]` — one layer downstream of the build point. Empirically `cos(h[L], h[L+1]) ≈ 0.88` for these traits, so the applied vector is ~88 % aligned with what was intended. Qualitative findings (direction-specificity, dose-response, trajectory separability) are robust; quantitative magnitudes are deflated ~12 % from the design intent.

2. **Extraction convention.** Our per-role mean vectors are computed at the **last token** of the full `(system, user, assistant)` sequence. The prior project (and the upstream `assistant_axis` library) uses **mean over assistant-turn tokens**. We use the HF α=0 condition (same last-token convention) as the comparison baseline throughout, so within-pipeline deltas are clean. The shipped `pca_basis/` was built on the prior convention — interpretable as "how does our steering displacement align with the variance directions of a related cloud" rather than "the natural PCs of our manifold."

3. **Controls.** `random_unit_*` controls are Gaussian vectors rescaled to match the mean trait-vector norm — the magnitude-matched null direction. Three independent seeds (s42/s43/s44) per model for statistical robustness. `partition_anchor_*` vectors also exist in `steering_vectors/` for the curious but were empirically uninformative for this experiment and are not featured in the headline analyses.

---

## Reproducibility

Source code: see the `configs/` folder for the exact model IDs, layer indices, trait list, alpha grid, and control seeds used. The full pipeline (build steering vectors → run steered axis → project into baseline PC → render figures) is reproducible with `scripts/build_vectors.py`, `scripts/build_controls.py`, `scripts/run_steered.py`, `scripts/project_into_baseline.py`, `scripts/plot_steered_clouds.py` in the source repository.

## Citation

This dataset operationalizes ideas from two recent papers:

- *Persona Vectors: Monitoring and Controlling Character Traits in Language Models* — safety-research/persona_vectors
- *The Assistant Axis: Measuring and Controlling Persona Structure in LLMs* — safety-research/assistant-axis
