# File inventory for the data/results/utils/evaluations/analyze_probes/probe restructure

Proposed destination for every file in `prompt_probes/pipeline/` (the
`prompt_probes` method) and the baseline-probing set inside `tool_calling/`
(the `baseline_probes` method), per `REFACTOR_STRUCTURE_UNDERSTANDING.md`.

**RESOLVED naming conflict:** the six directories are nested under one new
top-level directory, `probing/`, rather than being true repo-root siblings —
`probing/data/`, `probing/results/`, `probing/utils/`, `probing/evaluations/`,
`probing/analyze_probes/`, `probing/probe/`. This avoids a name collision
with the existing repo-root `utils/` (`inference.py`, `model.py`, etc.),
which is untouched and stays exactly where it is. Every destination in the
tables below is implicitly prefixed with `probing/` (e.g. "`utils/`" in a
table cell means `probing/utils/`, never the existing repo-root one).

## `prompt_probes/pipeline/` → new structure

| File | Destination | Why |
|---|---|---|
| `common.py` | `utils/` | Cell/slug bookkeeping, run-metadata helpers shared by every stage |
| `fetch_user_prompts.py` | `data/` | Builds the shared Perez/Dolly user-prompt pool |
| `dataset_control.py` | `data/` | Matched-backend Perez/Dolly dataset-condition control |
| `prompt_wording_control.py` | `data/` | Paraphrase-control dataset variant |
| `build_llama31_subset.py` | `data/` | Builds a specific run's dataset subset |
| `generate_response.py` | `evaluations/prompt_probes/generation/` | Stage 1: contrastive-prompt generation |
| `generate_response_openrouter.py` | `evaluations/prompt_probes/generation/` | OpenRouter variant of the above |
| `generate_syconbench_gpu.py` | `evaluations/prompt_probes/generation/` | SyconBench GPU generation |
| `finish_syconbench_vast.py` | `evaluations/prompt_probes/generation/` | Vast.ai remote-generation follow-up |
| `get_activations.py` | `probe/` | Stage 2: activation caching that directly feeds probe training |
| `train_probes.py` | `probe/` | Stage 3: probe training (the method's namesake) |
| `train_dim.py` | `probe/` | DIM training, confirmed a third probing method — "non-parametric counterpart to train_probes.py" |
| `judge_behaviors.py` | `evaluations/prompt_probes/judge/` | Blinded behavior audit judge |
| `judge_syconbench_budgeted.py` | `evaluations/prompt_probes/judge/` | Budgeted Anthropic SYCON judging |
| `analyze_probes.py` | `analyze_probes/` | Stage 4: cross-probe analysis (exact name match) |
| `eval_common.py` | `analyze_probes/` | Shared "cache activations, score vs. probe" machinery used by every `eval_*.py` |
| `eval_are_you_sure.py` | `analyze_probes/` | OOD eval — are_you_sure |
| `eval_elephant.py` → rename `eval_social_sycophancy.py` | `analyze_probes/` | Already social-sycophancy-only (confirmed earlier); rename to match |
| `eval_moral.py` | `analyze_probes/` | OOD eval — moral |
| `eval_sypr.py` | `analyze_probes/` | OOD eval — praise/SyPR |
| `eval_syconbench.py` | `analyze_probes/` | OOD eval — syconbench |
| `eval_cross_cell.py` | `analyze_probes/` | Cross-cell transfer analysis |
| `eval_specificity.py` | `analyze_probes/` | Specificity-vs-benign-lookalikes analysis |
| `eval_matrix.py` | `analyze_probes/` | Probe-by-eval-target performance matrix |
| `summarize_syconbench.py` | `analyze_probes/` | SyconBench cluster/report summary |
| `cluster_syconbench_stability.py` | `analyze_probes/` | Bootstrap stability analysis |
| `activation_variance.py` | `analyze_probes/` | Descriptive activation-variance decomposition |
| `build_probe_html_report.py` + `probe_report_assets/` | `analyze_probes/` | Visualization/report output (the doc explicitly puts visualization here) |
| `audit_disagreements.py` | `analyze_probes/` | Probe-disagreement audit |
| `audit_external_persona_basis.py` | `analyze_probes/` | External-basis audit |
| `run_c_sweep.sh`, `run_c_sweep_eval.sh` | `probe/` + `analyze_probes/` (split) | Orchestration wrappers — the training-sweep half goes with `probe/`, the eval half with `analyze_probes/`; may need splitting into two scripts |
| `*.supervisor.conf` (dataset_control, prompt_wording_control, syconbench_extraction, syconbench_generation) | co-located with the script each configures | Operational config, not a separate concern |

## `tool_calling/` baseline-probing set → new structure

| File/dir | Destination | Why |
|---|---|---|
| `sycophancy_probes.py` | `probe/` | Core linear-probe class/module |
| `pipeline_scripts/probing/train_probes.py` | `probe/` | Baseline-probe training entrypoint |
| `pipeline_scripts/training.py` | `probe/` | Shared training core (cross-validation, holdout, etc.) |
| `scripts/oeq_probe_pipeline.py`, `sypr_probe_pipeline.py`, `mixed_probe_pipeline.py`, `category_residual_probe_pipeline.py`, `mixture_residual_probe_pipeline*.py`, `mixture_category_residual_probe_pipeline.py`, `moral_avg_residual_probe_pipeline.py` | `probe/` | End-to-end probe-training pipelines per dataset/variant |
| `pipeline_scripts/cache.py`, `common.py`, `datasets.py` | `utils/` | Shared activation-cache/dataset-prep mechanics |
| `pipeline_scripts/activations/extract_activations.py` | `utils/` | Activation extraction |
| `gpu_memory.py` | `utils/` | GPU memory management |
| `sycophancy_data.py`, `sypr_data.py` | `data/` | Dataset loaders/builders (TruthfulQA-sycophancy, SyPR) |
| `pipeline_scripts/generations/*.py` | `evaluations/baseline_probes/generation/` | Generation entrypoints per dataset |
| `scripts/*_generate.py` (are_you_sure_mc/freeform, moral, social, sypr_praise_full, truthfulqa_sycophancyeval, colab_multimodel, dissociating_sycophancy, syconbench_fetch_generations) | `evaluations/baseline_probes/generation/` | Generation scripts (incl. those absorbed from `worktree-fix-steerer-asymmetries`) |
| `pipeline_scripts/judges/*.py` | `evaluations/baseline_probes/judge/` | Judge entrypoints per dataset |
| `are_you_sure_correctness_judge.py`, `moral_sycophancy_judge.py`, `social_sycophancy_judge.py`, `sycophantic_praise_judge.py`, `truthfulqa_verdict_judge.py`, `incremental_judge_moral_flip.py`, `incremental_judge_social.py` | `evaluations/baseline_probes/judge/` | Judge implementations |
| `pipeline_scripts/visualizations/*.py`, `scripts/plot_all_probes.py`, `scripts/plot_probe_transfer_summary.py` | `analyze_probes/` | Visualization (doc: analyze_probes owns all visualization) |
| `oeq_probe_aita_transfer_pipeline.py`, `truthfulqa_probe_transfer_pipeline.py`, `mixture_holdout_eval_pipeline.py`, `rq1_gate1_geometry.py`, `sycophancy_compare.py` | `analyze_probes/` | Cross-domain transfer / held-out eval / geometry / paper-comparison analysis |
| `pipeline_scripts/tests/test_pipeline.py` | co-located with whatever it tests, split as needed | Test coverage follows the code it tests |

## DIM (difference-in-means) — CONFIRMED a third probing method

DIM is extracted alongside `prompt_probes` and `baseline_probes` as a third
method. `results/` and `evaluations/` gain a `dim/` sibling wherever the
method-split happens, per the same shape as the other two:
`results/{baseline_probing,prompt_probing,dim}/...`.

DIM training pipelines (`aita_dim_pipeline.py`, `mixture_dim_pipeline.py`)
source from datasets that `evaluations/baseline_probes/generation+judge/`
already produces (AITA-NTA-FLIP, sycophancy_mixture) — no evidence of any
DIM-specific generation/judge scripts exists, so **no `evaluations/dim/` is
needed**; DIM reuses `baseline_probes`' (or `prompt_probes`') existing
generation/judge output as its input data. DIM's own new territory is
`probe/` (training) and `analyze_probes/` (evaluation/analysis, including
`results/dim/`).

| File | Destination | Why |
|---|---|---|
| `prompt_probes/pipeline/train_dim.py` | `probe/` | DIM training (prompt_probes-style contrastive-pair sourcing) |
| `tool_calling/tasks/sycophancy/sycophancy_dim.py` | `probe/` | Core shared DIM direction-finding module |
| `tool_calling/tasks/sycophancy/scripts/aita_dim_pipeline.py` | `probe/` | End-to-end DIM pipeline sourced from AITA-NTA-FLIP judged data |
| `tool_calling/tasks/sycophancy/scripts/mixture_dim_pipeline.py` | `probe/` | End-to-end DIM pipeline sourced from sycophancy_mixture |
| `tool_calling/tasks/sycophancy/pipeline_scripts/difference_in_means/run_dim.py` | `probe/` | DIM training core |
| `tool_calling/tasks/sycophancy/scripts/mixture_dim_layer_alpha_probe.py` | `analyze_probes/` | Analyzes DIM projection + probe accuracy by layer/alpha — an analysis script, not training |
| `tool_calling/tasks/sycophancy/pipeline_scripts/difference_in_means/steering_eval.py` | `analyze_probes/` | Evaluates a DIM direction's steering effect — DIM's "measure on probe" equivalent |
| `tool_calling/tasks/sycophancy/cross_dataset_generalization.py` | `analyze_probes/` | Shared cross-domain-generalization sweep, used by both baseline probe pipelines and DIM pipelines — now a shared `analyze_probes/` utility rather than tool_calling-only |

## CONFIRMED (2026-09-22, cross-branch extraction planning) — major scope correction

Investigating the actual cross-branch extraction (Phase 3: pulling `baseline_probes`/`dim` out of `building-agent`'s `tool_calling/` into `main`'s `probing/`) surfaced a much bigger architectural fact than assumed above: **most of `tool_calling/tasks/sycophancy/scripts/*.py`'s end-to-end pipeline files mix pure probe/DIM training with live steering-based validation in the same script**, not just the 3 files originally flagged (`cross_dataset_generalization.py`, `difference_in_means/steering_eval.py`, `run_steered_judge_aita.py`). A full dependency sweep found **14 `scripts/*.py` files** import `sycophancy_steering`/`ActivationSteerer` directly, including generation scripts (`are_you_sure_freeform_generate.py`, `moral_generate.py`, `social_generate.py`, `sypr_praise_full_generate.py`, `truthfulqa_sycophancyeval_generate.py`, `colab_multimodel_generate.py`) and probe/DIM pipelines (`mixed_probe_pipeline.py`, `mixture_dim_layer_alpha_probe.py`, `mixture_dim_pipeline.py`, `mmlu_truthfulqa_pipeline.py`, `oeq_probe_aita_transfer_pipeline.py`, `sypr_probe_pipeline.py`, plus the shared `scripts/tools.py` helper).

**Confirmed resolution (both points confirmed with the user):**
1. `sycophancy_model_registry.py` (currently `tool_calling/tasks/sycophancy/`, zero internal dependencies — only stdlib + torch) moves to the **shared repo-root `utils/`** (e.g. `utils/model_registry.py`), since it's needed by both `building-agent`'s steering code AND the baseline-probe activation-extraction code moving to `main`. This one move is what makes the rest of the extraction possible — it's used far more widely than steering alone (also by every `*_generate.py` script and `sycophancy_probes.py`'s activation-collection functions).
2. Scripts that need a **live `ActivationSteerer`** during their own run (all 14 identified above) **stay on `building-agent`** as steering-integrated experiment drivers — they are not "probing" in the extractable sense, they're applications that *consume* a trained probe/DIM direction to steer generation.

**What this means for the actual shape of Phase 3:** it is not a straight "move directory, delete original" operation like the `prompt_probes` migration was. It is:
- **Extract a shared library** to `main`'s `probing/`: the parts with zero steering dependency — confirmed clean by re-sweeping every file: the *entire* `pipeline_scripts/` package (`cache.py`, `common.py`, `datasets.py`, `training.py`, `probing/train_probes.py`, `activations/extract_activations.py`, all of `generations/*.py`, all of `judges/*.py`, all of `visualizations/*.py` — genuinely zero steering references anywhere in this package) plus `sycophancy_probes.py` in full (its `collect_activations`/`collect_sentence_activations` only needed `sycophancy_model_registry`, not live steering — once that module is shared, these extract cleanly too) plus `sycophancy_data.py`/`sypr_data.py` (data loaders — `sypr_data.py` was flagged as depending on `sycophancy_model_registry` too, not steering, so it also becomes clean once that's shared) plus the non-steering `scripts/*.py` analysis files (`bootstrap_nc1_pipeline.py`, `rq1_gate1_geometry.py`, `mixture_holdout_eval_pipeline.py`, `sypr_praise_count.py`, `sycophancy_compare.py`, plus the plotting scripts `plot_all_probes.py`/`plot_probe_transfer_summary.py`).
- **Leave the 14 steering-integrated scripts on `building-agent`**, but they will need their own imports updated afterward to consume the newly-shared `probing.*`/`utils.model_registry` library instead of their current same-directory/package-relative references (a distinct, smaller follow-up task on `building-agent`, not part of extracting the library itself).
- **Naming collision to resolve:** `pipeline_scripts/common.py` and `pipeline_scripts/cache.py` would collide with `prompt_probes`'s already-shipped `probing/utils/common.py` if merged into the same flat `probing/utils/` directory (different content/purpose entirely — `DatasetSpec`/`parse_dataset_spec` vs. `PROBING_DIR`/`CELL_SLUGS`). **Decision:** give the baseline_probes-side files distinct names on merge rather than retroactively restructuring the already-merged `prompt_probes` side: `pipeline_scripts/common.py` → `probing/utils/baseline_probes_common.py`, `pipeline_scripts/cache.py` → `probing/utils/baseline_probes_cache.py`, `pipeline_scripts/datasets.py` → `probing/utils/baseline_probes_datasets.py`. `pipeline_scripts/training.py` has no name clash, stays `probing/probe/training.py` (or similar) as-is.

## Stays in `tool_calling/` on `building-agent` (NOT extracted — not probing, or steering-integrated)

- `sycophancy_steering.py` — steering infrastructure, already the subject of the separate steerer-generation-consolidation plan; `building-agent` is where this lives going forward.
- `run_steered_judge_aita.py`, `cross_dataset_generalization.py`, `pipeline_scripts/difference_in_means/steering_eval.py` — operate on **steered** generations/directions, not on probe/DIM evaluation directly.
- The 14 steering-integrated `scripts/*.py` files identified above (generation scripts with an optional steering mode, and probe/DIM pipelines with an integrated steering-validation step) — see the correction above.
- `colab_load_secrets.py` — Colab-notebook-specific environment plumbing, unrelated to probing.
