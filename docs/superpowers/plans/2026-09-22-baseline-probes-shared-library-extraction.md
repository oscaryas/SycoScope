# Shared Probing Extraction and Three-Branch Split Implementation Plan

> **Execution rule:** complete this plan in order. Do not delete an original file until its replacement, rewritten consumers, and migrated tests pass. Use the repository's normal safe-editing workflow; do not stream `git show` directly into a destination path because a failed read can leave an empty file.

**Goal:** Extract reusable baseline-probing, DIM, dataset, generation, judging, and analysis code from `tool_calling/` into `main`'s shared `probing/` and `utils/` packages, integrate that shared history into both application branches, and then finish the branch split:

- `main`: shared probing/evaluation code; no `SAE/` and no `tool_calling/`.
- `SAE`: shared code plus `SAE/`; no `tool_calling/`.
- `building-agent`: shared code plus `tool_calling/`; no `SAE/`.

Generated JSONL, compressed generation checkpoints, and trained checkpoints remain versioned on the branch that owns them. Activation arrays and disposable rendered reports remain ignored.

## Why this revision is necessary

The previous draft was not executable safely. It attempted to check out `main` in two worktrees, independently added the same shared module on two branches, merged a whole main-derived branch into `building-agent` without a deletion guard, omitted `pipeline_scripts/activations/extract_activations.py`, overwrote the existing prompt-probe `probing/probe/train_probes.py`, updated only a subset of consumers, considered test migration after deleting the tests, and did not update runtime data/result paths after moving scripts.

This revision uses a shared-first, delete-last sequence. The only whole-branch integrations happen **before** any application directory is deleted and are protected by path-diff gates. After the split, shared changes flow to application branches as individual commits or explicit commit ranges; never merge a post-cleanup `main` wholesale into an application branch.

## Authoritative branch sequence

```text
legacy/pre-refactor
├── main ── prompt migration ── shared extraction ── remove SAE/ + tool_calling/
├── SAE  ────────────────────── integrate shared ─── remove tool_calling/
└── building-agent ─ steerer ── integrate shared ─── rewrite consumers ── remove SAE/
```

The existing `legacy/pre-refactor` tag is the immutable recovery point. The current branch tips are not assumed to be clean; record their actual hashes before starting.

## Global constraints

- Treat `PROBE_RESTRUCTURE_FILE_INVENTORY.md` as discovery input, not executable truth. This plan is authoritative where the inventory's early tables conflict with its later steering-dependency findings.
- `main` currently still contains both application directories. Extraction must finish and be integrated into `SAE` and `building-agent` before any branch removes those directories.
- Work on a new `baseline-probes-extraction` branch created from `main`; do not attach `main` to a second worktree.
- Do not modify `SAE/` or `tool_calling/` on the extraction branch. Source content is read from the `building-agent` worktree or `git show building-agent:<path>`.
- Do not overwrite the existing prompt-probe files in `probing/probe/`. Baseline-probe modules receive explicit names (`baseline_probes.py`, `baseline_training.py`, `train_baseline_probes.py`, `extract_baseline_activations.py`).
- Shared code must not import from `tool_calling`, `SAE`, `pipeline_scripts`, `sycophancy_model_registry`, bare `sycophancy_probes`, bare `sycophancy_data`, or bare `sypr_data`.
- A moved command must have its `Path(__file__)` calculations, default input/output paths, subprocess paths, and help-text examples reviewed. Merely fixing Python imports is insufficient.
- Prefer explicit CLI paths when an input remains application-owned. A shared module must not silently default to a path under an application directory that will disappear from `main`.
- The five raw ELEPHANT CSVs are shared inputs, not SAE implementation code. Extract `SAE/datasets/*.csv` to `probing/data/source/elephant/` and update `utils/datasets.py` before removing `SAE/` anywhere. Generated SAE results do **not** move with them.
- `tool_calling/tasks/sae/` and `utils/sae_utils.py` are SAE-specific hybrids despite living outside `SAE/`. Preserve them on the `SAE` branch as `SAE/agent_tools/` and `SAE/sae/utils.py`, respectively; remove them from `main` and `building-agent` during branch cleanup.
- Capture the actual test baseline independently in every worktree. Do not hard-code historic pass counts as acceptance criteria; require no new failures and document intentional skips.
- Every destructive cleanup task has a manifest and zero-reference gate immediately before `git rm`.

## Destination map

### Shared infrastructure and data

| Source on `building-agent` | Destination on `main` |
|---|---|
| `tool_calling/tasks/sycophancy/sycophancy_model_registry.py` | `utils/model_registry.py` |
| `tool_calling/tasks/sycophancy/gpu_memory.py` | `probing/utils/gpu_memory.py` |
| `pipeline_scripts/common.py` | `probing/utils/baseline_probes_common.py` |
| `pipeline_scripts/cache.py` | `probing/utils/baseline_probes_cache.py` |
| `pipeline_scripts/datasets.py` | `probing/utils/baseline_probes_datasets.py` |
| `sycophancy_data.py` | `probing/data/sycophancy_data.py` |
| Pure loading/sampling/prompt-building functions from `sypr_data.py` | `probing/data/sypr_data.py` |
| `scripts/build_sycophancy_mixture.py` | `probing/data/build_sycophancy_mixture.py` |
| `SAE/datasets/*.csv` (five tracked raw datasets) | `probing/data/source/elephant/*.csv` |
| `utils/datasets.py` | stays at `utils/datasets.py`, but `DATASETS_DIR` changes to the shared source directory |

`sypr_data.py` is a required split, not a whole-file move. Its `generate_and_label_sypr(...)` function imports and instantiates `ActivationSteerer`, so that function remains application-owned as `tool_calling/tasks/sycophancy/sypr_generation.py`. The shared `probing/data/sypr_data.py` contains only dataset loading, filtering, sampling, row parsing, and chat-message construction. Both halves import shared helpers explicitly; neither duplicates the other's implementation.

### Probe and DIM code

| Source on `building-agent` | Destination on `main` |
|---|---|
| `pipeline_scripts/activations/extract_activations.py` | `probing/probe/extract_baseline_activations.py` |
| `pipeline_scripts/training.py` | `probing/probe/baseline_training.py` |
| `pipeline_scripts/probing/train_probes.py` | `probing/probe/train_baseline_probes.py` |
| `sycophancy_probes.py` | `probing/probe/baseline_probes.py` |
| `sycophancy_dim.py` | `probing/probe/dim.py` |
| `scripts/aita_dim_pipeline.py` | `probing/probe/aita_dim_pipeline.py` |
| `scripts/category_residual_probe_pipeline.py` | `probing/probe/category_residual_probe_pipeline.py` |
| `scripts/mixture_category_residual_probe_pipeline.py` | `probing/probe/mixture_category_residual_probe_pipeline.py` |
| `scripts/mixture_residual_probe_pipeline.py` | `probing/probe/mixture_residual_probe_pipeline.py` |
| `scripts/mixture_residual_probe_pipeline_v2.py` | `probing/probe/mixture_residual_probe_pipeline_v2.py` |
| `scripts/moral_avg_residual_probe_pipeline.py` | `probing/probe/moral_avg_residual_probe_pipeline.py` |
| `scripts/oeq_probe_pipeline.py` | `probing/probe/oeq_probe_pipeline.py` |

`pipeline_scripts/difference_in_means/run_dim.py` remains on `building-agent` for now because it conditionally invokes the live-steering evaluator. It moves with `steering_eval.py` to `tool_calling/tasks/sycophancy/difference_in_means/` during cleanup and imports its pure math/data dependencies from `probing.*`. A later focused refactor may split its pure training core into `probing/probe/dim.py`; do not duplicate that code during this migration.

### Baseline evaluation code

Move all six `pipeline_scripts/generations/*.py` modules into `probing/evaluations/baseline_probes/generation/`:

- `common.py`
- `generate_are_you_sure.py`
- `generate_sae.py`
- `generate_syconbench.py`
- `generate_sypr.py`
- `generate_truthfulqa.py`

Also move these steering-free generation utilities there:

- `scripts/dissociating_sycophancy_generate.py`
- `scripts/moral_generate_openrouter.py`
- `scripts/social_generate_openrouter.py`
- `scripts/syconbench_fetch_generations.py`
- `scripts/verify_model_generation_setup.py`

Move all seven `pipeline_scripts/judges/*.py` modules into `probing/evaluations/baseline_probes/judge/`, plus:

- `are_you_sure_correctness_judge.py`
- `moral_sycophancy_judge.py`
- `social_sycophancy_judge.py`
- `sycophantic_praise_judge.py`
- `truthfulqa_verdict_judge.py`
- `scripts/incremental_judge_moral_flip.py`
- `scripts/incremental_judge_social.py`
- `scripts/run_moral_sycophancy_judge_aita.py`
- `scripts/run_social_sycophancy_judge_oeq.py`

### Shared analysis code

Move into `probing/analyze_probes/`:

- `pipeline_scripts/visualizations/plot_judge_rates.py`
- `pipeline_scripts/visualizations/plot_probe_accuracy.py`
- `pipeline_scripts/visualizations/plot_steering_curve.py`
- `scripts/bootstrap_nc1_pipeline.py`
- `scripts/mixture_holdout_eval_pipeline.py`
- `scripts/plot_all_probes.py`
- `scripts/plot_category_nc1.py`
- `scripts/plot_nc1_bootstrap_ci.py`
- `scripts/plot_probe_transfer_summary.py`
- `scripts/plot_sycophancy_results.py`
- `scripts/rq1_gate1_geometry.py`
- `scripts/truthfulqa_probe_transfer_pipeline.py`
- `sycophancy_compare.py`

### Documentation and tests

| Source | Destination |
|---|---|
| `pipeline_scripts/README.md` | `docs/baseline_probes/README.md` |
| `pipeline_scripts/IMPLEMENTATION_RECAP.md` | `docs/baseline_probes/IMPLEMENTATION_RECAP.md` |
| `pipeline_scripts/PSYCHOMETRIC_SYCOPHANCY_OSF.md` | `docs/baseline_probes/PSYCHOMETRIC_SYCOPHANCY_OSF.md` |
| `pipeline_scripts/tests/test_pipeline.py` | `tests/test_baseline_probe_pipeline.py` |

### Files that remain application-owned on `building-agent`

Keep these in `tool_calling/` and rewrite their imports to shared modules:

- `sycophancy_steering.py`, `run_steered_judge_aita.py`, `cross_dataset_generalization.py`
- `sypr_generation.py` (the `ActivationSteerer`-dependent generation function split out of `sypr_data.py`)
- `pipeline_scripts/difference_in_means/{run_dim.py,steering_eval.py}` (relocated as described above)
- `scripts/are_you_sure_freeform_generate.py`
- `scripts/are_you_sure_mc_generate.py`
- `scripts/colab_multimodel_generate.py`
- `scripts/mixed_probe_pipeline.py`
- `scripts/mixture_dim_layer_alpha_probe.py`
- `scripts/mixture_dim_pipeline.py`
- `scripts/mmlu_truthfulqa_pipeline.py`
- `scripts/moral_generate.py`
- `scripts/oeq_probe_aita_transfer_pipeline.py`
- `scripts/social_generate.py`
- `scripts/sypr_praise_full_generate.py`
- `scripts/sypr_probe_pipeline.py`
- `scripts/sypr_praise_count.py` (transitively uses `ActivationSteerer`-backed generation through `generate_and_label_sypr`)
- `scripts/truthfulqa_sycophancyeval_generate.py`
- `scripts/tools.py`
- `scripts/colab_load_secrets.py`
- notebooks, skills, application tests, application results, generated JSONL, and checkpoints

The list above describes ownership, not the consumer-rewrite scope. **Every** remaining Python consumer discovered by the repository-wide scan must be updated, including non-steering files that remain in `tool_calling/` for operational reasons.

---

## Task 0: Safety snapshot and worktrees

- [ ] Record `git status --short --branch`, `git rev-parse` for `main`, `refs/heads/SAE`, and `building-agent`, plus `git worktree list --porcelain` in the execution ledger.
- [ ] Commit or otherwise safely preserve the revised plan, inventory, and structure document before beginning. Do not hide unrelated work in a broad stash.
- [ ] Confirm `legacy/pre-refactor` resolves and do not move or recreate it.
- [ ] Reuse `.claude/worktrees/building-agent` if clean and at `building-agent`'s tip.
- [ ] Create the extraction worktree with a **new branch**, not `main`:

```bash
git worktree add -b baseline-probes-extraction .claude/worktrees/baseline-probes-extraction main
```

- [ ] Create/reuse an SAE worktree using the unambiguous branch ref if needed:

```bash
git worktree add .claude/worktrees/sae refs/heads/SAE
```

- [ ] In each worktree, install/sync dependencies only if required, run the existing test suite, and record the observed baseline.

## Task 1: Freeze source, destination, and consumer manifests

Run this task before copying anything.

- [ ] On `building-agent`, enumerate every source path in the destination tables and verify it exists with `git cat-file -e building-agent:<path>`.
- [ ] Enumerate the full old package before cleanup:

```bash
git ls-tree -r --name-only building-agent:tool_calling/tasks/sycophancy/pipeline_scripts
```

- [ ] Capture every Python consumer across the whole branch, not only `tool_calling/`:

```bash
git grep -n -E 'pipeline_scripts\.|sycophancy_model_registry|(^|[[:space:]])(from|import)[[:space:]]+(sycophancy_probes|sycophancy_data|sypr_data|sycophancy_dim)([[:space:].]|$)' building-agent -- '*.py'
```

- [ ] Save the resulting path list in the execution ledger. It becomes the mandatory rewrite checklist for Task 8.
- [ ] Capture cross-application path dependencies before the split. Classify every executable match, not merely documentation:

```bash
git grep -n -E 'SAE/|utils\.sae_utils|tool_calling/tasks/sae' building-agent -- '*.py' '*.sh' '*.md' ':!*.ipynb'
```

- [ ] Confirm the known dependencies are represented in later tasks: `utils/datasets.py` and four sycophancy generation scripts need the shared raw CSV directory; sycophancy judges/steering scripts need non-SAE defaults or required CLI paths; `tool_calling/tasks/sae/` and `utils/sae_utils.py` move only to the `SAE` branch.
- [ ] On the extraction branch, verify that no destination already exists. The one expected collision is the prompt-probe `probing/probe/train_probes.py`; this plan deliberately uses `train_baseline_probes.py` instead.
- [ ] Establish a deletion guard: from the merge base, current `main` must not contain post-snapshot changes under `SAE/` or `tool_calling/`. If this prints any path, stop and reassess before integrating branches:

```bash
git diff --name-only "$(git merge-base main building-agent)"..main -- SAE tool_calling
```

## Task 2: Extract shared infrastructure and data onto `baseline-probes-extraction`

- [ ] Create `utils/model_registry.py`, the four baseline utility modules, and the data modules exactly as listed in the destination map. Copy from the `building-agent` blobs without modifying the source branch.
- [ ] Extract the five tracked raw CSVs from `SAE/datasets/` to `probing/data/source/elephant/` and change `utils/datasets.py::DATASETS_DIR` to that shared location. Update its docstrings and tests. Do not move anything from `SAE/results/`.
- [ ] Split `sypr_data.py`: put only pure dataset/prompt helpers in `probing/data/sypr_data.py`. Record the exact `generate_and_label_sypr(...)` source range in the ledger so Task 8 can relocate that function from the still-intact original into `tool_calling/tasks/sycophancy/sypr_generation.py`. Add tests proving existing pure helper behavior is unchanged.
- [ ] Rewrite imports to canonical absolute names such as `from utils.model_registry import ...` and `from probing.utils.baseline_probes_cache import ...`.
- [ ] Review every `Path(__file__)`, `REPO_ROOT`, `SYCOPHANCY_DIR`, default dataset path, and default results path in the moved files. Shared code must resolve the repository root correctly from its new location and must not default into `tool_calling/` or `SAE/`.
- [ ] Add focused import/path tests for `utils.model_registry`, `probing.data.sycophancy_data`, and `probing.data.sypr_data`.
- [ ] Run the full extraction-branch test suite and commit:

```bash
git commit -m "refactor: extract shared model registry, baseline utilities, and data loaders"
```

## Task 3: Extract baseline-probe and DIM core

- [ ] Create the five core modules and seven steering-free probe drivers listed under “Probe and DIM code.”
- [ ] Apply these canonical import mappings everywhere in the moved code:

```text
pipeline_scripts.cache                    -> probing.utils.baseline_probes_cache
pipeline_scripts.common                   -> probing.utils.baseline_probes_common
pipeline_scripts.datasets                 -> probing.utils.baseline_probes_datasets
pipeline_scripts.training                 -> probing.probe.baseline_training
pipeline_scripts.probing.train_probes     -> probing.probe.train_baseline_probes
sycophancy_probes                         -> probing.probe.baseline_probes
sycophancy_dim                            -> probing.probe.dim
sycophancy_model_registry                 -> utils.model_registry
gpu_memory                                -> probing.utils.gpu_memory
sycophancy_data                           -> probing.data.sycophancy_data
sypr_data                                 -> probing.data.sypr_data
```

- [ ] Do not consolidate the two existing `LinearProbe` implementations merely because their names match. First add behavioral tests that document their constructor, state-dict, and scoring contracts. Deduplicate only in a later change if those contracts are proven equivalent.
- [ ] Port the activation extractor; this is a hard gate. Verify `probing/probe/extract_baseline_activations.py` imports `utils.model_registry` and the shared cache/dataset modules.
- [ ] Update runtime paths and command examples for every moved driver.
- [ ] Run focused tests plus the full suite and commit:

```bash
git commit -m "refactor: extract baseline-probe and DIM core into probing/probe"
```

## Task 4: Extract baseline generation and judging

- [ ] Create `probing/evaluations/baseline_probes/{generation,judge}/__init__.py` and move every file listed in “Baseline evaluation code.”
- [ ] Rewrite all intra-package imports to `probing.evaluations.baseline_probes...` and all shared utility/data imports using Task 3's mapping.
- [ ] Replace old location-derived defaults with either `probing/data`/`probing/results` paths or required CLI arguments. Preserve application-owned outputs by requiring an explicit path; do not invent a new copy on `main`.
- [ ] Update subprocess launch paths in orchestration scripts so they invoke the moved modules (`python -m probing.evaluations...`) rather than deleted filenames.
- [ ] Add smoke tests for module import and `--help` parsing that do not load models or contact external services.
- [ ] Run the full suite and commit:

```bash
git commit -m "refactor: extract baseline generation and judge modules"
```

## Task 5: Extract analysis code and documentation

- [ ] Move every file listed under “Shared analysis code” and all three baseline-probe documents.
- [ ] Rewrite imports using Task 3's mapping.
- [ ] Audit all data/result defaults. In particular, eliminate old `SYCOPHANCY_DIR / "results"` assumptions from the moved scripts and update examples in their docstrings.
- [ ] Verify each moved analysis module has no runtime import of `sycophancy_steering` or `ActivationSteerer`. A historical string or plot label is acceptable; an executable dependency is not.
- [ ] Run import/`--help` smoke tests and the full suite, then commit:

```bash
git commit -m "refactor: extract baseline-probe analysis and documentation"
```

## Task 6: Migrate tests before deleting originals

- [ ] Port `pipeline_scripts/tests/test_pipeline.py` to `tests/test_baseline_probe_pipeline.py` and rewrite imports to the shared package.
- [ ] Preserve its coverage of cache handling, confidence intervals, dataset preparation, DIM computation, SyconBench generation helpers, judge scoring, and training utilities. If the DIM test imports `run_dim.py`, move the pure function test to `probing.probe.dim` or add a small shared helper first; do not make main's test import `tool_calling/`.
- [ ] Add a test that recursively imports the moved modules without model/network access where feasible.
- [ ] Run `python -m compileall probing utils tests` and the complete test suite.
- [ ] Require zero imports of application packages from the new shared code. `utils/sae_utils.py` is explicitly excluded at this pre-cleanup point because it is relocated on `SAE` and removed elsewhere in Tasks 8-10:

```bash
git grep -n -E '(^|[[:space:]])(from|import)[[:space:]]+(tool_calling|SAE|pipeline_scripts|sycophancy_model_registry|sycophancy_probes|sycophancy_data|sypr_data|sycophancy_dim)([[:space:].]|$)' -- probing utils ':(exclude)utils/sae_utils.py'
git grep -n -E 'SAE/|tool_calling/' -- probing utils ':(exclude)utils/sae_utils.py'
```

The second command is a review list rather than an automatic failure: historical prose may mention an old location, but executable imports, defaults, and commands must not depend on it.

- [ ] Commit:

```bash
git commit -m "test: migrate baseline-probe pipeline coverage to shared packages"
```

## Task 7: Land shared history and integrate it before branch cleanup

This is the only point where whole-branch integration is allowed.

- [ ] Record the extraction branch's ordered commit list and tip hash.
- [ ] In the original `main` checkout, require a clean index/worktree for paths touched by the extraction and fast-forward:

```bash
git merge --ff-only baseline-probes-extraction
```

- [ ] Re-run the deletion guard from Task 1. `main` must still have no post-snapshot changes under `SAE/` or `tool_calling/`.
- [ ] Integrate this pre-cleanup `main` into `SAE`. If `SAE` is still exactly the legacy snapshot, this should be a fast-forward; otherwise use a normal merge and review it. Do not remove either application directory yet.
- [ ] Integrate this same pre-cleanup `main` into `building-agent`. The path-diff gate makes this safe: main contributes only shared/prompt-migration paths, while the branch retains its tool-calling commits. Resolve no deletion of `tool_calling/` or `SAE/`; if either appears, abort and reassess.
- [ ] Run the full suite on all three branches before any deletion.
- [ ] Record a `shared-pre-split` tag or immutable commit hash in the ledger. Do not merge `main` wholesale into either application branch after the cleanup commits in Tasks 8-10.

## Task 8: Rewrite and clean `building-agent`

- [ ] Move `pipeline_scripts/difference_in_means/` to `tool_calling/tasks/sycophancy/difference_in_means/` and rewrite it to consume `probing.*` and `utils.model_registry`.
- [ ] Create `tool_calling/tasks/sycophancy/sypr_generation.py` from the `ActivationSteerer`-dependent generation portion of the old `sypr_data.py`; make it import pure dataset helpers from `probing.data.sypr_data`. Update `sypr_probe_pipeline.py` and `sypr_praise_count.py` accordingly.
- [ ] Using Task 1's consumer manifest, rewrite **every** remaining Python consumer across the branch. Do not limit the sweep to the 14 steering-integrated scripts.
- [ ] Update application scripts that invoke moved files by path to use the new module path or a stable wrapper.
- [ ] Port or update application-specific tests before removing originals.
- [ ] Immediately before deletion, run these whole-branch searches and reconcile every match against the consumer/source manifest. No match may remain in a file that will survive cleanup:

```bash
git grep -n -E 'pipeline_scripts\.|sycophancy_model_registry|(^|[[:space:]])(from|import)[[:space:]]+(sycophancy_probes|sycophancy_data|sypr_data|sycophancy_dim)([[:space:].]|$)' -- '*.py'
git grep -n 'tool_calling/tasks/sycophancy/pipeline_scripts' -- ':!*.ipynb'
```

- [ ] Compare the actual old-source manifest with the destination map. Every entry must be classified as moved, deliberately retained/relocated, test-migrated, documentation-migrated, or generated/cache content. No unclassified file may be deleted.
- [ ] Delete only the migrated originals, then remove the now-empty `pipeline_scripts/` directory.
- [ ] Repeat both old-import searches after deletion; now they must return no executable-code matches anywhere on the branch.
- [ ] Replace executable defaults under `SAE/results/` with building-agent-owned paths or required CLI arguments. Require a whole-branch executable-code grep for `SAE/`, `utils.sae_utils`, and `tool_calling/tasks/sae` to return no unresolved dependencies.
- [ ] Remove `SAE/`, `tool_calling/tasks/sae/`, and `utils/sae_utils.py` from `building-agent`. Do not remove the rest of `tool_calling/`, its sycophancy notebooks/skills/tests, or its JSONL/checkpoints.
- [ ] Run compileall, the full suite, and representative `--help` smoke tests. Commit cleanup separately from shared extraction:

```bash
git commit -m "refactor: consume shared probing library from building-agent"
git commit -m "chore: remove SAE application from building-agent branch"
```

## Task 9: Clean the `SAE` branch

- [ ] Verify the branch contains the shared `probing/` and `utils/` commits and still contains `SAE/`.
- [ ] Move `tool_calling/tasks/sae/` to `SAE/agent_tools/` and `utils/sae_utils.py` to `SAE/sae/utils.py`; update imports, repository-root calculations, skill references, and SAE tests before deleting `tool_calling/`.
- [ ] Remove the now-duplicated `SAE/datasets/*.csv` after confirming `utils.datasets` reads byte-identical files from `probing/data/source/elephant/`. Keep `SAE/results/` and all SAE checkpoints/generations.
- [ ] Search `SAE/` for imports of files that are about to disappear with `tool_calling/`; rewrite any real dependency to the shared package before deletion.
- [ ] Remove `tool_calling/` from `SAE` while preserving `SAE/`, shared code, SAE checkpoints, and SAE-owned generated JSONL.
- [ ] Run compileall, the full suite, and an SAE generation/import smoke test.
- [ ] Commit:

```bash
git commit -m "chore: remove tool-calling application from SAE branch"
```

## Task 10: Clean `main`

- [ ] Verify Tasks 8 and 9 are complete and both application branches pass their tests. This is a hard gate.
- [ ] On `main`, remove both `SAE/` and `tool_calling/`, plus the SAE-specific `utils/sae_utils.py`. Their histories and artifacts remain reachable from `legacy/pre-refactor`, `SAE`, and `building-agent`; the shared raw CSVs remain under `probing/data/source/elephant/`.
- [ ] Remove any root documentation links or commands that point to application-only paths; replace them with branch-specific links or clearly label the owning branch.
- [ ] Run the shared compileall/test suite and verify `git ls-tree --name-only HEAD` contains neither application directory.
- [ ] Commit:

```bash
git commit -m "chore: complete application branch split on main"
```

## Task 11: Final cross-branch verification

Run all checks from clean worktrees.

### `main`

- [ ] Contains `probing/`, `utils/`, shared tests, and shared docs.
- [ ] Contains neither `SAE/` nor `tool_calling/`.
- [ ] Shared-code forbidden-import grep returns no matches.
- [ ] Full suite passes.

### `SAE`

- [ ] Contains `SAE/`, `probing/`, and `utils/`.
- [ ] Contains no `tool_calling/`.
- [ ] SAE checkpoints and generated JSONL remain tracked; activation arrays/rendered reports remain ignored.
- [ ] Full suite and SAE smoke test pass.

### `building-agent`

- [ ] Contains `tool_calling/`, `probing/`, and `utils/`.
- [ ] Contains no `SAE/` and no old `pipeline_scripts/` package.
- [ ] `difference_in_means/` remains application-owned and imports shared pure helpers.
- [ ] Tool-calling JSONL, gzip checkpoints, and trained checkpoints remain tracked.
- [ ] Full suite and steerer delegation tests pass.

### Cross-branch integrity

- [ ] Compare hashes—not samples—for every shared file at the recorded `shared-pre-split` point. Later branch-specific edits to a shared file are forbidden; fixes must land as new shared commits on `main` and be cherry-picked explicitly.
- [ ] Confirm `legacy/pre-refactor` still resolves.
- [ ] Confirm no worktree is left locked or attached to a temporary extraction branch once work is complete.
- [ ] Only after all three branches pass should the GitHub default branch be switched to the refactored `main`.

## Commit policy after the split

- Shared fixes: commit on `main`, then cherry-pick the specific commit(s) into `SAE` and/or `building-agent`.
- SAE-only work: commit only on `SAE`.
- Agent/tool-calling work: commit only on `building-agent`.
- Never merge an application branch wholesale back into `main`.
- Never merge post-cleanup `main` wholesale into an application branch, because its application-directory deletions are intentional branch-specific history.
