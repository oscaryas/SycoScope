# Probing Structure Migration (prompt_probes) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `prompt_probes/pipeline/` (37 files) into the new `probing/{utils,data,probe,analyze_probes,evaluations}/` structure on `main`, converting every internal cross-file reference from flat sibling imports (`import common`, `sys.path.insert(0, .../"prompt_probes/pipeline")`) to proper absolute package imports (`from probing.utils import common`), with zero behavior change.

**Architecture:** `probing/` becomes a new top-level Python package (nested to avoid colliding with the existing repo-root `utils/`), with five importable subpackages — `probing/utils/`, `probing/data/`, `probing/probe/`, `probing/analyze_probes/`, `probing/evaluations/prompt_probes/{generation,judge}/` — each getting an `__init__.py`. Every file that currently does `HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import <sibling>` is rewritten to compute `REPO_ROOT` (at the correct depth for its new location) and use `from probing.<pkg> import <module>` instead — this works uniformly regardless of how many directories deep the importing file now sits, unlike the old flat-sibling convention. This is Phase 1 of the larger restructure in `REFACTOR_STRUCTURE_UNDERSTANDING.md` — it covers only `prompt_probes/`, not the `tool_calling` baseline_probes/dim extraction (a separate, cross-branch follow-up plan, since that content currently lives on `building-agent`, not `main`).

**Tech Stack:** Python, `unittest`/`pytest`.

**Spec:** `REFACTOR_STRUCTURE_UNDERSTANDING.md` (repo root, "Resolved so far" + the `probing/` nesting resolution) and `PROBE_RESTRUCTURE_FILE_INVENTORY.md` (repo root — the authoritative source→destination table for every file; this plan sequences and executes exactly what that table specifies, it does not re-derive destinations).

## Global Constraints

- **Import rewrite rule (apply uniformly, no exceptions):** every occurrence of the old flat-sibling imports below, anywhere in a moved file or in a test file that references one, is replaced per this exact table. Delete the old file's `HERE = Path(__file__).resolve().parent` / `sys.path.insert(0, str(HERE))` boilerplate wherever its only purpose was enabling these flat imports.

  | Old (flat sibling import) | New (absolute package import) |
  |---|---|
  | `import common` | `from probing.utils import common` |
  | `import get_activations as ga` | `from probing.probe import get_activations as ga` |
  | `from train_probes import X, Y` | `from probing.probe.train_probes import X, Y` |
  | `import eval_common` | `from probing.analyze_probes import eval_common` |
  | `from eval_common import X, Y` | `from probing.analyze_probes.eval_common import X, Y` |
  | `from analyze_probes import X, Y` | `from probing.analyze_probes.analyze_probes import X, Y` |

- **REPO_ROOT depth by new location** (needed so `sys.path` contains the repo root before the `from probing... import` lines run):
  - Files landing directly under `probing/utils/`, `probing/data/`, `probing/probe/`, `probing/analyze_probes/`: `REPO_ROOT = Path(__file__).resolve().parents[1]`.
  - Files landing under `probing/evaluations/prompt_probes/generation/` or `probing/evaluations/prompt_probes/judge/`: `REPO_ROOT = Path(__file__).resolve().parents[3]`.
  - Test files under `tests/`: `REPO_ROOT = Path(__file__).resolve().parents[1]` (unchanged from today — `tests/` itself doesn't move).
- Every new subpackage directory needs an `__init__.py` (matching the existing repo convention of explicit `__init__.py`, e.g. `prompt_probes/pipeline/__init__.py`): `probing/__init__.py`, `probing/utils/__init__.py`, `probing/data/__init__.py`, `probing/probe/__init__.py`, `probing/analyze_probes/__init__.py`, `probing/evaluations/__init__.py`, `probing/evaluations/prompt_probes/__init__.py`, `probing/evaluations/prompt_probes/generation/__init__.py`, `probing/evaluations/prompt_probes/judge/__init__.py`.
- `probing/results/` and `probing/data/` (the data directories, as opposed to `probing/data/` the *package* of dataset-construction scripts — note the inventory maps dataset-construction *scripts* to `probing/data/` alongside where their *output* would also conceptually live; this plan does not move any existing `prompt_probes/results/*` run data — that stays where it is on disk for now and is out of scope here, addressed by a later results-migration plan) do not need `__init__.py` — they are not imported as Python packages.
- `eval_elephant.py` is renamed to `eval_social_sycophancy.py` on this move (confirmed: it already only covers social sycophancy — this is a pure rename, no content restructuring).
- `run_c_sweep.sh` and `run_c_sweep_eval.sh` need **no path edits** — their relative paths (`../results/...`, `../../tool_calling/...`) resolve identically whether run from `prompt_probes/pipeline/` or from `probing/probe/` and `probing/analyze_probes/` respectively, because both old and new locations are two directories below the repo root with `results/` as a sibling one level up. Verify this assumption in Task 2/4's steps rather than trusting it blindly.
- Do not move `prompt_probes/results/`, `prompt_probes/data/` (the data *files*, e.g. `perez_user_prompts_5k.jsonl`), or any of the `.md`/`.txt` planning documents in `prompt_probes/` — only `prompt_probes/pipeline/*` moves in this plan. (A later plan addresses `results/`.)
- After all tasks, `prompt_probes/pipeline/` must be empty except `__pycache__/` (safe to leave or remove) — verified in the final task.
- Run the full test suite (`./.venv/bin/python -m pytest tests/ -v`, from repo root, with `unset VIRTUAL_ENV` first per this repo's venv quirk) after every task; each task must leave the suite green before moving to the next.

---

### Task 1: Scaffold `probing/` and move `probing/utils/common.py`

**Files:**
- Create: `probing/__init__.py`, `probing/utils/__init__.py`, `probing/utils/common.py` (moved from `prompt_probes/pipeline/common.py`)
- Delete: `prompt_probes/pipeline/common.py`
- No test changes yet (no test imports `common` directly as its own module — it's always imported alongside another module under test; covered by later tasks' test runs).

**Interfaces:**
- Produces: `probing.utils.common` — same public API as before (`PROMPT_PROBES_DIR`/equivalent, `read_jsonl`, etc. — whatever `common.py` currently exports; do not change any of its function signatures or behavior, only its location and its own internal path-computation constants if it computes `PROMPT_PROBES_DIR`/`REPO_ROOT` itself).

- [ ] **Step 1: Read the current file and note its self-referential path constants**

Run: `cat prompt_probes/pipeline/common.py` and identify every place it computes a path relative to `HERE`/`PROMPT_PROBES_DIR`/`REPO_ROOT`. Confirm `PROMPT_PROBES_DIR = HERE.parent` and `REPO_ROOT = HERE.parents[1]` (both computed from `HERE = Path(__file__).resolve().parent`) — these formulas are depth-relative and, since the new location `probing/utils/` is the same depth below the repo root as `prompt_probes/pipeline/` was, will resolve correctly to `probing/` and the repo root respectively **without changing the arithmetic** — only rename the `PROMPT_PROBES_DIR` constant to `PROBING_DIR` for clarity (a name that no longer says "prompt_probes" when it now means "probing"), updating every internal use of the old name in this same file.

- [ ] **Step 2: Create the package directories and move the file**

```bash
mkdir -p probing/utils
touch probing/__init__.py probing/utils/__init__.py
git mv prompt_probes/pipeline/common.py probing/utils/common.py
```

- [ ] **Step 3: Apply the `PROMPT_PROBES_DIR` → `PROBING_DIR` rename inside the moved file**

Edit `probing/utils/common.py`: rename every occurrence of `PROMPT_PROBES_DIR` to `PROBING_DIR`. Do not change the `HERE.parent` / `HERE.parents[1]` arithmetic itself.

- [ ] **Step 4: Verify the module imports cleanly**

Run: `./.venv/bin/python -c "import sys; sys.path.insert(0, '.'); from probing.utils import common; print(common.PROBING_DIR)"` from the repo root.
Expected: prints a path ending in `.../probing` (not `.../prompt_probes`), no import errors.

- [ ] **Step 5: Commit**

```bash
git add probing/__init__.py probing/utils/__init__.py probing/utils/common.py
git rm prompt_probes/pipeline/common.py 2>/dev/null || true
git commit -m "refactor: move prompt_probes/pipeline/common.py to probing/utils/common.py"
```

---

### Task 2: Move `probing/probe/` (get_activations.py, train_probes.py, train_dim.py, run_c_sweep.sh)

**Files:**
- Create: `probing/probe/__init__.py`, `probing/probe/get_activations.py`, `probing/probe/train_probes.py`, `probing/probe/train_dim.py`, `probing/probe/run_c_sweep.sh` (all moved from `prompt_probes/pipeline/`)
- Delete: the four corresponding files under `prompt_probes/pipeline/`
- Modify: `tests/test_prompt_probes_dim.py` (imports `train_dim`), `tests/test_prompt_probes_spans.py` (imports `get_activations`)

**Interfaces:**
- Consumes: `probing.utils.common` (Task 1).
- Produces: `probing.probe.get_activations`, `probing.probe.train_probes`, `probing.probe.train_dim` — same public functions as before (e.g. `train_probes.load_cell`, `.paired_win_rate`, `.safe_auc`, `.DROP_REASONS`; `get_activations` module itself, referenced elsewhere as `ga`), consumed by Task 3 (data — no, data doesn't depend on probe) and Task 4 (analyze_probes).

- [ ] **Step 1: Move the files**

```bash
mkdir -p probing/probe
touch probing/probe/__init__.py
git mv prompt_probes/pipeline/get_activations.py probing/probe/get_activations.py
git mv prompt_probes/pipeline/train_probes.py probing/probe/train_probes.py
git mv prompt_probes/pipeline/train_dim.py probing/probe/train_dim.py
git mv prompt_probes/pipeline/run_c_sweep.sh probing/probe/run_c_sweep.sh
```

- [ ] **Step 2: Fix imports in `probing/probe/get_activations.py`**

Replace:
```python
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
```
with:
```python
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.utils import common  # noqa: E402
```

- [ ] **Step 3: Fix imports in `probing/probe/train_probes.py`**

Replace the same `HERE`/`sys.path` block with the `REPO_ROOT = Path(__file__).resolve().parents[1]` form (as in Step 2). Replace:
```python
import common  # noqa: E402
import get_activations as ga  # noqa: E402
```
with:
```python
from probing.utils import common  # noqa: E402
from probing.probe import get_activations as ga  # noqa: E402
```

- [ ] **Step 4: Fix imports in `probing/probe/train_dim.py`**

Same `REPO_ROOT` fix as Step 2/3. Replace:
```python
import common  # noqa: E402
import get_activations as ga  # noqa: E402
from train_probes import (  # noqa: E402
```
with:
```python
from probing.utils import common  # noqa: E402
from probing.probe import get_activations as ga  # noqa: E402
from probing.probe.train_probes import (  # noqa: E402
```
(keep whatever names follow on the subsequent lines of that `from train_probes import (` block exactly as they are — only the module path changes.)

- [ ] **Step 5: Verify `run_c_sweep.sh`'s relative paths still resolve**

Run: `cd probing/probe && grep -n '\.\./results\|python3' run_c_sweep.sh` — confirm `BASE=../results/llama31_5k_subset` still points at `probing/results/llama31_5k_subset` (i.e. `probing/probe/../results` = `probing/results`) and `python3 train_probes.py` still refers to the sibling file that now lives in the same directory (`probing/probe/train_probes.py`). No edits needed if both check out; if `probing/results/llama31_5k_subset` doesn't exist yet on disk, that's expected (results migration is a separate later plan) — this step only confirms the *path arithmetic*, not that the directory currently exists.

- [ ] **Step 6: Update `tests/test_prompt_probes_dim.py`**

Replace:
```python
PIPELINE = Path(__file__).resolve().parents[1] / "prompt_probes" / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

import train_dim  # noqa: E402
```
with:
```python
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.probe import train_dim  # noqa: E402
```

- [ ] **Step 7: Update `tests/test_prompt_probes_spans.py`**

Find its `PIPELINE = REPO_ROOT / "prompt_probes" / "pipeline"` block and the `sys.path.insert` loop that follows. Since this file's own `REPO_ROOT` variable is already computed correctly (it's under `tests/`, unaffected by this move), only remove the `PIPELINE` path from whatever `sys.path` insertion list it's added to (the repo root is what's needed now, not a pipeline-specific path) — read the file first to see the exact surrounding loop before editing, since it inserts multiple paths in a list/loop rather than a single line. Update its lazy import of `SAE.pipeline.cache_activations.response_token_span` (unaffected — different subsystem) but fix the docstring reference at the top of the file from `prompt_probes/pipeline/get_activations.py` to `probing/probe/get_activations.py` if that module is directly imported anywhere in the file (check with `grep -n "get_activations" tests/test_prompt_probes_spans.py`); if it's only referenced in a comment/docstring, update the comment for accuracy but this is not functionally required.

- [ ] **Step 8: Run the full test suite**

Run: `unset VIRTUAL_ENV && ./.venv/bin/python -m pytest tests/ -v 2>&1 | tail -40`
Expected: `test_prompt_probes_dim.py` and `test_prompt_probes_spans.py` PASS (or SKIP for network-dependent cases, matching their pre-existing behavior), no new failures anywhere else.

- [ ] **Step 9: Commit**

```bash
git add probing/probe tests/test_prompt_probes_dim.py tests/test_prompt_probes_spans.py
git rm prompt_probes/pipeline/get_activations.py prompt_probes/pipeline/train_probes.py prompt_probes/pipeline/train_dim.py prompt_probes/pipeline/run_c_sweep.sh 2>/dev/null || true
git commit -m "refactor: move prompt_probes/pipeline probe-training files to probing/probe/"
```

---

### Task 3: Move `probing/data/` (fetch_user_prompts.py, dataset_control.py, prompt_wording_control.py, build_llama31_subset.py, + supervisor confs)

**Files:**
- Create: `probing/data/__init__.py`, `probing/data/fetch_user_prompts.py`, `probing/data/dataset_control.py`, `probing/data/prompt_wording_control.py`, `probing/data/build_llama31_subset.py`, `probing/data/dataset_control.supervisor.conf`, `probing/data/prompt_wording_control.supervisor.conf`
- Delete: the six corresponding files under `prompt_probes/pipeline/`
- Modify: `tests/test_dataset_control.py`, `tests/test_prompt_wording_control.py`

**Interfaces:**
- Consumes: `probing.utils.common` (Task 1); `prompt_wording_control.py` also consumes `probing.probe.train_probes` (`fit_probe`, `safe_auc`) and `probing.analyze_probes.analyze_probes` (`apply_probe`) — the latter doesn't exist until Task 4. **Order dependency: do Task 4 before finishing `prompt_wording_control.py`'s import fix, OR fix the import now (it's just a text change) and accept that `prompt_wording_control.py` won't successfully run end-to-end / its test may fail until Task 4 lands.** Prefer the second (fix the import text now, note the test may fail until Task 4, and confirm it's fixed retroactively in Task 4's test run) so this task's file moves aren't blocked on Task 4's completion.
- Produces: `probing.data.dataset_control`, `probing.data.prompt_wording_control`, `probing.data.fetch_user_prompts`, `probing.data.build_llama31_subset`.

- [ ] **Step 1: Move the files**

```bash
mkdir -p probing/data
touch probing/data/__init__.py
git mv prompt_probes/pipeline/fetch_user_prompts.py probing/data/fetch_user_prompts.py
git mv prompt_probes/pipeline/dataset_control.py probing/data/dataset_control.py
git mv prompt_probes/pipeline/prompt_wording_control.py probing/data/prompt_wording_control.py
git mv prompt_probes/pipeline/build_llama31_subset.py probing/data/build_llama31_subset.py
git mv prompt_probes/pipeline/dataset_control.supervisor.conf probing/data/dataset_control.supervisor.conf
git mv prompt_probes/pipeline/prompt_wording_control.supervisor.conf probing/data/prompt_wording_control.supervisor.conf
```

- [ ] **Step 2: Fix imports in `probing/data/fetch_user_prompts.py`, `dataset_control.py`, `build_llama31_subset.py`**

Each has (or, for `build_llama31_subset.py`, may have no internal import at all — confirmed earlier it has zero internal sibling imports, so skip it here) a `HERE`/`sys.path.insert` block feeding `import common`. For `fetch_user_prompts.py` and `dataset_control.py`, apply the same fix as Task 2 Step 2: `REPO_ROOT = Path(__file__).resolve().parents[1]`, `from probing.utils import common`.

- [ ] **Step 3: Fix imports in `probing/data/prompt_wording_control.py`**

Replace:
```python
import common  # noqa: E402
from analyze_probes import apply_probe  # noqa: E402
from train_probes import fit_probe, safe_auc  # noqa: E402
```
with:
```python
from probing.utils import common  # noqa: E402
from probing.analyze_probes.analyze_probes import apply_probe  # noqa: E402
from probing.probe.train_probes import fit_probe, safe_auc  # noqa: E402
```
along with the same `REPO_ROOT` fix for its `HERE`/`sys.path` block. (This import will only resolve once Task 4 has created `probing/analyze_probes/analyze_probes.py` — that's fine; Python doesn't evaluate the import until the module is actually run, and this task's test step below only exercises the parts of `prompt_wording_control.py` that don't require `apply_probe`. If the existing test does exercise it, note the failure and it will self-resolve once Task 4 lands; do not block this task on that — record it as a note in your task report instead.)

- [ ] **Step 4: Update the two supervisor `.conf` files' hardcoded paths**

In `probing/data/dataset_control.supervisor.conf`: replace every occurrence of `prompt_probes/pipeline/dataset_control.py` with `probing/data/dataset_control.py`, and every occurrence of `prompt_probes/results/` with `probing/results/` (these appear in the `command=` and `stdout_logfile=` lines).
In `probing/data/prompt_wording_control.supervisor.conf`: apply the same two substitutions for whatever script/log paths it references (read the file first — it wasn't dumped in this plan's research, but follows the identical pattern as `dataset_control.supervisor.conf`; verify with `grep -n "prompt_probes" probing/data/prompt_wording_control.supervisor.conf` before and after).

- [ ] **Step 5: Update `tests/test_dataset_control.py`**

Replace:
```python
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'prompt_probes/pipeline'))
import dataset_control as dc
```
with:
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.data import dataset_control as dc
```

- [ ] **Step 6: Update `tests/test_prompt_wording_control.py`**

Replace:
```python
PIPELINE = Path(__file__).resolve().parents[1] / "prompt_probes" / "pipeline"
sys.path.insert(0, str(PIPELINE))
```
with:
```python
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
```
Then find and update whatever `import <module>` line follows to `from probing.data import <module>` (read the file to find the exact current import statement — it wasn't captured verbatim in this plan's research beyond confirming the `PIPELINE`/`sys.path` header; likely `import prompt_wording_control as pwc` or similar, following the same pattern as `test_dataset_control.py`).

- [ ] **Step 7: Run the full test suite**

Run: `unset VIRTUAL_ENV && ./.venv/bin/python -m pytest tests/ -v 2>&1 | tail -40`
Expected: `test_dataset_control.py` PASSes. `test_prompt_wording_control.py` may still fail/error on the `apply_probe` import until Task 4 lands (per Step 3's note) — if so, confirm the failure is specifically an `ImportError`/`ModuleNotFoundError` for `probing.analyze_probes.analyze_probes` and nothing else, record it in your report, and do not treat it as this task's own defect.

- [ ] **Step 8: Commit**

```bash
git add probing/data tests/test_dataset_control.py tests/test_prompt_wording_control.py
git rm prompt_probes/pipeline/fetch_user_prompts.py prompt_probes/pipeline/dataset_control.py prompt_probes/pipeline/prompt_wording_control.py prompt_probes/pipeline/build_llama31_subset.py prompt_probes/pipeline/dataset_control.supervisor.conf prompt_probes/pipeline/prompt_wording_control.supervisor.conf 2>/dev/null || true
git commit -m "refactor: move prompt_probes/pipeline dataset-construction files to probing/data/"
```

---

### Task 4: Move `probing/analyze_probes/` (analyze_probes.py, eval_common.py, all eval_*.py, summarize/cluster/activation_variance/build_probe_html_report/audit_*.py, run_c_sweep_eval.sh)

**Files:**
- Create: `probing/analyze_probes/__init__.py`, `probing/analyze_probes/analyze_probes.py`, `probing/analyze_probes/eval_common.py`, `probing/analyze_probes/eval_are_you_sure.py`, `probing/analyze_probes/eval_social_sycophancy.py` (renamed from `eval_elephant.py`), `probing/analyze_probes/eval_moral.py`, `probing/analyze_probes/eval_sypr.py`, `probing/analyze_probes/eval_syconbench.py`, `probing/analyze_probes/eval_cross_cell.py`, `probing/analyze_probes/eval_specificity.py`, `probing/analyze_probes/eval_matrix.py`, `probing/analyze_probes/summarize_syconbench.py`, `probing/analyze_probes/cluster_syconbench_stability.py`, `probing/analyze_probes/activation_variance.py`, `probing/analyze_probes/build_probe_html_report.py`, `probing/analyze_probes/probe_report_assets/` (directory), `probing/analyze_probes/audit_disagreements.py`, `probing/analyze_probes/audit_external_persona_basis.py`, `probing/analyze_probes/run_c_sweep_eval.sh`
- Delete: all corresponding files under `prompt_probes/pipeline/`
- Modify: `tests/test_activation_variance.py`, `tests/test_prompt_probes_syconbench.py`, `tests/test_syconbench_clusters.py`, `tests/test_syconbench_budgeted_judge.py` (this last one only if it imports `eval_common`/`analyze_probes` — verify; per earlier research it imports `judge_syconbench_budgeted`, which is Task 6's concern, not this one — double check before editing)

**Interfaces:**
- Consumes: `probing.utils.common`, `probing.probe.get_activations`, `probing.probe.train_probes` (all from Tasks 1-2).
- Produces: `probing.analyze_probes.analyze_probes` (`apply_probe`, `load_probes`, `cluster_sweep`, `heatmap`, `correlation_dendrogram`), `probing.analyze_probes.eval_common` (`group_mean`, `selection_split`, `split_key`), consumed by Task 3's `prompt_wording_control.py` (retroactively resolved here) and by files within this same task.

- [ ] **Step 1: Move the files (including the rename)**

```bash
mkdir -p probing/analyze_probes
touch probing/analyze_probes/__init__.py
for f in analyze_probes eval_common eval_are_you_sure eval_moral eval_sypr eval_syconbench eval_cross_cell eval_specificity eval_matrix summarize_syconbench cluster_syconbench_stability activation_variance build_probe_html_report audit_disagreements audit_external_persona_basis; do
  git mv "prompt_probes/pipeline/${f}.py" "probing/analyze_probes/${f}.py"
done
git mv prompt_probes/pipeline/eval_elephant.py probing/analyze_probes/eval_social_sycophancy.py
git mv prompt_probes/pipeline/probe_report_assets probing/analyze_probes/probe_report_assets
git mv prompt_probes/pipeline/run_c_sweep_eval.sh probing/analyze_probes/run_c_sweep_eval.sh
```

- [ ] **Step 2: Fix imports in `probing/analyze_probes/analyze_probes.py`**

Apply the `REPO_ROOT = Path(__file__).resolve().parents[1]` fix. Replace:
```python
import common  # noqa: E402
import get_activations as ga  # noqa: E402
from train_probes import load_cell, paired_win_rate, safe_auc  # noqa: E402
```
with:
```python
from probing.utils import common  # noqa: E402
from probing.probe import get_activations as ga  # noqa: E402
from probing.probe.train_probes import load_cell, paired_win_rate, safe_auc  # noqa: E402
```

- [ ] **Step 3: Fix imports in `probing/analyze_probes/eval_common.py`**

Apply the `REPO_ROOT` fix. Replace:
```python
import common
import get_activations as ga
from analyze_probes import apply_probe, load_probes
from train_probes import safe_auc
```
with:
```python
from probing.utils import common
from probing.probe import get_activations as ga
from probing.analyze_probes.analyze_probes import apply_probe, load_probes
from probing.probe.train_probes import safe_auc
```

- [ ] **Step 4: Fix imports in the six simple `eval_*.py` files (`eval_are_you_sure.py`, `eval_social_sycophancy.py`, `eval_moral.py`, `eval_sypr.py`)**

Each has the pattern:
```python
import common  # noqa: E402
import eval_common  # noqa: E402
```
Replace with:
```python
from probing.utils import common  # noqa: E402
from probing.analyze_probes import eval_common  # noqa: E402
```
plus the `REPO_ROOT` fix, in each of these four files. (`eval_syconbench.py` only has `import common`, no `eval_common` — handle it alone with just the `common` substitution.) (`eval_cross_cell.py` has the same two-line pattern as this group — include it here too if not already listed in Step 5.)

- [ ] **Step 5: Fix imports in `eval_matrix.py`, `eval_specificity.py`**

Replace:
```python
import common  # noqa: E402
import get_activations as ga  # noqa: E402
from analyze_probes import apply_probe, load_probes  # noqa: E402
from eval_common import group_mean, selection_split, split_key  # noqa: E402   # (eval_matrix.py only)
from train_probes import safe_auc  # noqa: E402   # (eval_matrix.py)
from train_probes import DROP_REASONS, safe_auc  # noqa: E402   # (eval_specificity.py)
```
with the `probing.`-prefixed equivalents per the Global Constraints table, plus `REPO_ROOT` fix. Handle each file's exact symbol list precisely (don't merge the two files' distinct import lists — `eval_matrix.py` additionally imports from `eval_common`, `eval_specificity.py` does not).

- [ ] **Step 6: Fix imports in `summarize_syconbench.py`**

Replace:
```python
import common
import get_activations as ga
from analyze_probes import apply_probe, load_probes, cluster_sweep, heatmap, correlation_dendrogram
from eval_common import selection_split, split_key
from train_probes import safe_auc
```
with the `probing.`-prefixed equivalents, plus `REPO_ROOT` fix.

- [ ] **Step 7: Fix imports in `cluster_syconbench_stability.py`**

Replace:
```python
import common
from analyze_probes import apply_probe, load_probes, heatmap
```
with:
```python
from probing.utils import common
from probing.analyze_probes.analyze_probes import apply_probe, load_probes, heatmap
```
plus `REPO_ROOT` fix.

- [ ] **Step 8: Fix imports in `activation_variance.py`, `audit_disagreements.py`, `audit_external_persona_basis.py`**

`activation_variance.py`: `import common` → `from probing.utils import common`, plus `REPO_ROOT` fix.
`audit_disagreements.py`: replace
```python
import common  # noqa: E402
import get_activations as ga  # noqa: E402
from analyze_probes import apply_probe, load_probes  # noqa: E402
from train_probes import DROP_REASONS  # noqa: E402
```
with the `probing.`-prefixed equivalents, plus `REPO_ROOT` fix.
`audit_external_persona_basis.py`: `import common` → `from probing.utils import common`, plus `REPO_ROOT` fix.

- [ ] **Step 9: `build_probe_html_report.py` — check for internal imports and fix if present**

This file had no internal sibling import matched in this plan's research grep (only checked for `common`/`get_activations`/`train_probes`/`eval_common`/`analyze_probes`) — but re-verify with `grep -nE "^import |^from " probing/analyze_probes/build_probe_html_report.py` since it reads `probe_report_assets/` and may reference other pipeline modules not covered by that grep pattern. Apply the same fix pattern to anything found; if genuinely nothing internal, only add the boilerplate-removal (delete any now-unused `HERE`/`sys.path` block if one exists but is now dead code).

- [ ] **Step 10: Verify `run_c_sweep_eval.sh`'s relative paths**

Run: `cd probing/analyze_probes && grep -n '\.\./results\|\.\./\.\./tool_calling\|python3' run_c_sweep_eval.sh` — confirm `BASE=../results/llama31_5k_subset` resolves to `probing/results/llama31_5k_subset` and `AITA=../../tool_calling/...` resolves to the repo-root `tool_calling/...` path, both unchanged in meaning from the old location (same depth-from-repo-root as before). No edits needed if confirmed; note in your report if the arithmetic doesn't check out and stop to ask rather than guessing a fix.

- [ ] **Step 11: Update `tests/test_activation_variance.py`**

Replace:
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prompt_probes/pipeline"))
from activation_variance import decompose, bootstrap_kernels, resampled_shares, question_bootstrap
```
with:
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.analyze_probes.activation_variance import decompose, bootstrap_kernels, resampled_shares, question_bootstrap
```

- [ ] **Step 12: Update `tests/test_prompt_probes_syconbench.py`**

Replace:
```python
PIPELINE = Path(__file__).resolve().parents[1] / "prompt_probes/pipeline"
sys.path.insert(0, str(PIPELINE))
import eval_syconbench as sycon
```
with:
```python
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from probing.analyze_probes import eval_syconbench as sycon
```

- [ ] **Step 13: Update `tests/test_syconbench_clusters.py`**

Replace:
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prompt_probes/pipeline"))
import cluster_syconbench_stability as cluster
from analyze_probes import cluster_sweep
```
with:
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probing.analyze_probes import cluster_syconbench_stability as cluster
from probing.analyze_probes.analyze_probes import cluster_sweep
```

- [ ] **Step 14: Check `tests/test_syconbench_budgeted_judge.py` for any `analyze_probes`/`eval_common` reference**

Run: `grep -nE "eval_common|analyze_probes" tests/test_syconbench_budgeted_judge.py`. Per this plan's research it only imports `judge_syconbench_budgeted` (Task 6's file) — if the grep confirms no match, make no changes here (Task 6 handles its `sys.path`/import fix). If it does match something, fix per the same substitution table before proceeding.

- [ ] **Step 15: Run the full test suite**

Run: `unset VIRTUAL_ENV && ./.venv/bin/python -m pytest tests/ -v 2>&1 | tail -60`
Expected: all of `test_activation_variance.py`, `test_prompt_probes_syconbench.py`, `test_syconbench_clusters.py` PASS. `test_prompt_wording_control.py` (from Task 3, previously blocked on this task) should now also PASS — confirm this explicitly and note it in your report.

- [ ] **Step 16: Commit**

```bash
git add probing/analyze_probes tests/test_activation_variance.py tests/test_prompt_probes_syconbench.py tests/test_syconbench_clusters.py
git rm -r prompt_probes/pipeline/analyze_probes.py prompt_probes/pipeline/eval_common.py prompt_probes/pipeline/eval_are_you_sure.py prompt_probes/pipeline/eval_elephant.py prompt_probes/pipeline/eval_moral.py prompt_probes/pipeline/eval_sypr.py prompt_probes/pipeline/eval_syconbench.py prompt_probes/pipeline/eval_cross_cell.py prompt_probes/pipeline/eval_specificity.py prompt_probes/pipeline/eval_matrix.py prompt_probes/pipeline/summarize_syconbench.py prompt_probes/pipeline/cluster_syconbench_stability.py prompt_probes/pipeline/activation_variance.py prompt_probes/pipeline/build_probe_html_report.py prompt_probes/pipeline/probe_report_assets prompt_probes/pipeline/audit_disagreements.py prompt_probes/pipeline/audit_external_persona_basis.py prompt_probes/pipeline/run_c_sweep_eval.sh 2>/dev/null || true
git commit -m "refactor: move prompt_probes/pipeline analysis files to probing/analyze_probes/, rename eval_elephant.py to eval_social_sycophancy.py"
```

---

### Task 5: Move `probing/evaluations/prompt_probes/generation/` (generate_response.py, generate_response_openrouter.py, generate_syconbench_gpu.py, finish_syconbench_vast.py)

**Files:**
- Create: `probing/evaluations/__init__.py`, `probing/evaluations/prompt_probes/__init__.py`, `probing/evaluations/prompt_probes/generation/__init__.py`, and the four moved `.py` files under that last directory
- Delete: the four corresponding files under `prompt_probes/pipeline/`
- Modify: `tests/test_finish_syconbench_vast.py`

**Interfaces:**
- Consumes: `probing.utils.common`.
- Produces: `probing.evaluations.prompt_probes.generation.{generate_response,generate_response_openrouter,generate_syconbench_gpu,finish_syconbench_vast}`.

- [ ] **Step 1: Move the files**

```bash
mkdir -p probing/evaluations/prompt_probes/generation
touch probing/evaluations/__init__.py probing/evaluations/prompt_probes/__init__.py probing/evaluations/prompt_probes/generation/__init__.py
git mv prompt_probes/pipeline/generate_response.py probing/evaluations/prompt_probes/generation/generate_response.py
git mv prompt_probes/pipeline/generate_response_openrouter.py probing/evaluations/prompt_probes/generation/generate_response_openrouter.py
git mv prompt_probes/pipeline/generate_syconbench_gpu.py probing/evaluations/prompt_probes/generation/generate_syconbench_gpu.py
git mv prompt_probes/pipeline/finish_syconbench_vast.py probing/evaluations/prompt_probes/generation/finish_syconbench_vast.py
```

- [ ] **Step 2: Fix imports — note the DIFFERENT depth for this location**

These files now sit 3 directories below `probing/` (`evaluations/prompt_probes/generation/`), so per Global Constraints their `REPO_ROOT` is `Path(__file__).resolve().parents[3]`, not `parents[1]`. For `generate_response.py` and `generate_response_openrouter.py` (both have `import common  # noqa: E402`), replace their `HERE`/`sys.path` block with:
```python
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.utils import common  # noqa: E402
```
`generate_syconbench_gpu.py` and `finish_syconbench_vast.py` had no internal sibling imports matched in this plan's research — verify with `grep -nE "^import |^from |HERE|sys\.path" probing/evaluations/prompt_probes/generation/generate_syconbench_gpu.py probing/evaluations/prompt_probes/generation/finish_syconbench_vast.py`; if either does have a dead `HERE`/`sys.path` block with nothing depending on it, it's harmless to leave, but remove it for cleanliness if you find one.

- [ ] **Step 3: Update `tests/test_finish_syconbench_vast.py`**

It currently does `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prompt_probes/pipeline"))` — read the rest of the file to find its actual `import finish_syconbench_vast` (or similar) line, since this plan's research only captured the header. Replace the `sys.path.insert` target with `str(Path(__file__).resolve().parents[1])` (still `parents[1]` here — `tests/` itself hasn't moved) and change the import to `from probing.evaluations.prompt_probes.generation import finish_syconbench_vast` (or the equivalent `from ... import <symbol>` form matching whatever the file currently does).

- [ ] **Step 4: Run the full test suite**

Run: `unset VIRTUAL_ENV && ./.venv/bin/python -m pytest tests/ -v 2>&1 | tail -40`
Expected: `test_finish_syconbench_vast.py` PASSes (or SKIPs per its existing network/subprocess-mocking behavior), no new failures.

- [ ] **Step 5: Commit**

```bash
git add probing/evaluations tests/test_finish_syconbench_vast.py
git rm prompt_probes/pipeline/generate_response.py prompt_probes/pipeline/generate_response_openrouter.py prompt_probes/pipeline/generate_syconbench_gpu.py prompt_probes/pipeline/finish_syconbench_vast.py 2>/dev/null || true
git commit -m "refactor: move prompt_probes/pipeline generation scripts to probing/evaluations/prompt_probes/generation/"
```

---

### Task 6: Move `probing/evaluations/prompt_probes/judge/` (judge_behaviors.py, judge_syconbench_budgeted.py, + supervisor confs)

**Files:**
- Create: `probing/evaluations/prompt_probes/judge/__init__.py`, `probing/evaluations/prompt_probes/judge/judge_behaviors.py`, `probing/evaluations/prompt_probes/judge/judge_syconbench_budgeted.py`, `probing/evaluations/prompt_probes/judge/syconbench_extraction.supervisor.conf`, `probing/evaluations/prompt_probes/judge/syconbench_generation.supervisor.conf`
- Delete: the four corresponding files under `prompt_probes/pipeline/`
- Modify: `tests/test_prompt_probes_behavior_judge.py`, `tests/test_syconbench_budgeted_judge.py`

**Interfaces:**
- Consumes: `probing.utils.common` (if either file uses it — verify, neither was confirmed to import `common` in this plan's research).
- Produces: `probing.evaluations.prompt_probes.judge.{judge_behaviors,judge_syconbench_budgeted}`.

- [ ] **Step 1: Move the files**

```bash
mkdir -p probing/evaluations/prompt_probes/judge
touch probing/evaluations/prompt_probes/judge/__init__.py
git mv prompt_probes/pipeline/judge_behaviors.py probing/evaluations/prompt_probes/judge/judge_behaviors.py
git mv prompt_probes/pipeline/judge_syconbench_budgeted.py probing/evaluations/prompt_probes/judge/judge_syconbench_budgeted.py
git mv prompt_probes/pipeline/syconbench_extraction.supervisor.conf probing/evaluations/prompt_probes/judge/syconbench_extraction.supervisor.conf
git mv prompt_probes/pipeline/syconbench_generation.supervisor.conf probing/evaluations/prompt_probes/judge/syconbench_generation.supervisor.conf
```

- [ ] **Step 2: Check and fix any internal imports**

Run: `grep -nE "^import |^from |HERE|sys\.path" probing/evaluations/prompt_probes/judge/judge_behaviors.py probing/evaluations/prompt_probes/judge/judge_syconbench_budgeted.py`. Neither was matched against `common`/`get_activations`/`train_probes`/`eval_common`/`analyze_probes` in this plan's research, but re-verify directly (they may still have a `HERE`/`sys.path` block for other purposes, e.g. importing a shared judge helper from elsewhere in the repo — if so, use `REPO_ROOT = Path(__file__).resolve().parents[3]` per this location's depth, and update whatever import target it fed to its new `probing.`-prefixed or otherwise-repo-root-relative path).

- [ ] **Step 3: Update the two remaining supervisor `.conf` files' hardcoded paths**

Same substitution as Task 3 Step 4: replace `prompt_probes/pipeline/<script>.py` with `probing/evaluations/prompt_probes/judge/<script>.py`, and `prompt_probes/results/` with `probing/results/`, in both `syconbench_extraction.supervisor.conf` and `syconbench_generation.supervisor.conf`. Read each file first to find its exact script/log path references before editing.

- [ ] **Step 4: Update `tests/test_prompt_probes_behavior_judge.py`**

Read the file (its header wasn't fully captured in this plan's research beyond confirming it references `prompt_probes` somewhere) to find its exact `sys.path`/import pattern for `judge_behaviors`, and apply the same style of fix as the other test updates in this plan: point `sys.path`/`REPO_ROOT` at the true repo root, and import via `from probing.evaluations.prompt_probes.judge import judge_behaviors` (or the specific symbols it uses from that module).

- [ ] **Step 5: Update `tests/test_syconbench_budgeted_judge.py`**

Replace:
```python
sys.path.insert(0, str(ROOT / "prompt_probes/pipeline"))
import judge_syconbench_budgeted as judge
```
with:
```python
sys.path.insert(0, str(ROOT))
from probing.evaluations.prompt_probes.judge import judge_syconbench_budgeted as judge
```
(keep whatever `ROOT = ...` computation already exists above this block unchanged, since it already correctly resolves to the repo root — only the inserted path and the import line change.)

- [ ] **Step 6: Run the full test suite**

Run: `unset VIRTUAL_ENV && ./.venv/bin/python -m pytest tests/ -v 2>&1 | tail -40`
Expected: `test_prompt_probes_behavior_judge.py` and `test_syconbench_budgeted_judge.py` PASS, no new failures.

- [ ] **Step 7: Commit**

```bash
git add probing/evaluations tests/test_prompt_probes_behavior_judge.py tests/test_syconbench_budgeted_judge.py
git rm prompt_probes/pipeline/judge_behaviors.py prompt_probes/pipeline/judge_syconbench_budgeted.py prompt_probes/pipeline/syconbench_extraction.supervisor.conf prompt_probes/pipeline/syconbench_generation.supervisor.conf 2>/dev/null || true
git commit -m "refactor: move prompt_probes/pipeline judge scripts to probing/evaluations/prompt_probes/judge/"
```

---

### Task 7: Final cleanup and whole-suite verification

**Files:**
- Delete: `prompt_probes/pipeline/__init__.py`, `prompt_probes/pipeline/__pycache__/` (if present), and the now-empty `prompt_probes/pipeline/` directory itself
- No other files created/modified.

**Interfaces:** none new — this task only verifies the end state.

- [ ] **Step 1: Confirm `prompt_probes/pipeline/` is empty of real content**

Run: `find prompt_probes/pipeline -type f -not -name '*.pyc' -not -path '*__pycache__*'`
Expected: only `prompt_probes/pipeline/__init__.py` remains (every other file was moved by Tasks 1-6).

- [ ] **Step 2: Remove the empty directory**

```bash
git rm prompt_probes/pipeline/__init__.py
rm -rf prompt_probes/pipeline/__pycache__
rmdir prompt_probes/pipeline
```

- [ ] **Step 3: Run the complete test suite one final time**

Run: `unset VIRTUAL_ENV && ./.venv/bin/python -m pytest tests/ -v 2>&1 | tail -80`
Expected: every test that passed before Task 1 (the original baseline) still passes; every test touched by Tasks 2-6 passes; no `ImportError`/`ModuleNotFoundError` anywhere in the output.

- [ ] **Step 4: Grep the whole repo for any remaining stale reference to the old path**

Run: `grep -rn "prompt_probes/pipeline\|prompt_probes\.pipeline" --include='*.py' --include='*.md' --include='*.conf' --include='*.sh' . 2>/dev/null | grep -v '\.venv/\|third_party/'`
Expected: no matches (or only matches in historical/legacy documentation that intentionally describes the old structure, e.g. `REFACTOR_STRUCTURE_UNDERSTANDING.md`'s own discussion of what used to exist — use judgment, but any match inside actual code or currently-active docs is a bug to fix).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove empty prompt_probes/pipeline/ directory after probing/ migration"
```

---

## Self-Review Notes

- **Spec coverage:** every file listed in `PROBE_RESTRUCTURE_FILE_INVENTORY.md`'s `prompt_probes/pipeline/` table has a task that moves it (Tasks 1-6), and the directory's final emptiness is verified (Task 7). The `eval_elephant.py` → `eval_social_sycophancy.py` rename and the two shell scripts' relocation (with a verification-not-blind-trust step for their relative paths) are both explicit tasks/steps.
- **Placeholder scan:** every import fix gives the literal before/after text or an exact, mechanical substitution rule (the Global Constraints table) — no "update the imports appropriately" language. A few steps (Task 2 Step 7, Task 4 Steps 9/14, Task 5 Step 2, Task 6 Steps 2/4) explicitly ask the implementer to `grep`/read a file first because this plan's own research did not capture that file's exact current content byte-for-byte — this is flagged as a verification step with a stated expectation, not a vague instruction, and matches how a careful engineer would actually approach a large mechanical refactor of files not all individually inspected in advance.
- **Type consistency:** the import-rewrite table in Global Constraints is applied identically and by name in every task (`common` → `probing.utils.common`, `get_activations`/`ga` → `probing.probe.get_activations`, etc.) — no task invents a different target name for the same source module.
- **Ordering:** Task 3's `prompt_wording_control.py` has a forward dependency on Task 4's `probing.analyze_probes.analyze_probes` module, explicitly called out with instructions not to block on it and to confirm resolution retroactively in Task 4 — this is the one real inter-task dependency wrinkle in an otherwise linear task order, and it's handled explicitly rather than silently.
