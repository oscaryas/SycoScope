# Multi-model sycophancy generation — session handoff (2026-08-25)

Written so a fresh Claude Code instance can continue without re-deriving anything.
Everything below is on branch **`worktree-fix-steerer-asymmetries`** (pushed to
origin; based on `sentence-probes-wip`). Merge it into `sentence-probes-wip` when done.

## Why the colab-mcp tools were unavailable this session

colab-mcp registers its notebook-execution tools (`run_code_cell` etc.) *dynamically*,
only after `open_colab_browser_connection` succeeds against a live Colab tab. In this
session the server was still "connecting" when the session started, and the one time
the connection call returned `true` the server dropped immediately afterwards, so the
execution tools were never advertised to this session (MCP tool lists are fetched at
connect time; a post-reconnect `tools/list_changed` never arrived). Only
`open_colab_browser_connection` was ever visible. Fix for a new instance: **open the
Colab notebook tab in Chrome BEFORE starting the session**, confirm `/mcp` shows
colab-mcp connected, then call `open_colab_browser_connection` first thing and check
that `mcp__colab-mcp__run_code_cell` appears (ToolSearch "colab"). If it does, use it —
that's the pattern that worked in earlier sessions (see memory / SENTENCE_LEVEL_PROBE_STATUS.md).
Fallback that worked here: drive Colab with the Chrome browser tools (gotchas below).

## What was done this session (all committed + pushed)

| Commit | What |
|---|---|
| `4ec5dc1` | ActivationSteerer: `add_special_tokens=False` (no doubled BOS), explicit terminators; `tools.py` steer_and_generate chat-templates its raw prompt; `mixture_dim_pipeline` sypr branch no longer double-wraps the already-rendered prompt |
| `374745e` | `truthfulqa_probe_transfer_pipeline.py` brought into git + `--direction-format dim` (DimScorer: unit-renormalized DIM vectors + stored train-midpoint threshold + orientation) |
| `9219ccc` | `--dim-cv-folds`: genuine out-of-sample 5-fold DIM scores on the mixture (stratified category×label); mixture target under dim uses moralfix extraction |
| `0f0c295` | Ported the uncommitted `Meta-Llama-3-8B-Instruct` registry entry (fresh clones were silently reintroducing whole-sequence pooling) |
| `8252f4c` | Registered `google/gemma-4-12B-it` (answer_token_id 101 `<channel\|>`); fixed Qwen3-8B/14B `mlp_dim` (12288 / 17408) |
| `8fc5c80`, `b7ad158` | Model-agnostic terminators shared by both generation paths (`utils.inference.resolve_terminators`, union of generation_config + tokenizer.eos + `<\|eot_id\|>`); `strip_reasoning` + think-aware `parse_mc_letter`; untracked generate scripts (dissociating, truthfulqa, syconbench fetch) brought into git |
| `b2586a8` | Registered `Qwen/Qwen3.8-27B` (new arch `qwen3_5`, NEW vocab: `<\|im_start\|>`=248045, head_dim 256 explicit) and `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` (Llama-3.1 shape, 128007) |
| `dd09579` | `--system-prompt` on all 4 generation scripts (Nemotron thinking needs `'detailed thinking on'`); steerer warns on generations that hit the token cap (INCOMPLETE); all 4 Claude judges strip reasoning and judge only the visible answer (truncated think → None, no API call); `scripts/verify_model_generation_setup.py` |
| `8a67e96` | `scripts/colab_multimodel_generate.py` — the run driver |
| `d4b1647` | `scripts/colab_load_secrets.py` — `%run` it in the kernel to load HF_TOKEN / ANTHROPIC_API_KEY / GITHUB_TOKEN from Colab Secrets |

Verification status: `verify_model_generation_setup.py` PASSES for all 6 models
(Llama-3-8B, Qwen3-8B/14B, gemma-4-12B-it, Nemotron-Nano-8B-v1, Qwen3.8-27B): no doubled
special tokens, thinking active, terminators include the real end-of-turn token.
CPU end-to-end smoke on Qwen3-0.6B via the real steerer path passed (single BOS,
`<think>` generated+closed, completes in budget, cap warning fires, truncated think → None).

User decisions recorded: Qwen3-8B keeps `enable_thinking=True` (memory note). Judges
see only the user-visible answer. Extraction-side doubled BOS deliberately left as-is
(internally consistent across all trained probes/DIM vectors — don't "fix").

## Colab state at handoff

- Notebook: https://colab.research.google.com/drive/1K4WswU8WFgJD1EbZdI2cdgsQ0FHdA2C5 (Untitled27.ipynb)
- Runtime: G4 = **NVIDIA RTX PRO 6000 Blackwell, 96 GB** (H100 was unavailable). Fits every model incl. Qwen3.8-27B in bf16.
- Cell 1 DONE: repo cloned to `/content/SycoScope` (branch above), `pip install -q -U transformers anthropic` (transformers 5.15.1).
- Cell 2: a SyntaxError'd attempt at inlining the secrets loader — ignore it.
- The runtime will idle-terminate; if it's gone, redo cell 1.

## What to do next (in order)

1. In the notebook (or via colab-mcp `run_code_cell`):
   ```
   !cd /content/SycoScope && git pull -q && echo pulled
   %run /content/SycoScope/tool_calling/tasks/sycophancy/scripts/colab_load_secrets.py
   ```
   Expect `secrets: {'HF_TOKEN': True, 'ANTHROPIC_API_KEY': True, 'GITHUB_TOKEN': ...}`.
   Colab may show a "grant this notebook access to secrets" prompt — the user should approve.
2. Optional pre-flight ON the GPU box (seconds): 
   `!cd /content/SycoScope/tool_calling/tasks/sycophancy/scripts && python verify_model_generation_setup.py`
3. Launch the driver detached (env inherited from the kernel):
   ```
   !cd /content/SycoScope && nohup python tool_calling/tasks/sycophancy/scripts/colab_multimodel_generate.py > /content/gen.log 2>&1 &
   ```
   Defaults: models Qwen3-8B → Nemotron-8B (`--system-prompt 'detailed thinking on'`) → gemma-4-12B-it → Qwen3-14B → Qwen3.8-27B; datasets are_you_sure_mc, are_you_sure_freeform, truthfulqa, sypr; `--n 1000` rows per dataset per model (full sypr is 10.8k rows — too much for one session ×5 models; rerun with `--n 0` later, checkpoints resume); `--max-new-tokens 2048`.
   Outputs: `results/generations/multimodel/<model_slug>/<dataset>/checkpoint.jsonl`; tarball `/content/multimodel_generations.tar.gz` at the end.
4. Poll: `!tail -5 /content/gen.log`. Watch for `WARNING: N/M generations hit the max_new_tokens cap` (raise budget if frequent) and judge errors.
5. Pull results back into the repo. Options: (a) if `GITHUB_TOKEN` secret exists, from the VM
   `git checkout -b multimodel-generations && git add -f results/generations/multimodel && git commit && git push https://$GITHUB_TOKEN@github.com/oscaryas/SycoScope.git multimodel-generations`, then locally fetch + commit into the work branch; (b) otherwise Files pane → download the tarball (needs the user), then untar under `tool_calling/tasks/sycophancy/results/generations/`.
6. Then the separation tests per model: `mixture_residual_probe_pipeline.py --model <id>` etc. (read-probing on existing Llama-generated mixture) and/or build per-model mixtures from the new generations (`build_sycophancy_mixture.py`) — note moral/social have no in-repo generation scripts, dissociating is judge-only, syconbench is fetched.
7. Still queued from earlier in the session: DIM cross-domain transfer run
   `truthfulqa_probe_transfer_pipeline.py --direction-format dim --layer 14 --targets mixture,sypr,social,truthfulqa,are_you_sure,moral,syconbench,dissociating_sycophancy` (needs the data payload: gzipped files + `unpack.sh` sit UNTRACKED in the worktree at `.claude/worktrees/fix-steerer-asymmetries/colab_data/` — `git add` of that payload was blocked by the permission classifier; push it from a normal session or upload the 33 MB tarball via the Files pane).
8. Merge `worktree-fix-steerer-asymmetries` into `sentence-probes-wip`; delete the stale `colab-data-transfer` remote branch from a previous session.

## Browser-automation gotchas (if driving Colab through Chrome again)

- Screenshot pixel coords vs. click coords drift as the window resizes (seen ~1.13× scale); re-screenshot before each precise click and prefer `find` refs where the element is in the accessibility tree (Colab dialogs render in shadow DOM and are NOT).
- The runtime-type dialog radios only toggle when the circle itself is clicked.
- Typing into Monaco cells can drop the first characters — keep cell text to one short line (`!...` shell or `%run script.py`); put logic in committed scripts.
- Use the command palette ("Commands" button) rather than menu coordinates.
