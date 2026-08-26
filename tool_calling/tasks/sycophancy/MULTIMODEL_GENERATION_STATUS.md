# Session 3 (2026-08-26) — second GPU run: gemma-4-12B, Qwen3-14B, Qwen3.8-27B (partial); stopped by user

Branch `worktree-fix-steerer-asymmetries`. **No code changes this session** — the run used the
code pushed in session 2 (`d0377b8`). This section records what was generated, what the numbers
look like, and exactly what remains. Read it before the Session-2 and Session-1 sections below.

## Outcome in one paragraph

A fresh Colab **A100-SXM4 80 GB** VM ran the remaining models as one detached bash chain
(`gemma-4-12B-it --batch-scale 4` → `Qwen3-14B --batch-scale 2` → `Qwen3.8-27B --batch-scale 4`
→ `Qwen3-8B --datasets truthfulqa`, all `--n 200`). **gemma-4-12B-it and Qwen3-14B completed all
four datasets.** Qwen3.8-27B completed are_you_sure_mc and (all but the last batch of)
are_you_sure_freeform; the user stopped the chain there because the 27B runs ~3.4 h per dataset
at batch 8. **Not generated:** Qwen3.8-27B truthfulqa + sypr, and Qwen3-8B truthfulqa. Results
were pulled SHA-verified and committed to git (force-added past `.gitignore`) — see "Where the
data is".

## Run timeline (local time, 2026-08-26)

- colab-mcp reconnect landed on a **CPU** runtime first; the user switched it to A100 and
  reconnected. Setup: clone branch, `pip install -U transformers anthropic` (transformers 5.15.1,
  torch 2.11.0+cu128), `%run colab_load_secrets.py`.
- Session-2 checkpoints were **not** restored onto the VM (the base64 cell payload was too large
  to push through one MCP call); gemma's 54 partial rows were simply regenerated (~10 min).
  Finished models were excluded with `--models`, so nothing else was redone.
- ~2 h: gemma done (rarely hits the 2048 cap). ~4 h 20 min: Qwen3-14B done. ~7 h 40 min: 27B
  are_you_sure_mc done. ~9 h 40 min: user asked to stop after freeform → chain parent and
  driver killed by pid; the freeform generator kept running as an orphan.
- ~10 h 40 min: the 27B freeform generator **crashed on its final batch** (source rows
  193–200) with `anthropic.OverloadedError: 529` from the Claude judge. 123 rows were already
  checkpointed; `checkpoint.resume_state.json` says `n_consumed: 192`, so a rerun finishes the
  last 8 rows. The judges have no retry/backoff — see "What needs to be done".
- Pull: tar+gzip → base64 in a cell → `decode_mm.py` (SHA-verified) → repo.

## Where the data is

`tool_calling/tasks/sycophancy/results/generations/multimodel/<model_slug>/<dataset>/`
(`checkpoint.jsonl`, `checkpoint.truncated.jsonl`, `checkpoint.resume_state.json`,
`summary.json`). Session-2 data (Qwen3-8B, Nemotron-Nano-8B) lives in the same tree. These paths
match `.gitignore` (`results/**/*.jsonl`) and were added with `git add -f`; add future pulls the
same way.

| model / dataset | labeled rows | pos (sycophantic) | neg | truncated set aside |
|---|---|---|---|---|
| gemma-4-12B-it / are_you_sure_mc | 173 | 2 | 171 | 9 |
| gemma-4-12B-it / are_you_sure_freeform | 137 | 22 | 115 | 2 |
| gemma-4-12B-it / truthfulqa | 184 | 57 | 127 | 1 |
| gemma-4-12B-it / sypr | 192 | 21 | 171 | 8 |
| Qwen3-14B / are_you_sure_mc | 104 | 1 | 103 | 78 |
| Qwen3-14B / are_you_sure_freeform | 146 | 6 | 140 | 5 |
| Qwen3-14B / truthfulqa | 179 | 46 | 133 | 12 |
| Qwen3-14B / sypr | 159 | 28 | 131 | 41 |
| Qwen3.8-27B / are_you_sure_mc | 132 | 1 | 131 | 63 |
| Qwen3.8-27B / are_you_sure_freeform | 123 (192/200 consumed) | 12 | 111 | 36 |
| Qwen3.8-27B / truthfulqa | — not run | | | |
| Qwen3.8-27B / sypr | — not run | | | |
| Qwen3-8B / truthfulqa | — not run | | | |

Session-2 rows (Qwen3-8B ×3, Nemotron ×4) are in the Session-2 table below.

## Findings worth keeping

- **gemma-4-12B-it is the fastest and cleanest**: ~55 rows / 20 min at batch 64 including
  judging, <5 % cap-hits. Caves on freeform pushback 16 % (22/137), almost never on MC (2/173);
  31 % imitative falsehoods on truthfulqa; praises poor work 11 % (21/192).
- **Qwen3-14B behaves like Qwen3-8B** (thinking on): MC pushback 1/104, ~45 % of MC
  generations hit the 2048 cap (78 set aside), sypr 18 % (28/159), truthfulqa 26 % (46/179).
  `--batch-scale 2` (batch 32) peaked at 77.5 GB on the 80 GB card — do not raise it.
- **Qwen3.8-27B is slow and memory-bound**: batch 8 sits at 79–80.4 GB, ~3.4 h per 200-row
  dataset, ~48 % cap-hits on MC. MC pushback 1/132; freeform 10 % (12/123).
- Across all five models, MC "are you sure" sycophancy is near zero (0–3 positives per model);
  are_you_sure positives for a balanced mixture have to come from freeform.
- Throughput on A100-80GB at 2048 max-new-tokens: 8B thinking batch 64 ≈ 64 rows / 10 min;
  12B non-thinking batch 64 ≈ 55 rows / 20 min; 14B batch 32 ≈ 30 rows / 20 min; 27B batch 8 ≈
  12–16 rows / 20 min.

## What needs to be done (remaining work)

1. **Finish generation** — all resumable from the checkpoints now in git (clone the branch on
   the VM and the checkpoints are already in place):
   ```
   S=tool_calling/tasks/sycophancy/scripts/colab_multimodel_generate.py
   python $S --models Qwen/Qwen3.8-27B --datasets are_you_sure_freeform --n 200 --batch-scale 4  # last 8 rows, ~15 min incl. load
   python $S --models Qwen/Qwen3-8B    --datasets truthfulqa --n 200 --batch-scale 4             # ~40 min
   python $S --models Qwen/Qwen3.8-27B --datasets truthfulqa,sypr --n 200 --batch-scale 4         # ~7 h; or --n 100, or drop 27B
   ```
2. **Add retry/backoff to the Claude judges.** All four generation scripts call the judge
   directly; a single 529/overloaded response kills the run (it cost the last 27B batch). Wrap
   the `client.messages.create` calls (are_you_sure freeform judge, sypr judge,
   `truthfulqa_verdict_judge.judge_truthful`) with exponential backoff on
   `anthropic.APIStatusError` / `OverloadedError` / `RateLimitError`, or pass `max_retries=8` to
   `anthropic.Anthropic(...)`.
3. **Decide what to do with the truncated rows.** Every `checkpoint.truncated.jsonl` holds
   cap-hit generations (25–50 % for the Qwen3 thinking models). Options: rerun those prompts
   with `--max-new-tokens 4096` (there is no `--resume-truncated` mode yet — it would read the
   sidecar, regenerate, and append to the main checkpoint), or accept the bias toward
   short-thinking prompts.
4. **Per-model mixtures.** `build_sycophancy_mixture.py` needs 4 balanced categories (sypr,
   are_you_sure, social, moral) capped at min(pos, neg). This run yields only sypr + are_you_sure;
   social and moral have no generation scripts. Either write `social_*_generate.py` /
   `moral_*_generate.py` counterparts (same checkpoint/truncated/judge pattern), or build a
   2-category mixture. With `--n 200` the balanced caps are small (are_you_sure ≈ 6–22 pairs,
   sypr ≈ 21–28 pairs per model); raise `--n` for the positive-starved datasets if a bigger probe
   set is needed.
5. **Probe training / steering per model** then follows the Session-1 "What to do next" list
   from step 6 onward.
6. Disconnect the Colab VM after pulling (it was left allocated at the end of this session).

## colab-mcp notes added this session

- After `/mcp` reconnect the notebook can be attached to a **CPU** runtime — check `nvidia-smi`
  before setup and have the user switch to A100 if needed.
- Stopping mid-chain: kill the `bash -c` parent **and** the driver
  (`colab_multimodel_generate.py`) by pid; the per-dataset generator child keeps running as an
  orphan and finishes its checkpoint cleanly.
- The Claude Code job lost macOS access to `~/Desktop` mid-session (EPERM on every file; every
  Bash call rejected). Granting Files-and-Folders access to the host app only applies to newly
  launched processes, so the final decode/commit was done by the user from a normal terminal
  with `decode_mm.py <repo results dir>`.

---

# Session 2 (2026-08-25, later) — first GPU run, stopped by user

Everything below happened on branch `worktree-fix-steerer-asymmetries` (all code pushed).
Read this section first; the Session-1 handoff below it is still accurate for the
code layout and remaining steps.

## Outcome in one paragraph

colab-mcp worked end-to-end this time (recipe below). The driver ran on a fresh
Colab **A100-SXM4 80 GB** VM (not the G4 from session 1). Two models completed all
their datasets (Qwen3-8B minus truthfulqa, Nemotron-Nano-8B all four), gemma-4-12B-it
got partway through are_you_sure_mc, then the user stopped the run at 18:55. Results
were pulled back SHA-verified into
`tool_calling/tasks/sycophancy/results/generations/multimodel/` (gitignored — see
"Where the data is"). Four real bugs/gaps were found and fixed along the way.

## Commits this session

| Commit | What |
|---|---|
| `bede1d6` | `colab_multimodel_generate.py --batch-scale N` multiplies each model's per-model generation batch size. |
| `7604300` | **Real bug**: none of the four generation scripts forwarded `--generation-batch-size` to `ActivationSteerer.generate_batch`, which chunks internally with its own default `batch_size=8`. The GPU batch was therefore always 8 regardless of the flag (the flag only set checkpoint granularity). Fixed in all 6 call sites. Also dropped the Qwen3.8-27B base batch to 2 (64 layers × head_dim 256 KV cache next to 54 GB of weights). |
| `62929b6` | Cap-hit (INCOMPLETE) generations are now **set aside** instead of silently discarded: `generate_batch` records `steerer.last_truncated` (per-row flags); each script writes those rows (prompt, response, stage, source fields) to `checkpoint.truncated.jsonl` next to `checkpoint.jsonl`, excludes them from the Claude judges and the labeled checkpoint, and logs `N cap-hit (set aside)` per batch. Before this, a truncated turn-1 counted as "ineligible (wrong turn1)" / judge-UNCLEAR. |
| `5f17a05` | `truthfulqa_verdict_judge.py` was **untracked in git** (existed only in the main checkout), so the truthfulqa dataset failed on Colab with `ModuleNotFoundError`. Brought into git and made reasoning-aware (`strip_reasoning` before judging, `UNCLEAR` on an empty visible answer — same pattern as the other four judges). |

## Run timeline (Colab, local time)

- 15:09 first launch (default batch 16) — lost ~20 min later when an MCP reconnect attached to a **new notebook/VM**; nothing had been checkpointed.
- 15:33 relaunch on the new VM, default settings. First 16-row batch took ~10 min → projected 200+ h for the full matrix.
- 15:47 relaunch with `--batch-scale 4 --n 200`. No speedup, GPU memory flat at 26 GB → found the batch-size passthrough bug.
- 16:06 relaunch on `7604300`: 64-row batches, GPU 47–73 GB, ~4× throughput (≈64 rows / 10 min for 8B thinking models).
- 16:36 log showed ~50 % of every batch hitting the 2048-token cap → user chose to keep running but **save the truncated rows aside** → `62929b6`, relaunch 16:48 (resumed from checkpoints).
- 17:01 Qwen3-8B / truthfulqa failed on the missing judge module → `5f17a05`; a waiter process was queued to rerun `--datasets truthfulqa` after the main driver (never fired — run stopped first).
- 17:29 Qwen3-8B done; 18:38 Nemotron done; gemma-4-12B-it started.
- 18:55 user stopped the run; all processes killed by pid; tarball pulled.

## Where the data is

`tool_calling/tasks/sycophancy/results/generations/multimodel/<model_slug>/<dataset>/`
in the **worktree** `.claude/worktrees/fix-steerer-asymmetries` — `checkpoint.jsonl`,
`checkpoint.truncated.jsonl`, `checkpoint.resume_state.json`, `summary.json`.
These paths match `.gitignore` (`results/**/*.jsonl`), so they are NOT in git; `git add -f`
them (or copy them out of the worktree) before the worktree is deleted.

| model / dataset | labeled rows | pos (sycophantic) | neg | truncated set aside |
|---|---|---|---|---|
| Qwen3-8B / are_you_sure_mc | 85 | 0 | 85 | (predates sidecar) |
| Qwen3-8B / are_you_sure_freeform | 123 | 12 | 111 | 5 (last 72 rows only) |
| Qwen3-8B / sypr | 154 | 26 | 128 | 46 |
| Qwen3-8B / truthfulqa | — not run (judge missing) | | | |
| Nemotron-Nano-8B / are_you_sure_mc | 98 | 3 | 95 | 70 |
| Nemotron-Nano-8B / are_you_sure_freeform | 70 | 8 | 62 | 1 |
| Nemotron-Nano-8B / truthfulqa | 177 | 111 | 66 | 1 |
| Nemotron-Nano-8B / sypr | 171 | 26 | 145 | 29 |
| gemma-4-12B-it / are_you_sure_mc | 54 (partial) | — | — | 4 |

Not started: gemma freeform/truthfulqa/sypr, Qwen3-14B, Qwen3.8-27B. All resumable.

## Findings worth keeping

- **Qwen3-8B with thinking on is very pushback-resistant**: never caves on MC after
  pushback (0/85), caves on ~10 % freeform, praises poor work ~17 %. Roughly 25–50 %
  of its generations hit the 2048 cap (the surviving rows before `62929b6` are therefore
  biased toward short-thinking questions).
- **Nemotron's reasoning is optional per query**: with `detailed thinking on` it emits
  `<think>` on ~half of MC (math) rows and 66/70 of the MC cap-hits, but on **0/70**
  freeform trivia rows — those are effectively non-thinking generations. Its truthfulqa
  positives dominate (111/177).
- Consequence for per-model mixtures: the balanced cap is `min(pos, neg)` per category,
  so at `--n 200` you get ~12 balanced are_you_sure pairs and ~25 sypr pairs for Qwen3-8B.
  Either raise `--n` selectively (checkpoints resume; ~2.5 h per 1000 rows per dataset for
  an 8B thinking model at batch 64) or train on the natural skew with class weights.
- The mixture (`build_sycophancy_mixture.py`) needs 4 categories (sypr, are_you_sure,
  social, moral). This run only produces the first two; social/moral have no in-repo
  generation scripts, so a true per-model 4-way mixture needs new scripts.
- GPU memory at batch 64 peaked at **73 GB** on an 8B model (long thinking traces).
  Qwen3-14B at `--batch-scale 4` is a real OOM risk — use `--batch-scale 2` for it.

## colab-mcp recipe that worked (and its failure modes)

1. User runs `/mcp` → reconnect colab-mcp with the Colab tab open.
2. Call `open_colab_browser_connection` → then `ToolSearch "+colab"` lists all 7 tools
   (`run_code_cell`, `add_code_cell`, `update_cell`, `get_cells`, …).
3. Setup cell: `git clone -b worktree-fix-steerer-asymmetries …`, `pip install -q -U
   transformers anthropic`, then `get_ipython().run_line_magic("run", ".../colab_load_secrets.py")`
   (HF_TOKEN + ANTHROPIC_API_KEY present; GITHUB_TOKEN secret is missing).
4. Launch the driver as `subprocess.Popen(..., start_new_session=True)` with stdout to
   `/content/gen.log`; poll with a **no-sleep** read-only cell (sleeps inside cells caused
   a 900 s MCP timeout once).
5. Pull results: tar+gzip → base64 printed in a cell → lands in a local tool-results file
   → decode with a script (strip literal `\n` escapes first) → verify SHA → extract.
   Used `$CLAUDE_JOB_DIR/tmp/decode_mm.py`.

Failure modes seen: the server silently drops its notebook tools after any rejected or
timed-out call (`Unknown tool: update_cell`) — re-run steps 1–2. Once, the reconnect
attached to a **brand-new notebook/VM** and the old runtime (and its process) was lost.
`pkill -f '<pattern>'` from a cell kills the cell's own `/bin/sh -c` wrapper first (the
pattern is in its own command line) and leaves the target alive — kill by pid from `ps`.

## Remaining next steps

1. Decide durability of the pulled data (`git add -f` onto the branch, or copy out).
2. Resume: redo setup cell, restore `multimodel/` onto the VM, relaunch
   `colab_multimodel_generate.py --batch-scale 4 --n 200` (use `--batch-scale 2` for
   Qwen3-14B; 27B already runs at 8). Then `--datasets truthfulqa` for Qwen3-8B.
3. Optionally raise `--n` for the positive-starved datasets.
4. Everything in the Session-1 "What to do next" list from step 6 onward still applies.
5. The Colab VM was left allocated (idle) — disconnect it from the Colab UI.

---

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
