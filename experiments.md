# Experiments

Overview of the experimental tooling built in this session: a TopK sparse autoencoder
training package, and an end-to-end sycophancy-probing pipeline (labels → activations →
probes → steering), both exposed through the `tool_calling` agent harness. Nothing here
has been committed — see `git status` for the current uncommitted diff (per explicit
instruction: no pushing).

## 1. TopK sparse autoencoder training (`SAE/sae/`)

- `model.py` — `TopKSAE`: hard top-k over encoder pre-activations, decoder columns
  renormalized to unit norm after every optimizer step, auxk loss to keep features alive.
- `data.py` — `load_cache_meta`, `ActivationSplit`: reads the frozen activation cache
  produced by `SAE/pipeline/cache_activations.py` (`sentences.parquet`/`responses.parquet`
  + per-layer memmaps), splits by `response_id` so sentences from one response never
  straddle train/val.
- `train.py` — `run_training`/`evaluate`/`make_run_id`: resumable training loop (manifest +
  checkpoint), one SAE per layer, `--granularity {sentence,response}`. Logs FVU, L0, and
  dead-feature fraction to `metrics.jsonl` and a shared `sweep_log.jsonl` across runs.
- Exposed as an agent-drivable `tool_calling` task (`tool_calling/tasks/sae/tools.py`):
  `list_activation_cache`, `train_sae`, `read_sweep_log`, `write_analysis` — lets an LLM
  agent iteratively sweep `(dict_size, k)` configs and reason about the FVU/L0/dead-feature
  tradeoff without touching the model directly.

## 2. Sycophancy probing task (`tool_calling/tasks/sycophancy/`)

Ported from a sibling repo (`autointerp`) where this pattern originates, then fixed for
this repo's flat layout (no `src/` layer) — path math, missing dependency modules
(`gpu_memory.py`, `sycophancy_data.py`, `sycophancy_model_registry.py`,
`sycophancy_probes.py`, `sycophancy_compare.py`), and `run_agent.py`'s `repo_root`
resolution were all broken on arrival.

**Labeling sources** (both call Claude as judge, `thinking={"type": "disabled"}` since
this is one-token classification, judge calls parallelized via `ThreadPoolExecutor` —
~7-8x speedup at n=500 vs. sequential):

- **Moral sycophancy** (`moral_sycophancy_judge.py`) — ELEPHANT-paper style: judges
  YTA/NTA verdict on both the `original_post` and `flipped_story` framing of the same AITA
  conflict (`SAE/results/AITA-NTA-FLIP.jsonl`). Moral sycophancy = both sides judged NTA.
- **Social sycophancy** (`social_sycophancy_judge.py`) — ELEPHANT's other three metrics:
  validation, indirectness, framing. Single-response judging, no pairing needed, over
  open-ended advice prompts (`OEQ.jsonl`/`SS.jsonl`).

**Probes** — linear probes (`nn.Linear` + `BCEWithLogitsLoss`, Adam lr=0.001, 25 epochs,
batch size 25 — matching ELEPHANT's paper hyperparameters) trained separately over MHA
(per-head), MLP, and residual-stream activations at every layer.

- `train_probe` evaluates with 5-fold stratified cross-validation (every example held out
  exactly once, Wilson-score CI on pooled predictions, final probe refit on all data for
  the steering direction).
- Class imbalance: the majority class is undersampled down to `min(n_pos, n_neg)` before
  CV, so a skewed input (e.g. 80 positive vs. 910 negative) doesn't get trivially "solved"
  by the probe just predicting the majority class. (A class-weighted loss and a separate
  `balanced_accuracy` metric were tried first and rolled back in favor of this simpler,
  single fix.)

**Steering** (`sycophancy_steering.py`, `ActivationSteerer`) — forward hooks matched
exactly to where activations were extracted for probing: a pre-hook on `self_attn.o_proj`
(per-head slice) for MHA, a post-hook on `mlp.down_proj` for MLP, a post-hook on the whole
decoder-layer module for the residual stream. Direction = unit-norm probe weight vector,
scaled by its training-set projection std (`alpha=1.0` ≈ "shift by ~1 std").

## 3. Notebooks (`tool_calling/tasks/sycophancy/notebooks/`)

- **`moral_sycophancy_probes.ipynb`** — repo-cloned Colab notebook (needs the repo on the
  Colab filesystem).
- **`moral_sycophancy_probes_colab_standalone.ipynb`** — fully self-contained version (no
  repo clone; every helper inlined, data/config uploaded directly) — the primary iteration
  target this session, built after an explicit request for a standalone notebook instead
  of the repo-dependent one. `LABEL_SOURCE` toggles between `"moral"` and `"social"`.
  - Probe-accuracy-by-layer plot now shows CV error bars (std across the 5 folds); for
    MHA, the error bar comes from the specific best head chosen at that layer, not an
    aggregate across heads.
  - The steering section no longer just prints one before/after example — it measures the
    actual sycophancy **rate** (moral: fraction of held-out AITA pairs both judged NTA;
    social: fraction of held-out responses judged 1 for `SOCIAL_METRIC`) on held-out data
    (sliced right after what was used for training, so the probe hasn't seen it), swept
    across `STEER_ALPHAS = [-100 .. 100]` (13 points, denser near 0), and plotted as a bar
    chart — one bar per alpha, `alpha=0.0` is the true unsteered baseline.
  - The steering bar chart shows **change vs. the alpha=0.0 baseline** (not the raw rate)
    — every bar is a delta, so the unsteered baseline bar is always exactly zero and every
    other bar directly shows how much that steering strength moved the needle.
  - Data files (`DATA_PATH`) are fetched automatically via `!wget` from the main repo's
    `SAE/results/` on GitHub (public repo), falling back to the manual upload widget only
    if the download fails — no manual upload required in the common case.
  - A **cross-dataset generalization check** ("5b") reuses the exact direction found above
    and measures its effect (the same alpha sweep) against held-out examples from all 5
    available datasets (`AITA-NTA-FLIP`, `AITA-NTA-OG`, `AITA-YTA`, `OEQ`, `SS`), not just
    the one it was found from — `AITA-NTA-OG`/`AITA-YTA` are single-response AITA posts (no
    flip pairing), judged for a single NTA verdict; the others use the same judging as
    their main-section counterparts. Plotted as one line per dataset (delta vs. alpha).
- **`steering_sweep_social_sycophancy.ipynb`** — separate notebook: loads probe weights
  saved from a moral-sycophancy training run, discovers ~96 directions (MLP + residual at
  every layer, plus the single best-accuracy MHA head per layer), and for each direction
  measures whether steering increases social sycophancy (validation/indirectness/framing)
  on held-out advice prompts relative to an unsteered baseline — a direct test of whether
  the moral-sycophancy direction generalizes or is narrow to AITA verdicts. Also re-judges
  NTA/YTA on held-out AITA pairs under steering, as an in-domain sanity check that the
  direction still does what it was trained to do. Ranks directions by transfer effect and
  plots probe accuracy vs. transfer delta.
  - `TEST_PROMPTS`/`MORAL_TEST_PAIRS` are built from real, downloaded dataset files (same
    `iter_dataset_records`/`iter_flip_pairs` helpers as the probing notebooks) instead of
    hardcoded example strings — every response (baseline and steered) is generated fresh,
    autoregressively, from just the prompt, never read from a dataset's cached `response`.
  - Extended with the same **cross-dataset generalization check** as the DIM notebook:
    for every one of the ~96 directions, also tests against small held-out slices of
    `AITA-NTA-OG`, `AITA-YTA`, and whichever of `OEQ`/`SS` isn't already the primary
    `SOCIAL_DATASET` — a much smaller per-dataset cap (`CROSS_DATASET_EVAL_MAX`) than the
    main checks, since this multiplies across all ~96 directions.

## 4. Difference-in-means (DIM) notebook

**`moral_sycophancy_dim_colab_standalone.ipynb`** — a separate, lean notebook offering an
alternative to linear-probe training: **difference-in-means**
(`direction = mean(activations | label=1) - mean(activations | label=0)`, unit-normalized),
no gradient descent at all. Built as a copy-then-edit of the probing notebook — everything
except direction-finding (activation extraction, both judges, `ActivationSteerer`, the
steering/alpha-sweep/cross-dataset sections) is reused verbatim, since `ActivationSteerer`
only needs a raw direction vector + a scale and has no dependency on how that vector was
derived.

- `DIM_METHOD = "naive" | "cv_averaged"`: `"naive"` computes the mean difference once on
  the whole dataset; `"cv_averaged"` (default) computes it per 5-fold stratified CV split
  and averages the resulting unit directions for a more robust estimate — reuses the same
  `_stratified_folds` helper the probe pipeline uses for its own CV.
- No held-out classifier accuracy (there's no classifier) — directions are ranked by
  **Cohen's d** instead, which is *signed* (unlike bounded-[0,1] accuracy), so every
  "pick the best direction" comparison selects by `abs(effect_size)`, not the raw value.
  Verified this matters in practice: `cv_averaged`'s held-out per-fold effect sizes can
  legitimately go negative (a fold's direction not generalizing to noise), confirmed on
  synthetic pure-noise data.
- Otherwise identical downstream behavior to the probe notebook: CV-error-bar-style plot
  (Cohen's d by layer instead of accuracy), delta-vs-baseline alpha sweep, wget-based data
  download, and the cross-dataset generalization check.

## 5. Notebook enhancements and real experimental runs (2026-07-22)

Prior sessions built and mock-verified the DIM notebooks but never actually ran them on
Colab. This session added three features to the DIM pipeline, then ran three real
experiments end-to-end and organized the results for browsing.

**New notebook features**:
- **AUC-ROC** computed per-fold alongside Cohen's d in `compute_dim_direction`, plotted as a
  second by-layer chart. Purely diagnostic — direction selection stays `abs(cohen_d)`-based.
- **Early/middle/late layer-bucket cross-dataset check** — splits the 32 layers into thirds,
  selects the top-3 MHA directions per third by `abs(cohen_d)` (9 directions total), and
  runs the existing 5-dataset generalization check on each instead of just the single best.
- **`moral_sycophancy_dim_rowid_pooled_colab_standalone.ipynb`** — a pooling variant that
  averages every generation belonging to one AITA conflict (both `original_post`/
  `flipped_story` sides, all 3 samples each) into a single activation vector per
  row_id/conflict, rather than treating the two sides as separate labeled examples sharing
  a label. Confirmed against real data that every `(row_id, prompt_col)` group has exactly
  3 samples, so this is a genuine 6-generations-to-1-vector pooling, not a no-op.
- **Batched generation** (`ActivationSteerer.generate_batch`) replacing one-prompt-at-a-time
  calls in every steering-eval loop — left-padded, chunked by `GENERATION_BATCH_SIZE`,
  verified batch-size-invariant against the real test model before use.
- **`generations.jsonl` logging** — every steering-eval loop now saves the full raw
  prompt/completion/verdict/alpha/direction for every generation, not just aggregate rates.

**Three real runs**, each pulled locally to
`tool_calling/tasks/sycophancy/results/<run_name>/` (a `SUMMARY.md` with best directions,
full per-alpha cross-dataset tables, and representative examples; a `plots/` folder of
locally-regenerated PNGs; raw `generations.jsonl`; underlying `results.json`/`.pt` vectors):

1. **`moral_sycophancy_dim_rowid_pooled_n100/`** — row-id-pooled DIM at `N_EXAMPLES=100`.
   Best direction: MHA layer 13/head 23 (Cohen's d=1.403). The original "steering effect is
   in early layers" hypothesis was not confirmed — effects cluster in the middle third,
   though early layers aren't far behind and late layers are consistently negative-signed.
   Steering raises the home-dataset moral sycophancy rate from 41.7% to 71.4% at α=+20;
   cross-dataset generalization is real but uneven across the other 4 datasets.
2. **`dim_truth_generalization/`** — do the AITA-derived directions generalize to MMLU
   (accuracy) and TruthfulQA (imitative-falsehood rate)? MMLU's first pass was misleading
   (see Known limitations in `changelog.md`); after the fix, steering shows a modest,
   mostly *positive* effect on MMLU accuracy (no evidence it hurts capability) and a real
   but modest positive effect on TruthfulQA's imitative-falsehood rate for middle/late-bucket
   directions specifically.
3. **`truthfulqa_prefill_dim_moral_generalization/`** — the reverse experiment: a direction
   found purely from TruthfulQA's own correct/incorrect answers via contrastive prefill (no
   LLM judge needed for direction-finding), tested against moral/social sycophancy. Best
   direction is at a different layer (MHA layer 28, late) than the AITA-derived best (layer
   13), and its effect on AITA/OEQ/SS is small and non-monotonic — the relationship between
   truthfulness and moral sycophancy looks asymmetric, not a shared bidirectional direction.

**Browsable artifact**: a single self-contained HTML page (published via the Artifact tool)
covering all three runs — per-layer Cohen's d/AUC-ROC charts, steering-sweep and
cross-dataset charts, bucket comparison, the MMLU/TruthfulQA section, a paired generation
browser (every prompt shown across all 5 alphas side by side, filterable by
section/dataset/direction), and a curated chat-bubble-styled section with 12 hand-picked
baseline-vs-steered examples spanning AITA/OEQ/SS/TruthfulQA and all three layer buckets.
Built with a custom SVG chart renderer (no external charting library, to satisfy the
Artifact CSP) and a validated colorblind-safe categorical palette.

## 6. SyPR probe-based sycophantic-praise vector (2026-07-28)

**`sypr_probes_colab_standalone.ipynb`** — a new dataset/direction-finding pipeline, distinct
from every earlier notebook in that its source dataset (`vennemeyerd/sycophantic-praise`, the
"SyPR" benchmark, 13,200 rows: persona-calibrated utterances across 5 domains — `gsm8k`,
`mmlu_chemistry`, `mmlu_economics`, `long_form_moral_reasoning`, `pseudo_profundity` — each
with a ground-truth quality label) ships **no model responses of its own**. The pipeline:

1. Stratified-sample rows (24 per domain, split further by label-eligible quality bucket —
   reasoning: correct/incorrect; social: low/high, "medium" excluded as ambiguous).
2. Generate Llama-3-8B-Instruct's own evaluation of each utterance (batched, left-padded).
3. Judge whether the response **praises/validates** the utterance (`sycophantic_praise_judge.py`
   — narrower than the existing moral/social judges, no ground-truth awareness of its own).
4. Combine with SyPR's own ground-truth quality field: `sycophantic = 1` iff praised **and**
   the utterance is actually poor-quality.
5. Cache activations, train linear probes (MHA per-head/MLP/residual, every layer), pick the
   best-separating direction, steer, test generalization (in-domain held-out SyPR rows + the
   same 5-dataset cross-dataset check used throughout this session).

**Two balance-handling variants**, trained from the *same* cached activations (no
re-extraction) via a new `train_probe(..., balance_method=...)` parameter:
- `"undersample"` — existing default, majority class dropped to `min(n_pos, n_neg)`.
- `"upweight"` — new, keeps every example, corrects imbalance only via a `pos_weight`-scaled
  `BCEWithLogitsLoss` (already present in `_fit_probe`, just never previously reachable
  without also undersampling first).

Real class balance on the n=120 labeled sample: **12 sycophantic / 108 non-sycophantic
(10.0%/90.0%)** — confirmed the imbalance the user asked to check for. Results in
`results/sypr_probe_undersample/` and `results/sypr_probe_upweight/`; see their `SUMMARY.md`
files for the full numbers, and both files' "Known limitations" section for a real
data-transfer constraint this session hit (no Colab-to-local file-download channel was
available, so `generations.jsonl` is a representative excerpt and `mha_probe_vectors.pt`
holds only the single selected direction, not the full per-layer/head sweep).

## 7. OEQ probe-based sycophancy vector, script-driven (2026-07-29)

**`tool_calling/tasks/sycophancy/oeq_probe_pipeline.py`** — the first real run this session
built and executed as a plain Python script instead of a Colab notebook. Sourced from the
user's original request to find a "purely sycophancy" direction from the OEQ dataset (an
earlier message accidentally linked an unrelated external dataset instead, which became the
SyPR run above — this closes the original loop using OEQ itself).

Label source: `social_sycophancy_judge.generate_social_sycophancy_labels(metric="validation")`
against `SAE/results/OEQ.jsonl`'s already-cached generations — no fresh model generation
needed for labeling (unlike SyPR, which shipped no responses of its own). "Validation"
(ELEPHANT's own definition: does the response emotionally validate the user, regardless of
whether validation is warranted) is the closest existing single-response, judge-free-of-
correctness-context signal to "purely sycophancy" already built in this repo.

Pipeline (all reusing existing tracked functions, no notebook-only helpers): extract
activations once via `collect_activations` -> train both `train_probe(..., balance_method=...)`
variants (`undersample`/`upweight`) off the same cached activations -> `save_probe_results`
(full per-layer/head `.pth`/`.pt` checkpoints, the standard tracked format) -> pick each
variant's best MHA direction from `probe_metadata.json` -> `load_steering_vectors` ->
`ActivationSteerer` cross-dataset generalization sweep (alpha in [-20,-5,0,5,20], the same 5
datasets used throughout this session, with the `OEQ` entry deliberately using a held-out
row slice disjoint from the labeling sample) -> write `SUMMARY.md`/`results.json`/full
`generations.jsonl`/plots directly to disk.

Code pushed to Colab via direct Google Drive upload (`create_file` with
`disableConversionToGoogleType=true`) instead of `%%writefile` cells, then copied into
`/content/repo/...` with a single `cp`/`mkdir` cell — since none of these `.py` modules are
committed to GitHub yet, this was the only way to get real code onto Colab without retyping
it into cells by hand. Data files (`SAE/results/*.jsonl`) were still fetched via `!wget` from
GitHub raw URLs, since those *are* already committed/pushed and are 16-36MB each — far too
large to push through the same Drive-upload-as-text path efficiently.

Real class balance on the n=150 labeled sample: **143 validating / 5 not-validating (96.6% /
3.4%)** — an even more extreme imbalance than SyPR's, in the opposite direction. Both balance
variants trained; results in `results/oeq_probe_undersample/` and `results/oeq_probe_upweight/`
— see their `SUMMARY.md` files for full cross-dataset numbers. Headline finding: generalization
is weak and largely non-monotonic for both variants, and notably the *in-domain* OEQ check
(does the direction even move validation rate on OEQ's own held-out rows) is close to flat for
both — read as a low-power result driven by the tiny 5-example negative class, not evidence
against a "validation" direction existing at all.

## Key methodology notes

- `thinking={"type": "disabled"}` is required for one-token judge classification on
  Sonnet 5 — adaptive thinking runs by default otherwise (the old fixed `budget_tokens`
  knob no longer exists on this model family).
- Judge response parsing is ThinkingBlock-safe: `next((b.text for b in msg.content if
  hasattr(b, "text")), "")` — some responses emit a thinking block before the text block,
  so `content[0].text` isn't reliable.
- Every notebook edit was validated the same way: JSON parse → per-cell `compile()` syntax
  check (skipping magic lines) → full mocked end-to-end execution (fake Anthropic client +
  stub `ActivationSteerer`, or the real `hf-internal-testing/tiny-random-LlamaForCausalLM`
  test model), run under `MPLBACKEND=Agg` to avoid `plt.show()` blocking on a real window.
- For cells with `!wget`-style shell-magic lines nested inside `if`/`for` blocks, the syntax
  check substitutes `pass` at the same indentation rather than dropping the line entirely —
  removing it outright can leave an empty block body and a false-positive `IndentationError`.
- A near-zero measured effect can be a methodology artifact, not a real null result — MMLU's
  original `max_new_tokens=8` cap left no room for steering to act before generation
  truncated, which looked like "no effect" until generation length was fixed. Always check
  whether a null/near-null result could be a measurement ceiling before concluding the
  underlying effect is actually absent.
- Python's `str(float)` and JavaScript's `String(float)` disagree on whole numbers
  (`"-20.0"` vs. `"-20"`) — matters whenever Python-generated JSON keys are looked up by a
  JS-computed key built from the same numbers; format both sides the same way explicitly
  rather than relying on default stringification.
- `"truthful_qa"` (bare HF dataset ID) fails on current `datasets`/`huggingface_hub`
  versions — use the namespaced `"truthfulqa/truthful_qa"` instead.
- When driving a long-running Colab cell via an agent, prefer a single blocking
  `run_code_cell` call over a separate "check progress" cell submitted in parallel — Jupyter
  kernels execute cells sequentially, so a lightweight diagnostic cell just queues behind
  the long-running one and returns nothing until it finishes anyway.
- For periodic status checks on a long-running background job, prefer a cheap read-only
  check (e.g. `get_cells` on a notebook's tail) over resuming a full agent/fork each time —
  resuming a fork that inherited a long conversation's context costs hundreds of thousands
  of tokens per resume, which is far too expensive to do on a 5-minute cadence.
- **Never re-invoke `run_code_cell` on a cell that's already executing** — it restarts the
  cell from scratch rather than reattaching to the in-flight execution (confirmed by
  re-printed section headers producing different numbers on each "restart" in the SyPR run).
  If a cell is genuinely just slow, poll with `get_cells` only until it finishes.
- `get_cells` can show byte-identical stale output for 30-45+ minutes while the underlying
  cell is actually still running and later completes successfully in full — apparent silence
  is not proof of a hang. Confirm via an independent signal (e.g. a small new diagnostic cell
  actually producing fresh output) before concluding a kernel is stuck.
- This session's `colab-mcp` server had no file-download channel back to the driving agent —
  only cell-output text, and `get_cells` responses over ~85KB get truncated to a file the
  agent must then read locally. For results with large binary artifacts (probe weight
  checkpoints) or long raw logs (`generations.jsonl` in the tens of thousands of rows),
  either keep the transferred summary intentionally small (print only what's needed: best
  direction vectors, per-layer accuracy curves, a handful of representative examples) or
  budget for many chunked print/reassemble round-trips — don't assume a `files.download()`
  call actually gets bytes back to the agent.
- Once Google Drive is mounted in a Colab session, pushing local code to it directly via the
  Drive API's `create_file` (`textContent` + `disableConversionToGoogleType=true` to keep a
  `.py`/text file plain instead of being converted to a Google Doc) is more reliable than
  retyping file contents into `%%writefile` cells one at a time — and results can be pulled
  back the same way: have Colab `zip` the whole output directory on Drive, then download that
  single zip file rather than many small files (hand-copying several small binaries as inline
  text was found to silently corrupt them in the SyPR run above — bad CRC, same file size —
  whereas the zip-then-single-download path was confirmed CRC-clean).
- A cell that runs `subprocess.run(..., capture_output=True)` shows **zero** incremental
  output via `get_cells` for the cell's entire duration — the child process's stdout/stderr
  are fully buffered and only reach the parent cell's own output after the subprocess exits.
  This is a stronger case than the already-documented "stale output" issue: there is no
  partial signal to read at all, not even old text. When driving a long external script this
  way, arrange for an independent out-of-band progress signal instead (e.g. have the script
  itself write intermediate artifacts to a mounted Drive folder, and poll Drive directly,
  bypassing the Colab kernel entirely) rather than trying to infer progress from the cell.
