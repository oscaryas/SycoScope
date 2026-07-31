# Changelog

All notable changes made in this session, grouped by kind. Nothing here has been
committed — see `git status` for the current uncommitted diff (per explicit instruction:
no pushing). See `experiments.md` for what exists now, `session_log.md`/`session_log_2026-07-22.md`
for the narrative of why things changed in the order they did.

## [Unreleased] — 2026-07-29

### Added
- **`tool_calling/tasks/sycophancy/oeq_probe_pipeline.py`** — new standalone Python *script*
  (not a notebook) implementing an end-to-end probe-based sycophancy-direction pipeline
  sourced from `SAE/results/OEQ.jsonl`'s cached generations, labeled via
  `social_sycophancy_judge`'s existing "validation" metric (no fresh judge needed) — the
  closest existing "purely sycophancy" signal in this repo (does the response emotionally
  validate the user regardless of whether validation is warranted). Deliberate architecture
  change from every prior run this session: driven as a single blocking
  `python oeq_probe_pipeline.py` call instead of hand-built interactive Colab cells, writing
  every output straight to disk — motivated by repeated notebook pain points on the SyPR run
  (re-invoking `run_code_cell` on an in-flight cell restarts it from scratch, `get_cells`
  showing stale output for 30+ minutes, and cell-output-size caps forcing a truncated
  `generations.jsonl`/single-direction `.pt` file).
- `ActivationSteerer.generate_batch` ported into the tracked `sycophancy_steering.py` module
  (left-padded, greedy, chunked) — previously only existed inline inside the SyPR notebook.
- Code dependencies for this run (the script itself plus `sycophancy_steering.py`,
  `sycophancy_model_registry.py`, `sycophancy_probes.py`, `social_sycophancy_judge.py`,
  `moral_sycophancy_judge.py`, `utils/model.py`, `utils/inference.py`) were pushed to the
  Colab session via a direct Google Drive upload (`create_file` with
  `disableConversionToGoogleType=true`) rather than typed into `%%writefile` cells — avoids
  an entire class of interactive-cell risk, at the cost of a rename step for the two `utils/`
  files (Drive doesn't nest folders the same way the local repo does).
- Two real Colab runs: `oeq_probe_undersample/` and `oeq_probe_upweight/` under
  `tool_calling/tasks/sycophancy/results/`, `n_label=150`, `cross_dataset_n=30`, real model
  (`meta-llama/Meta-Llama-3-8B-Instruct`), real Anthropic judge — full un-truncated
  `generations.jsonl` (900 rows each) and full per-(layer,head)/per-layer `.pth`/`.pt` probe
  checkpoints (1024 MHA keys, 32 MLP/residual keys each), unlike the SyPR run's
  Colab-output-cap-truncated artifacts.

### Findings
- Class balance on OEQ's "validation" metric at n=150 labeled: **143 validating / 5
  not-validating (96.6% / 3.4%)** — far more extreme than SyPR's 10%/90% imbalance, and in
  the opposite direction: this model validates almost every open-ended advice response, so
  "does it withhold validation" is the rare event, not "does it validate."
- Best directions: `undersample` — MHA layer 31/head 24, accuracy 0.900 (CI [0.60, 0.98],
  reflecting the tiny 5-pos/5-neg undersampled training set); `upweight` — MHA layer 6/head
  14, accuracy 0.973 (full 148-example set, `pos_weight` correction).
- Cross-dataset generalization is weak and largely non-monotonic for both variants — no
  direction here reproduces the clean, strong, monotonic effect the AITA-derived DIM
  direction showed earlier this session. The clearest partial signal: `undersample`'s SS
  entry moves monotonically with alpha (+3.33% to +10.00%), and its AITA-YTA entry moves
  positively at small alphas (+11.90% to +13.64%) though drops back to 0.00% at the extreme.
- Most notable: the **in-domain check itself is nearly flat for both variants**
  (`undersample` OEQ held-out: 0.00%, -0.46%, +3.33%, 0.00%; `upweight`: -3.33%, -3.33%,
  0.00%, -3.33%) — a direction that barely moves its own training-domain metric on held-out
  data is a weak direction, not just a non-generalizing one. Given the underlying label pool
  has only 5 negative examples total, this reads as a genuine low-power result rather than
  evidence the "validation" signal has no real direction at all.

### Fixed
- Sequential (non-thread-pooled) judge calls inside the cross-dataset sweep
  (`_judge_single_rate`/`judge_flip_pairs_rate` in `oeq_probe_pipeline.py`) — unlike the
  labeling step, which reuses `social_sycophancy_judge`'s existing
  `ThreadPoolExecutor`-parallelized path, the sweep's ~1,800 judge calls ran one at a time,
  plausibly the dominant cost in this run's multi-hour wall-clock time. Not fixed this
  session — flagged as the clear next optimization for any future script-based run.

### Known limitations (flagged, not fixed)
- Same sequential-judging performance gap noted above — a straightforward
  `ThreadPoolExecutor` wrap of the per-alpha judge loop would likely cut wall-clock time
  substantially.
- `get_cells` on a cell using `subprocess.run(..., capture_output=True)` shows **no**
  incremental output at all until the subprocess exits — a stronger version of the
  already-documented "stale output" issue: here there's no partial signal whatsoever, not
  even stale text. Confirming progress on this kind of cell requires an independent,
  non-invasive channel (this run used direct Google Drive file-timestamp checks, since the
  pipeline itself writes each variant's probe-training artifacts to Drive before starting the
  slower generalization sweep).

## [Unreleased] — 2026-07-28

### Added
- **`tool_calling/tasks/sycophancy/notebooks/sypr_probes_colab_standalone.ipynb`** — new
  standalone Colab notebook training a linear-probe sycophancy direction from a new external
  dataset, `vennemeyerd/sycophantic-praise` (the "SyPR" benchmark, 13,200 rows: persona-
  calibrated utterances across 5 domains with a ground-truth quality label, but no model
  responses of its own). Pipeline: stratified-sample rows -> generate Llama-3-8B-Instruct's
  own evaluation of each utterance -> judge whether the response praises it -> combine with
  SyPR's ground-truth quality to label `sycophantic = praises an utterance that is actually
  poor-quality` -> cache activations -> train probes -> steer -> test generalization.
- **`tool_calling/tasks/sycophancy/sycophantic_praise_judge.py`** — new single-response LLM
  judge (Claude, same conventions as `social_sycophancy_judge.py`) answering one narrower
  question than the existing judges: does a response praise/validate the thing it's
  evaluating, regardless of whether that praise is warranted.
- **`tool_calling/tasks/sycophancy/sypr_data.py`** — dataset loading/stratified-sampling/
  chat-prompt-building helpers for the SyPR dataset (`load_sypr_dataset`, `is_poor_quality`,
  `is_label_eligible`, `stratified_sample`, `build_chat_messages`).
- `balance_method` parameter on `sycophancy_probes.train_probe` (`"undersample"` — existing
  default behavior, drop majority class to `min(n_pos, n_neg)` — or `"upweight"` — keep every
  example, correct imbalance only via `_fit_probe`'s existing `pos_weight`-scaled
  `BCEWithLogitsLoss`) — user-requested, to compare both imbalance-handling strategies on
  the same cached activations rather than assuming one is correct.
- Two real Colab runs executed end-to-end: `sypr_probe_undersample/` and
  `sypr_probe_upweight/` under `tool_calling/tasks/sycophancy/results/`, both from the same
  n=120 stratified SyPR sample (24/domain) and same cached activations, differing only in
  `balance_method`. Each has `SUMMARY.md`, `plots/` (probe accuracy by layer, in-domain
  steering sweep, cross-dataset generalization), a `generations.jsonl` excerpt,
  `results.json`, and `mha_probe_vectors.pt` (see Known limitations below on reduced scope).

### Findings
- Class balance on the labeled SyPR sample: **12 sycophantic / 108 non-sycophantic (10.0% /
  90.0%)** — confirms the user's instinct that this dataset would not land near 50/50 and
  justifies training both variants rather than assuming undersampling's default was fine.
- `upweight` (full n=120, `pos_weight`-corrected) reaches meaningfully higher probe accuracy
  than `undersample` (n=24, class-balanced) at every component — MHA 0.900 vs. 0.833, MLP
  0.875 vs. 0.792, Residual 0.900 vs. 0.833 — though this comparison has a real caveat noted
  in both SUMMARY.md files: `upweight`'s test folds are imbalanced, so a chance/trivial
  always-predict-majority baseline already scores ~90% there, unlike `undersample`'s
  class-balanced (50% chance) test folds.
- The two variants select **different best directions** (undersample: MHA layer 14/head 9;
  upweight: MHA layer 0/head 7), not just different accuracy scores for the same underlying
  signal — training strategy changed which direction the probe actually found.
- Cross-dataset generalization is present but noisy for both variants — `upweight`'s
  direction shows the cleanest transfer (OEQ: consistently +8 to +16pp across all nonzero
  alphas), but neither variant reproduces the clean, mostly-monotonic pattern the AITA-
  derived DIM directions showed earlier this session. In-domain (held-out SyPR) steering
  effects are small/flat for both variants at this sample size (n=30 held-out).

### Known limitations (flagged, not fixed)
- **This Colab session's kernel exhibited a severe display/execution desync**: `get_cells`
  repeatedly returned byte-identical stale output for 30-45+ minutes while a cell was
  actually still running server-side and eventually completed successfully in full — silence
  did not mean the cell was hung. Separately, **re-invoking `run_code_cell` on an
  already-running cell was found to restart it from scratch** (confirmed by re-printed
  section headers producing different numbers on each "restart") rather than reattaching to
  the in-flight execution — this cost significant wall-clock time and Anthropic API budget
  before being diagnosed. Fix for future sessions: poll long-running Colab cells with
  `get_cells` only, never re-invoke `run_code_cell` on a cell already in flight, and don't
  treat the tool's ~900s per-call timeout as evidence of a hang.
- **No file-download channel was available from Colab back to the driving agent** in this
  session — only cell-output text (capped at roughly 85 KB per `get_cells` read). As a
  result, both results folders' `generations.jsonl` is a 3-row representative excerpt (not
  the full ~2.4 MB raw log per variant), and `mha_probe_vectors.pt` contains only the single
  globally-selected direction (not the full 1024-entry per-(layer,head) sweep the earlier
  DIM runs' `.pt` files contain). The full raw logs and probe weight checkpoints
  (`*_probe_weights.pth`, `*_projection_stds.pt`) exist in the (session-scoped) Colab
  filesystem but were not pulled out. Documented explicitly in each SUMMARY.md.
- Sample size (n=120 labeled, 12 positive) is small for a 10%/90%-imbalanced label — treat
  both runs as a first pass, not a well-powered result. No cross-dataset-generalization
  check exists yet from a *different* SyPR-derived direction (e.g. a residual or MLP
  direction) to see whether the choice of best-MHA-direction is itself representative.

## [Unreleased] — 2026-07-22

### Added
- AUC-ROC (per-fold CV, alongside Cohen's d) added to `compute_dim_direction` and the
  effect-size-by-layer plot in `moral_sycophancy_dim_colab_standalone.ipynb`.
- Early/middle/late layer-bucket cross-dataset generalization check (top-3 MHA directions
  by `abs(cohen_d)` per third of the network, 9 directions total, each tested against all 5
  datasets) — added to the DIM notebook.
- **`moral_sycophancy_dim_rowid_pooled_colab_standalone.ipynb`** — new notebook variant:
  averages every generation for one AITA conflict (both sides x all 3 samples each) into a
  single activation vector per row_id/conflict, instead of treating `original_post`/
  `flipped_story` as two separate labeled examples with a shared label.
- Batched generation (`ActivationSteerer.generate_batch`, left-padded, chunked by
  `GENERATION_BATCH_SIZE`) replacing one-prompt-at-a-time steering-eval loops — verified
  batch-size-invariant (`generate()` == `generate_batch(1)` == `generate_batch(4)`, exact
  text match) before trusting it on a real run.
- `generations.jsonl` logging in every steering-eval loop — every raw
  prompt/completion/verdict/alpha/direction, not just aggregate rates.
- **`moral_sycophancy_dim_truth_generalization_colab_standalone.ipynb`** — tests whether
  AITA-derived DIM directions generalize to MMLU (programmatic accuracy) and TruthfulQA
  (Claude-judged truthful vs. imitative falsehood, using the dataset's own
  `correct_answers`/`incorrect_answers` as judge context).
- **`truthfulqa_prefill_dim_moral_generalization_colab_standalone.ipynb`** — the reverse
  experiment: finds a direction directly from TruthfulQA via contrastive prefill
  (`question + best_answer` vs. `question + incorrect_answers[0]`, no LLM judge needed for
  direction-finding), tests whether it generalizes to moral/social sycophancy.
- Three real Colab runs executed end-to-end (previously the DIM notebooks were only built
  and mock-verified, never actually run) — results pulled locally to
  `tool_calling/tasks/sycophancy/results/{moral_sycophancy_dim_rowid_pooled_n100,
  dim_truth_generalization, truthfulqa_prefill_dim_moral_generalization}/`, each with
  `SUMMARY.md`, a `plots/` folder of locally-regenerated PNGs, raw `generations.jsonl`, and
  underlying `results.json`/`.pt` direction vectors.
- A single self-contained HTML artifact (published via the Artifact tool) making all three
  runs browsable: per-layer Cohen's d/AUC-ROC charts, steering-sweep and cross-dataset
  charts, bucket comparison, MMLU/TruthfulQA section, a paired generation browser (every
  prompt across all 5 alphas, filterable), and a curated chat-bubble-styled "see it in
  action" section (12 hand-picked baseline-vs-steered examples across AITA/OEQ/SS/TruthfulQA).
- `session_log_2026-07-22.md` — this session's narrative log.

### Changed
- `N_MMLU`/MMLU prompt format in the truth-generalization notebook: `max_new_tokens` 8→200,
  reasoning-then-`"Answer: X"` prompt, last-letter parsing instead of first-letter (see Fixed).
- Results folder location standardized to `tool_calling/tasks/sycophancy/results/` (moved
  out from under `notebooks/results/` partway through the session).

### Fixed
- MMLU truth-generalization check originally showed a near-null effect (90.2% of
  generations byte-identical across all 5 alphas) — root cause was `max_new_tokens=8`
  leaving no room for a steering nudge to override a confident greedy-decoded first token,
  not a real absence of effect. Fixed via longer generation + reasoning-then-answer format;
  byte-identical rate dropped to 12.0%, verdict-change rate rose from 6.8% to 21.9%.
- Artifact: a categorical color palette failed the colorblind-safety validator on first
  draft — fixed by using the dataviz skill's pre-validated 8-color ordering instead of
  hand-picked hues.
- Artifact: `str(-20.0)` (Python, used to build the paired-generation JSON) produces
  `"-20.0"` but `String(-20.0)` (JavaScript, used to look it up) produces `"-20"` — silently
  broke every alpha-column lookup in the paired generation browser. Fixed by matching
  Python's float-to-string formatting in the JS lookup key.
- `"truthful_qa"` (bare HF dataset ID) fails on current `datasets`/`huggingface_hub`
  versions — fixed to the namespaced `"truthfulqa/truthful_qa"` in both notebooks that load
  TruthfulQA.
- Multiple forks stalled repeatedly this session by using `Monitor`/background processes to
  wait on their own long-running Colab cells/tests, then pausing their turn instead of
  blocking synchronously on a plain `run_code_cell` call (which already blocks until the
  cell finishes) — fixed by explicitly forbidding async/background patterns up front in
  later fork prompts, which resolved it completely for every fork launched afterward.

### Known limitations (flagged, not fixed)
- The two later notebooks (truth-generalization, reverse) don't save a full 32-layer
  Cohen's d/AUC-ROC sweep like the row-id-pooled notebook does (only the 9 selected bucket
  directions' values) — their local per-layer plots show scattered known points, not a
  smooth curve.
- No "MMLU-derived direction → AITA" experiment exists (only AITA→MMLU/TruthfulQA and
  TruthfulQA→AITA/OEQ/SS were built) — would need a new contrastive-prefill-style notebook
  for MMLU specifically.

## [Unreleased] — 2026-07-14 to 2026-07-21

### Added
- `SAE/sae/`: TopK sparse autoencoder (`model.py`), activation-cache data loader
  (`data.py`), resumable training loop with FVU/L0/dead-feature-fraction metrics
  (`train.py`).
- `tool_calling/tasks/sae/`: agent-drivable tool wrappers (`list_activation_cache`,
  `train_sae`, `read_sweep_log`, `write_analysis`) around the SAE package.
- `tool_calling/tasks/sycophancy/`: ported `gpu_memory.py`, `sycophancy_data.py`,
  `sycophancy_model_registry.py`, `sycophancy_probes.py`, `sycophancy_compare.py`,
  `sycophancy_steering.py` from a sibling repo (`autointerp`).
- `moral_sycophancy_judge.py` — LLM-judge moral sycophancy labeling (AITA NTA/YTA
  pairwise verdict on `original_post`/`flipped_story`).
- `social_sycophancy_judge.py` — LLM-judge social sycophancy labeling
  (validation/indirectness/framing, single-response).
- 5-fold stratified cross-validation in `train_probe` (`sycophancy_probes.py`), with
  Wilson-score CI on pooled predictions.
- `moral_sycophancy_probes_colab_standalone.ipynb` — fully self-contained Colab notebook
  (labels → activations → probes → steering), `LABEL_SOURCE` toggle for moral/social.
- `steering_sweep_social_sycophancy.ipynb` — sweeps ~96 trained directions (every
  MLP/residual layer + best MHA head per layer) testing whether the moral-sycophancy
  direction transfers to social sycophancy.
- NTA/YTA in-domain check added to the steering-sweep notebook (does steering still work
  where the probe was trained, not just on the transfer target).
- CV error bars on the probe-accuracy-by-layer plot (std across the 5 folds; MHA uses the
  specific best head's fold spread, not an aggregate across heads).
- Sycophancy-**rate**-based steering evaluation (baseline vs. steered, on real held-out
  data) replacing a single hardcoded before/after text example.
- Full alpha sweep for steering: `STEER_ALPHAS` spanning -100 to +100 (13 points, denser
  near 0), `alpha=0.0` as a true unsteered baseline (hook not attached, not "attached with
  a zero vector").
- **`moral_sycophancy_dim_colab_standalone.ipynb`** — new notebook implementing
  difference-in-means (DIM) as an alternative to linear-probe training, with a
  `DIM_METHOD = "naive" | "cv_averaged"` toggle and Cohen's-d-based direction ranking.
- Delta-vs-baseline plots (change in sycophancy rate vs. `alpha=0.0`, not the raw rate) in
  both the probe and DIM notebooks' steering sections.
- `!wget`-based automatic dataset download (falls back to manual upload on failure) in all
  three notebooks, pointed at the correct `SAE/results/*.jsonl` (not `SAE/datasets/*.csv`,
  which lacks the `response` field these notebooks need).
- Real-dataset-sourced `TEST_PROMPTS`/`MORAL_TEST_PAIRS` in the steering-sweep notebook
  (previously 5+3 hardcoded example strings) — same shape, so no downstream code changed.
- Cross-dataset generalization check — tests the found/trained direction against held-out
  examples from **all 5** available datasets (`AITA-NTA-FLIP`, `AITA-NTA-OG`, `AITA-YTA`,
  `OEQ`, `SS`), not just the one used for training — added to both the DIM notebook and the
  steering-sweep notebook (adapted to its ~96-direction sweep, with a much smaller
  per-dataset cap given the multiplicative cost).
- `experiments.md`, `session_log.md`, `changelog.md` — session documentation.

### Changed
- Class-imbalance handling in `train_probe`, iterated twice before landing: (1)
  class-weighted `BCEWithLogitsLoss`, (2) a `balanced_accuracy` diagnostic metric (added
  then rolled back — "don't want another metric"), (3) final: undersample the majority
  class to `min(n_pos, n_neg)` before cross-validation, applied identically to
  `sycophancy_probes.py` and its notebook-inlined mirror.
- `tool_calling/run_agent.py` path resolution fixed for this repo's flat layout (no `src/`
  layer, unlike the sibling repo it was copied from).

### Fixed
- `AttributeError: 'ThinkingBlock' object has no attribute 'text'` in judge response
  parsing — fixed everywhere both judges appear (modules + notebooks) by searching for
  whichever content block actually has `.text`.
- Judge calls running adaptive thinking by default on a one-token classification task —
  fixed with explicit `thinking={"type": "disabled"}`.
- Sequential judge API calls (slow at n=500) — parallelized via `ThreadPoolExecutor`
  (~7-8x speedup verified with an artificially-delayed mock client).
- `SOCIAL_METRIC` (singular) referenced but never defined in the steering-sweep notebook's
  new cross-dataset check code — caught during mocked end-to-end testing, fixed by adding
  it to Config (`SOCIAL_METRIC = SOCIAL_METRICS[0]`).

### Known limitations (flagged, not fixed)
- `collect_activations`'s `batch_size` parameter is accepted but never used to actually
  batch forward passes — still one example at a time. Likely the next bottleneck at larger
  sample sizes.
- No cross-notebook compatibility shim between the DIM notebook's save format
  (`{component}_dim_vectors.pt`) and the probe pipeline's `load_steering_vectors` (expects
  an `nn.Linear`-shaped checkpoint) — noted as a clean future extension, not built.
