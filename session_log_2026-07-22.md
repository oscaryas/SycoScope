# Session Log — 2026-07-22

Continuation of the DIM (difference-in-means) sycophancy-steering work from the prior
session (see `session_log.md`, `changelog.md`, `experiments.md` for everything up through
2026-07-21). This session shifted from building notebooks to actually **running them** —
real Colab experiments, real findings, and turning the results into something browsable.

## Notebook enhancements (start of session)

Before any real runs, three features were added to `moral_sycophancy_dim_colab_standalone.ipynb`:

- **AUC-ROC alongside Cohen's d** — per-fold CV AUC-ROC computed the same way Cohen's d
  already was, added to `compute_dim_direction`'s return dict and to the effect-size-by-layer
  plot as a second chart. Selection logic untouched (still `abs(cohen_d)`-based).
- **Early/middle/late layer-bucket cross-dataset check** — partitions the 32 layers into
  thirds, picks the top-3 MHA directions per third by `abs(cohen_d)` (9 directions total),
  and runs the existing 5-dataset cross-generalization check on each.
- **Row-id pooling** — a new sibling notebook,
  `moral_sycophancy_dim_rowid_pooled_colab_standalone.ipynb`, that averages *all* generations
  for a conflict (both AITA sides × all 3 samples each) into one activation vector per
  row_id/conflict, instead of treating `original_post`/`flipped_story` as two separate
  labeled examples. Verified against real data that every `(row_id, prompt_col)` group has
  exactly 3 samples.
- **Batched generation** — `ActivationSteerer.generate_batch` (left-padded, chunked by
  `GENERATION_BATCH_SIZE`) replacing one-prompt-at-a-time calls in the steering-eval loops.
  Verified batch-size-invariance (`generate()` == `generate_batch(1)` == `generate_batch(4)`,
  exact text match) with the real tiny test model before trusting it on a real run.
- **`generations.jsonl` logging** — every raw generation (prompt/completion/verdict/alpha/
  direction) collected during steering and saved, not just aggregate rates — this turned out
  to matter a lot later (see "Raw generations" below).

All of the above were built by forks, verified locally (JSON parse → syntax check → mocked
end-to-end run with a fake Anthropic client + the real `tiny-random-LlamaForCausalLM` test
model) before ever touching Colab, matching the discipline established in the prior session.

**Fork reliability note**: multiple forks stalled repeatedly this session by using
`Monitor`/background processes to wait on their own long-running tests, then pausing their
turn instead of blocking synchronously — each stall required a manual resume message. This
was fixed by explicitly forbidding async/background patterns up front in later fork prompts,
which fixed it completely for every fork launched afterward.

## Run 1 — row-id-pooled DIM, moderate scale

First real Colab execution this session (previous sessions only built/verified locally).
Config: `N_EXAMPLES=100`, `STEER_ALPHAS=[-20,-5,0,5,20]`, `GENERATION_BATCH_SIZE=16`
(bumped from default 8 for A100 headroom), `N_EVAL_MAX`/`CROSS_DATASET_EVAL_MAX=35`.

**Findings** (full numbers in `tool_calling/tasks/sycophancy/results/moral_sycophancy_dim_rowid_pooled_n100/SUMMARY.md`):
- Best direction: **MHA layer 13, head 23** (Cohen's d=1.403, AUC-ROC=0.831).
- Original hypothesis ("steering effect is in early layers") **not confirmed** — strongest
  effects cluster in the middle third (13-19), though early layers aren't far behind
  (layer 10, d=1.317) and late layers are consistently *negative*-signed.
- Steering works: home-dataset moral sycophancy rate 41.7% → 71.4% at α=+20.
- Cross-dataset generalization is real but uneven: clean positive trend on all 5 datasets at
  α=+20, but AITA-NTA-OG/AITA-YTA/OEQ/SS show smaller and less monotonic effects than the
  home dataset.
- A user-caught design question mid-review: is row-id pooling only valid when both AITA
  sides *agree* (both NTA / both YTA), not when they're "mixed"? Investigated — confirmed
  the current label definition matches the user's own formal `S_moral` scoring formula
  exactly, so this was **not** a bug; pooling activations across a "mixed" conflict is a
  legitimate (if softer) signal under that definition, not something to fix.

## Raw generations became a first-class deliverable

Originally only aggregate rates were saved. After being asked to inspect actual generated
text, `generations.jsonl` was added to every subsequent run's save cell, and became the
basis for:
- Quantifying *how often* steering actually changes an individual example (not just the
  aggregate rate) — e.g. 44.7%-60% of AITA prompts change verdict across the alpha sweep,
  vs. only 6.8% of MMLU prompts in the original (8-token) run, which is what led to
  diagnosing the MMLU truncation problem below.
- Finding concrete "clean flip" examples (same prompt, α=0 vs α=+20, genuinely different
  verdict) for every dataset — 110 found across the AITA-derived-direction runs, 69 across
  the TruthfulQA-derived-direction run.

## Run 2 — does AITA-derived sycophancy generalize to MMLU/TruthfulQA?

New lean notebook, `moral_sycophancy_dim_truth_generalization_colab_standalone.ipynb`: takes
the 9 already-found bucket directions (no recomputation) and tests them against MMLU
(programmatic accuracy scoring) and TruthfulQA (Claude-judged truthful vs. imitative
falsehood, using the dataset's own `correct_answers`/`incorrect_answers` as judge context).

**First pass result was misleading**: 90.2% of MMLU generations were byte-identical across
all 5 alphas. Root cause: `max_new_tokens=8` (just enough for a letter) left no room for a
steering nudge to override a confident greedy-decoded first token.

**Fix**: reasoning-then-`"Answer: X"` prompt format, `max_new_tokens` 8→200, parse the LAST
letter/`Answer:` match instead of the first. Rerun (MMLU only, TruthfulQA untouched) dropped
the byte-identical rate to 12.0% and raised the verdict-change rate from 6.8% to 21.9% —
confirming the original near-zero MMLU effect was a measurement artifact, not a real null
result. Actual effect: modest and mostly *positive* (steering, either sign, slightly
increased MMLU accuracy on this sample) — no evidence steering meaningfully hurts raw
capability at this scale.

**TruthfulQA finding**: real but modest effect. Middle/late-bucket directions show small,
fairly consistent positive FALSE-rate deltas at α=+20 (more imitative falsehoods); early
layers trend the opposite way. Only 9 clean TRUE→FALSE flips exist in the whole run (vs.
110 for AITA), so the effect is real but far weaker than the moral-sycophancy effect on its
own home turf.

## Run 3 — the reverse experiment: does TruthfulQA-derived truthfulness generalize to moral sycophancy?

New notebook, `truthfulqa_prefill_dim_moral_generalization_colab_standalone.ipynb`: instead
of judging AITA responses with an LLM (as before), finds a direction directly from
TruthfulQA's own ground truth via **contrastive prefill** — teacher-force
`question + best_answer` (label=truthful) vs. `question + incorrect_answers[0]`
(label=misconception), no judge calls needed for direction-finding at all. Then tests
whether *that* direction increases moral/social sycophancy on AITA/OEQ/SS.

**Finding: the relationship is asymmetric.** Best direction here is **MHA layer 28** (late,
Cohen's d=-1.343) — a different location than the AITA-derived best (layer 13). Tested
against all 5 moral/social datasets, deltas are small and **non-monotonic** (e.g.
AITA-NTA-FLIP: +6.7%, +9.2%, 0%, -10.0%, +6.7% across the alpha sweep) — nothing like the
AITA-derived direction's clean, mostly-monotonic pattern on its own reverse test. Same
weak/noisy pattern holds across the full 9-direction bucket sweep. **Conclusion**:
truthfulness and moral sycophancy do not appear to share a clean, bidirectional common
direction in this model — steering toward sycophancy leaks a little into truthfulness, but
steering toward truthfulness doesn't reliably leak into sycophancy.

This run took much longer than expected (~5-6 hours estimated mid-run for the full 9×5
cross-check) — handled via a recurring 5-minute cron check that read the Colab notebook's
cell-execution state directly (cheap, read-only) rather than resuming the expensive fork
agent on every check; on the very first cron fire the underlying computation had already
finished (just hadn't run its plot/save cells yet), so those were run directly and the
results pulled down immediately.

## Results now live locally, not just in ephemeral Colab sessions

Every completed run's `generations.jsonl`/`results.json`/direction vectors were pulled out
of the Colab session (via the browser's native download — worked reliably for small files,
needed a chunked-print-and-reassemble workaround once for a file that kept failing
`files.download()` silently, likely a Chrome repeated-automatic-download block) and
organized under one consistent structure:

```
tool_calling/tasks/sycophancy/results/
├── moral_sycophancy_dim_rowid_pooled_n100/       (Run 1: AITA direction-finding + steering)
├── dim_truth_generalization/                      (Run 2: AITA direction -> MMLU/TruthfulQA)
└── truthfulqa_prefill_dim_moral_generalization/    (Run 3: TruthfulQA direction -> AITA/OEQ/SS)
```

Each folder has: `SUMMARY.md` (best directions, full per-alpha cross-dataset tables, a
handful of representative baseline-vs-steered example generations), a `plots/` folder of
locally-regenerated PNGs (matplotlib, not just Colab's inline `plt.show()`), the raw
`generations.jsonl`, and the underlying `results.json`/`.pt` direction vectors.

## The browsable artifact

A single HTML artifact (self-contained, no external dependencies, published via the
Artifact tool) was built and iterated on throughout the session to make all of this
explorable without digging through JSON:

- Per-layer Cohen's d + AUC-ROC charts (custom SVG line-chart renderer, no charting library)
- Main steering sweep and cross-dataset generalization charts (delta-vs-baseline convention
  throughout, diverging red/blue color coding)
- Early/middle/late bucket comparison
- MMLU/TruthfulQA truth-domain generalization section
- A **paired generation browser** — every prompt shown across all 5 alphas side-by-side,
  baseline highlighted, filterable by section/dataset/direction
- A **curated "see it in action" section** styled as chat conversations (prompt bubble +
  baseline/steered response bubbles with verdict badges) — 12 hand-picked examples spanning
  AITA/OEQ/SS/TruthfulQA and all three layer buckets

Two real bugs were caught and fixed in the artifact itself: a categorical color palette that
failed the colorblind-safety validator on first draft (fixed by using the dataviz skill's
pre-validated 8-color ordering instead of hand-picked hues), and a Python-vs-JavaScript
float-to-string mismatch (`str(-20.0)` → `"-20.0"` in Python, `String(-20.0)` → `"-20"` in
JS) that silently broke every alpha-column lookup in the paired generation browser — caught
because the user actually browsed the data and noticed nothing was populating.

## Open threads

- No "MMLU-derived direction → AITA" experiment exists (only AITA→MMLU/TruthfulQA and
  TruthfulQA→AITA/OEQ/SS) — would need a new contrastive-prefill-style notebook for MMLU
  specifically if wanted.
- The row-id-pooled notebook's per-layer Cohen's d/AUC-ROC arrays exist as `.pkl` files
  locally; the two later notebooks (truth-generalization, reverse) don't save a full
  32-layer sweep, only the 9 selected bucket directions' values — so their local plots show
  scattered known points rather than a smooth per-layer curve.
