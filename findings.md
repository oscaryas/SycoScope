# Findings — Sycophancy Direction-Finding, Steering & Generalization

Synthesis of every experiment run this session (2026-07-22 through 2026-07-29) on
`meta-llama/Meta-Llama-3-8B-Instruct`. See `changelog.md` for chronological detail on
what changed and why, `experiments.md` for the underlying tooling/infrastructure, and
`tool_calling/tasks/sycophancy/results/<run_name>/SUMMARY.md` for each run's full data
(per-alpha tables, representative generations). This document is the cross-experiment
synthesis — what the results mean taken together, not a restatement of each one.

## TL;DR

- The **single cleanest result of the whole session** is the AITA-derived
  difference-in-means (DIM) direction (`moral_sycophancy_dim_rowid_pooled_n100`,
  MHA layer 13/head 23, Cohen's d=1.4): a large, monotonic, cross-dataset-generalizing
  sycophancy effect, and steering it shows **no evidence of hurting general capability**
  (MMLU accuracy mostly moves *up* under steering) and only a modest, uneven effect on
  truthfulness.
- **Every probe-based, single-source direction tried after that (OEQ-alone, SyPR-alone)
  was noticeably weaker and noisier** — most likely because both of those sources have
  extreme class imbalance (OEQ: 96.6%/3.4%; SyPR: 10%/90%) versus AITA's comfortable
  ~42%/58% split, leaving too few examples of the minority class to pin down a reliable
  direction even after correcting for imbalance.
- **Truthfulness and sycophancy look related but not identical.** The AITA→TruthfulQA
  direction (forward) shows a real, if modest, effect; the TruthfulQA→AITA/OEQ/SS
  direction (reverse, found independently via contrastive prefill) is weak and
  non-monotonic. This asymmetry — not a clean shared axis — is itself a finding.
- A **mixed AITA+SyPR probe direction, swept against MMLU/TruthfulQA across all 32
  MLP and all 32 residual layers**, is running now (background, ~6-9 hrs) to test
  whether combining sources produces something more robust than either alone. Results
  pending — see "In progress" below.
- Methodologically: **class balance must always be checked, never assumed** (three
  different sources turned out imbalanced in three different ways), **judge calls must
  be parallelized** (a sequential-loop bug turned a should-be-30-minute run into
  several hours), and **short generation caps silently masquerade as null results**
  (an 8-token MMLU cap looked like "no effect" until raised to 200).

## Experiments run

| # | Run | Method | Data source(s) | Direction found | Tested against |
|---|---|---|---|---|---|
| 1 | `moral_sycophancy_dim_rowid_pooled_n100` | DIM | AITA-NTA-FLIP (moral sycophancy, row-id pooled) | MHA L13/H23 | AITA-NTA-FLIP/OG/YTA, OEQ, SS |
| 2 | `dim_truth_generalization` | DIM (reuses #1's directions) | — | 9 directions (early/mid/late buckets) from #1 | MMLU, TruthfulQA |
| 3 | `truthfulqa_prefill_dim_moral_generalization` | DIM, contrastive prefill | TruthfulQA correct/incorrect answers | MHA L28/H0 (+ 8 more, bucketed) | AITA-NTA-FLIP/OG/YTA, OEQ, SS |
| 4 | `sypr_probe_undersample` / `sypr_probe_upweight` | Linear probe | SyPR (`vennemeyerd/sycophantic-praise`) | undersample: L14/H9 · upweight: L0/H7 | AITA-NTA-FLIP/OG/YTA, OEQ, SS + SyPR held-out |
| 5 | `oeq_probe_undersample` / `oeq_probe_upweight` | Linear probe | OEQ "validation" (ELEPHANT metric) | undersample: L31/H24 · upweight: L6/H14 | AITA-NTA-FLIP/OG/YTA, OEQ held-out, SS |
| 6 | `mixed_probe_undersample` / `mixed_probe_upweight` (**in progress**) | Linear probe | AITA-NTA-FLIP + SyPR combined | 32 MLP layers + 32 residual layers (no MHA) | MMLU, TruthfulQA |

All runs used real Llama-3-8B-Instruct generations, real Claude-judged labels, and a
real alpha sweep `[-20, -5, 0, 5, 20]` (`0` = true unsteered baseline, hook not attached).

## Per-experiment findings

### 1. AITA-derived DIM direction (the strongest result)

Best direction: **MHA layer 13, head 23** (Cohen's d=1.403, AUC-ROC=0.831). Steering it
moves the home-dataset moral sycophancy rate (both AITA sides judged NTA) from a 41.7%
baseline to **71.4% at α=+20** and down to essentially 0% at α=-20 — large and
monotonic. Generalizes, unevenly but really, to 4 other held-out datasets (AITA-NTA-OG,
AITA-YTA, OEQ, SS). Layer-bucket analysis (early/mid/late thirds, top-3 heads each)
found effects concentrate in **middle layers**, with **late layers flipping sign**
(steering positive there *reduces* sycophancy) — the original "early layers" hypothesis
was not confirmed.

### 2. AITA direction → MMLU/TruthfulQA (forward truth-generalization)

Tests whether the 9 bucketed AITA directions from #1 hurt capability or truthfulness.
**No evidence of a capability tax** — MMLU accuracy mostly moves *up* under steering
(e.g. +5–12pp at some alphas from a 46.7% baseline) across nearly every direction.
TruthfulQA's imitative-falsehood rate shows a **modest, mostly positive** effect
concentrated in middle/late-bucket directions, but on a much smaller signal (9 clean
flips across the whole run vs. 110 on the AITA side) — real, but far weaker than the
in-family effect.

*Caveat already documented*: the first MMLU pass used `max_new_tokens=8`, which looked
like a near-null result (90.2% byte-identical generations across alphas) purely because
generation was truncated before steering could act — fixed by raising to 200 tokens
(byte-identical rate dropped to 12.0%). Worth remembering whenever a "no effect" result
shows up with a short generation cap.

### 3. TruthfulQA → AITA/OEQ/SS (reverse test)

A direction found **independently** from TruthfulQA (contrastive prefill on its own
correct/incorrect answers, no LLM judge needed for direction-finding) lands at a
**different, later layer** (MHA L28/H0) than the AITA-derived direction's L13, and its
effect on AITA/OEQ/SS is **small and non-monotonic** — unlike the forward direction's
clean pattern. Read together with #2: truthfulness and moral sycophancy are **related
but not the same axis** in this model — pushing on one doesn't reliably move the other,
and it's more asymmetric in one direction than the other.

### 4. SyPR (sycophantic praise) probe — weak, small minority class

Labeling required generating fresh model responses (SyPR ships none) and judging
praise-on-poor-quality. **Class balance: 12 sycophantic / 108 non-sycophantic (10%/90%)**
at n=120 — confirms sycophancy-as-"praise something bad" is real but not the dominant
behavior here (the model actually praises *good* work more often, 45% vs. 20%).
Two variants trained on the same labels: undersample (12/12, best MHA L14/H9,
acc=0.833) and upweight (all 120, best MHA L0/H7, acc=0.900 — inflated by base-rate,
not directly comparable to undersample's balanced-fold accuracy). **Neither generalizes
cleanly**: undersample's cross-dataset pattern is noisy/non-monotonic; upweight shows a
consistently positive OEQ signal (+8–16pp) but a large non-monotonic swing on
AITA-NTA-FLIP. At n=12 positives, this is a legitimate first pass, not a well-powered
result.

### 5. OEQ "validation" probe — even more extreme imbalance, flat in-domain effect

Labels sourced from OEQ's own cached generations via ELEPHANT's "validation" metric
(does the response emotionally validate regardless of merit) — the closest existing
"pure sycophancy" signal to what was originally asked for, reusing cached data instead
of generating fresh. **Class balance: 143 validating / 5 not-validating (96.6%/3.4%)**
at n=148 — this model *almost always* validates open-ended advice. Two variants:
undersample (5/5, best MHA L31/H24, acc=0.900) and upweight (all 148, best MHA L6/H14,
acc=0.973 — same base-rate-inflation caveat as SyPR's upweight). **The in-domain
held-out check (the most direct test) is nearly flat for both variants** (0% to +3.3%
delta) — low-power, not evidence the signal doesn't exist, but this run doesn't
demonstrate a clean generalizing direction the way #1 did.

### Judge-success/failure rates (data-quality note, computed post hoc)

- SyPR's binary praise judge: **99.9–100% parseable** across 900 generations per variant.
- OEQ's cross-dataset sweep, which reuses the 3-way AITA verdict judge (NTA/YTA/**OTHER**)
  for the AITA-family datasets: only **86.3% judged** overall (647/750), with failures
  concentrated in the AITA domains specifically (67–78% success on AITA-NTA-FLIP,
  67–73% on AITA-YTA) while OEQ/SS's own binary judge succeeds ~100% of the time. The
  3-way verdict format is structurally more prone to ambiguous "OTHER" outcomes than a
  binary judge — worth knowing when comparing rates across datasets that use different
  judges.

## Cross-cutting synthesis

**Why did the AITA DIM direction work so much better than the later probe directions?**
Three candidate reasons, not mutually exclusive: (a) AITA's label is comparatively
well-balanced (~42%/58%) while SyPR and OEQ are both severely imbalanced in opposite
directions, giving AITA far more usable minority-class signal; (b) AITA's row-id-pooled
labeling averages 6 generations per conflict into one vector, a noise-reduction step
the later runs didn't replicate; (c) DIM itself (a simple mean-difference, no gradient
descent) may just be a more robust estimator than a gradient-trained linear probe at
these sample sizes — this session never ran a head-to-head DIM-vs-probe comparison on
the *same* data source to isolate method from data effects, which is a real gap.

**Does steering sycophancy directions cost general capability?** Across every MMLU
check run this session, **no** — accuracy holds steady or improves under steering in
nearly every direction tested. This is a mildly surprising, consistent pattern worth
trusting: whatever these directions encode, it doesn't appear to trade off against
raw factual-recall capability, at least not on MMLU specifically.

**Is there one shared "sycophancy direction," or several?** The layer locations found
across sources disagree substantially: AITA best at L13 (middle), TruthfulQA-reverse
best at L28 (late), SyPR undersample at L14/upweight at L0, OEQ undersample at
L31/upweight at L6. Some of this spread is likely balance-method artifact (undersample
vs. upweight on the *same* labels already land at different layers for both SyPR and
OEQ), but the AITA/TruthfulQA split in particular — two directions found from
genuinely different label sources — landing at opposite ends of the network is a real
signal that "sycophancy" as measured here is not one clean, unified direction.

## LLM-as-judge methodology, and how judge design shaped the results

Every label and every cross-dataset rate in this session came from Claude
(`claude-sonnet-5`) as a judge, never a programmatic classifier (the one exception is
MMLU, graded by letter-match, no judge needed) — but the *shape* of the judging
differs across experiments, and that shape measurably affects how much usable data
each run actually got.

### Four judge designs used

| Judge | Format | Response scope | Used for |
|---|---|---|---|
| `moral_sycophancy_judge.judge_verdict` | 3-way: `NTA` / `YTA` / `OTHER` | Paired — needs both sides of one AITA conflict | AITA-NTA-FLIP labeling, AITA-family cross-dataset checks |
| `social_sycophancy_judge.judge_metric` | Binary per metric (`validation`/`indirectness`/`framing`, ELEPHANT's own prompts, kept verbatim) | Single response | OEQ/SS labeling and cross-dataset checks |
| `sycophantic_praise_judge.judge_praise` | Binary: praise / no praise | Single response | SyPR labeling and cross-dataset checks |
| TruthfulQA judge (`dim_truth_generalization`) | 3-way: `TRUE` / `FALSE` / `UNCLEAR`, given the dataset's own correct/incorrect answer lists as context | Single response | MMLU/TruthfulQA generalization checks |

All four share the same underlying pattern: `claude-sonnet-5`, `thinking={"type":
"disabled"}` (a one-token classification task doesn't need adaptive reasoning, and
some models run it by default otherwise), and ThinkingBlock-safe text extraction
(`next((b.text for b in msg.content if hasattr(b, "text")), "")`, since a thinking
block sometimes precedes the text block even with thinking nominally off).

### The design choice that matters most: does the judge get an "I can't tell" option?

The binary judges (SyPR's praise, OEQ/SS's validation) force a 0/1 answer — there's no
hedge, so nearly every response gets classified: **99.9–100% parseable** across SyPR's
900 logged generations per variant, and OEQ's own held-out slice resolves at ~100%
too.

The two 3-way judges both include an explicit "can't tell" category (`OTHER` for AITA
verdicts, `UNCLEAR` for TruthfulQA), and both pay for it in yield. Across OEQ's
cross-dataset sweep, the AITA-family datasets (which reuse the AITA verdict judge)
succeed only **67–78% of the time** — roughly a quarter to a third of AITA-domain
responses come back genuinely ambiguous enough to be excluded from the rate
calculation — while the same sweep's binary-judged datasets (OEQ, SS) succeed ~100%
of the time. (The TruthfulQA judge's exact `UNCLEAR` rate was never computed as a
standalone number in this session — flagging that as a real gap rather than guessing
a figure — but the same structural pattern, a 3-way judge with a hedge option costing
yield that a binary judge doesn't, should apply there too.)

### Why this matters for reading the results

A lower parse-success rate isn't just "less data" — it silently shrinks the
*effective* `cross_dataset_n` for whichever datasets use the pickier judge, without
changing the number requested in any config. Two datasets both configured for `n=30`
aren't actually comparable at that n if one resolves to 30 judged examples and the
other to 20: the smaller effective sample is noisier, and a rate computed from it
deserves proportionally less confidence. This sits, uncontrolled, inside every
cross-dataset table this session that mixes AITA-family datasets (3-way judge) with
OEQ/SS/SyPR (binary judge) — worth flagging explicitly rather than reading percentage
columns as equally reliable across rows.

It also means the AITA-derived direction's unusually large "clean flip" count (110,
vs. 9 for TruthfulQA and comparably modest counts elsewhere) is partly a judge-design
effect layered on top of whatever the real underlying effect size is — the AITA
verdict judge, despite its lower per-attempt parse rate, still had the largest raw n
going in (row-id-pooled AITA data was the largest single labeled set built this
session), so more attempts survived into "clean flip" territory even at a lower
success rate per attempt.

## Infrastructure built this session

- **TopK sparse autoencoder package** (`SAE/sae/`) and its agent-tool wrapper
  (`tool_calling/tasks/sae/`) — separate from the sycophancy work above, not otherwise
  covered in this document.
- **Sycophancy probing pipeline** (`tool_calling/tasks/sycophancy/`): labeling judges
  (moral/social/SyPR-praise), linear probes with CV + Wilson CI, DIM as an alternative
  direction-finder, `ActivationSteerer` for MHA/MLP/residual steering with a batched
  `generate_batch` method (added this session).
  - **Architecture shift mid-session**: every run through the SyPR/OEQ probes was
    driven via hand-built Colab notebooks with real, repeated pain — re-invoking
    `run_code_cell` on an already-running cell restarts it from scratch; cell output
    caps forced `generations.jsonl` to a 3-row excerpt and `.pt` checkpoints to a single
    selected direction instead of the full sweep for the SyPR run. **Fixed for OEQ
    onward** by writing a real standalone `.py` script (`oeq_probe_pipeline.py`) that
    writes straight to disk, and by pushing code to Colab via direct Google Drive
    upload (`create_file`) instead of typing it into cells — eliminates both failure
    modes at once. The in-progress mixed-source run uses the same pattern plus
    incremental result-flushing, given its much longer expected runtime.
  - A real perf bug (sequential, unparallelized judge calls in `oeq_probe_pipeline.py`'s
    cross-dataset sweep) was found and documented but not fixed in place — the mixed
    run's script was written to parallelize from the start instead.

## In progress

**`mixed_probe_undersample` / `mixed_probe_upweight`** — combines AITA-NTA-FLIP +
SyPR into one training set, tests whether the combined direction generalizes better to
MMLU/TruthfulQA than the AITA-only result (#2 above), sweeping all 32 MLP layers and
all 32 residual layers (MHA excluded per an explicit compute-cost tradeoff). Running in
the background; expect several hours given the sweep size (~96,000 generations +
~48,000 judge calls). This document will need a follow-up entry once it lands.

## Where everything lives

- Results: `tool_calling/tasks/sycophancy/results/<run_name>/` — each has `SUMMARY.md`
  (readable writeup with per-alpha tables and example generations), `results.json`
  (raw numbers), `generations.jsonl` (full raw model outputs), `plots/`, and full
  per-layer/head `.pth`/`.pt` steering-vector checkpoints.
- Pipeline code: `tool_calling/tasks/sycophancy/*.py` (probes, steering, judges, model
  registry) + `notebooks/` (the pre-script-migration Colab notebooks) +
  `oeq_probe_pipeline.py`/`mixed_probe_pipeline.py` (the standalone-script pattern).
- Narrative history: `changelog.md` (what changed, in order), `experiments.md` (what
  the tooling does and why), `session_log*.md` (day-by-day narrative).
