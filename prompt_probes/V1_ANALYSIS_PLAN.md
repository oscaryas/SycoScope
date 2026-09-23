# V1 sycophancy probe analysis plan

## Research question

Which sycophancy definitions yield similar detection patterns, which yield distinct patterns, and which apparent distinctions are explained by warmth, praise, confidence, or other features of the prompt contrast?

This plan follows the observational approach in [Building Better Deception Probes Using Targeted Instruction Pairs](https://arxiv.org/html/2602.01425v1): compare external detection performance, correlate probe scores on common examples, and inspect disagreements. The goal is evidence about representational and behavioral specificity; causal steering is outside this phase.

## Scope

- Keep the existing prompt pairs fixed as **v1**: `data/sycophancy_probe_prompt_pairs.json`.
- Use `results/llama31_5k_subset/FINDINGS.md` as the starting findings document.
- Analyze the Llama-3.1 run separately from the older Llama-3.0 `results/main` run. Do not combine activations or apply probes across these models.
- Include all 20 conditions, distinguishing the eight taxonomy cells, the general baseline, and the control contrasts.
- Interpret control labels according to their actual instructions. A control's positive condition means the named behavior is present; it does not necessarily mean sycophancy. Do not reverse signs merely to improve correlations.

## 1. Reconcile the saved results

The initial inspection found 20 probe files and activation files in the Llama-3.1 subset, but only 15 cells in its aggregate `summary.json`. The omitted cells were:

- `general_baseline`
- `pv_implicit`
- `pe_implicit`
- `ctrl_pure_sycophancy`
- `ctrl_obsequiousness`

Reconcile the aggregate summary with the per-cell metrics and available artifacts. Verify model identity, layer conventions, activation positions, example alignment, and prompt splits before comparing cells. Keep related examples and both sides of a prompt pair in the same split.

**Deliverable:** a complete inventory of the 20 conditions and their available training and evaluation artifacts.

## 2. Build the external-performance matrix

Score every v1 probe on the same held-out examples for each evaluation target:

- **SyPR:** unwarranted praise.
- **AITA:** report the existing moral labels separately, including `unwarranted_nta` and `both_nta`.
- **ELEPHANT:** validation, indirectness, and framing, adding Llama-3.1 evaluation activations where unavailable.

Compare each taxonomy probe with `general_baseline`, relevant controls, and the strongest single probe selected on validation data. Select layers, token positions, and other configurations on validation examples, then report results on separate test examples. Since existing findings have already been inspected, describe analyses of those same examples as exploratory; reserve fresh examples for confirmation.

Report AUC and paired bootstrap confidence intervals for differences between probes. Resample at the underlying prompt or scenario level so related responses remain together. Where a target has matched positive and negative responses, also report paired win rate.

Specify semantic predictions before evaluating new targets. For example, `pt_explicit` should be relevant to unwarranted praise, and `pe_explicit` should be relevant to indiscriminate emotional validation. Treat benchmark-to-taxonomy mappings as hypotheses.

Look for crossovers: one probe performs better on praise while another performs better on emotional validation. Such a pattern is more informative about specialization than a probe winning on every target.

**Deliverable:** a probe-by-evaluation-target performance matrix with uncertainty intervals and comparisons against the general baseline.

## 3. Cluster probe scores on common examples

Construct an example-by-probe matrix of continuous decision scores. Within each layer and token position, every probe must be scored on the same activation rows.

- Calculate signed Pearson correlations between probe scores.
- Report correlations separately within each dataset and on a balanced or explicitly weighted combined evaluation set. Pooled correlations can reflect differences between datasets rather than agreement within them.
- Use hierarchical clustering and bootstrap examples to assess which groupings are stable.
- Explore the number of clusters rather than fixing it at five.
- Report signed correlations as the primary view. The current `analyze_probes.py` uses `1 - |similarity|`, which groups opposite-scoring probes together. An absolute-correlation view can be supplementary, but must be labeled as similarity of axes regardless of direction.

Questions to answer:

- Do explicit and implicit versions of a domain cluster together?
- Do several taxonomy cells collapse into a shared detection profile?
- Do emotion probes cluster with warmth, politeness, or appropriate support?
- Does unwarranted praise separate from warranted praise?
- Is the calibrated-hedging inversion consistent across external datasets?

Probe-direction cosines and split-half reliability can supplement score correlations. Low cosine alone does not establish different semantic properties, and score correlation depends on the evaluation distribution.

**Deliverable:** signed correlation heatmaps, dendrograms, and a summary of stable groups and their uncertainty.

## 4. Evaluate specificity against benign look-alikes

Prioritize these comparisons:

| Probe comparison | Main question |
|---|---|
| `pt_explicit`, `ctrl_obsequiousness`, and `ctrl_warranted_praise` | Does the probe detect unwarranted praise or praise generally? |
| `pe_explicit`, `pe_implicit`, and `ctrl_emotional_support` | Does the probe distinguish inappropriate validation from appropriate support? |
| Emotion probes and `ctrl_politeness` | How much of the detection pattern tracks warmth or courtesy? |
| `ctrl_evasive_hedging` and `ctrl_calibrated_hedging` | Does the probe distinguish avoiding disagreement from expressing justified uncertainty? |
| Position probes and `ctrl_genuine_agreement` | Does the probe distinguish incorrect agreement from agreement supported by the evidence? |

Use independently labeled external examples where possible. Include negative examples that retain the surface behavior: an honest response can praise, agree, hedge, or sound warm. Training-condition labels alone do not establish whether the generated behavior was warranted.

Measure false-positive rates at thresholds chosen on validation data, alongside AUC. The existing AITA `unwarranted_nta` comparison is a useful starting point because both classes contain an NTA verdict, though other contextual and stylistic differences still need checking.

**Deliverable:** a specificity report describing what each probe flags on appropriate versus inappropriate behavior.

## 5. Audit disagreements between probes

Extend the existing hedging investigation to praise, agreement, and emotional validation.

1. Choose probe pairs based on the planned comparisons and observed stable clusters.
2. Convert scores to within-probe percentile ranks on a common evaluation set, then sample large disagreements in both directions. Include a random comparison sample.
3. Have annotators inspect the full user context and response without seeing probe identities or scores.
4. Label observable behaviors: praise, endorsement, omitted correction, warmth, uncertainty, and whether the response is warranted by the context. Allow multiple labels and uncertainty.
5. Report behavior frequencies, counterexamples, and representative excerpts. Distinguish conclusions about the selected disagreement sample from conclusions about the whole dataset.

**Deliverable:** a structured disagreement audit that explains detection profiles with evidence beyond individual anecdotes.

## V1 write-up

Organize the results around three questions:

1. **Shared structure:** which definitions produce correlated detection patterns?
2. **Specialization:** which probes have reproducible advantages on particular behaviors?
3. **Specificity:** which probes separate sycophancy from appropriate praise, agreement, support, and uncertainty?

Report overlapping or collapsed definitions as findings. Strong within-cell AUC and cross-cell transfer alone do not establish that each definition corresponds to a distinct semantic representation. Describe conclusions as detection profiles supported by the tested examples, without claiming distinct causal circuits.

## Transition to v2

Preserve v1 prompts, artifacts, and results as the baseline. After completing this analysis, test targeted prompt revisions and multiple paraphrases per definition. Evaluate whether v2 reduces false positives on benign look-alikes while retaining detection of sycophancy and the meaningful distinctions observed in v1.

The v2 comparison should test whether improved contrast definitions improve specificity; it should not optimize the already-saturated within-cell training AUC.
