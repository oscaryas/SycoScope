# SYCON-Bench probe-score cluster analysis

Completed 2026-09-16. This extends the initial clustering with question-level bootstrap stability and nuisance/sensitivity checks. No additional generation, GPU extraction, judging, or probe training was used.

## Bottom line

**Shared probe-score structure recurs, but these results do not establish two or three distinct sycophancy concepts.** Two clusters are selected in 21/24 setting–layer combinations, yet 11 of those are simply calibrated hedging versus the other 19 probes. The most interpretable middle-layer example is an ethical L20 support/qualified-response group; some seemingly richer partitions are unstable. The original eight taxonomy cells do not emerge as a clean, consistent organization.

Importantly, **few clusters is not the same as one activation axis**: the first principal component explains only 28–41% of standardized score variance across these configurations.

## What was analyzed

- All 20 frozen probes, using **full current-response-average activations**, at L0/4/8/12/16/20/24/28. The probe and evaluation pooling/layer match.
- Three settings analyzed separately: debate (100 questions, 500 turns), ethical (200, 1,000), false presupposition (198, 990). Incomplete generations excluded by the existing evaluator are not reintroduced.
- Signed Pearson score correlations, average-linkage clustering, silhouette selection among k=2…9. This reproduces the existing 24 correlation matrices.
- **200 question-bootstrap draws per configuration and per raw/adjusted view**: each sampled question contributes all its turns; k is reselected each draw. Configurations reuse data and are not independent replications.
- Adjustment for categorical turn number and log(1 + response-token count), refitted within each bootstrap draw. Supplementary checks remove the calibrated-hedging probe, cluster each turn separately, average scores by question, or use unsigned distance 1−|r|.
- All available questions are used for descriptive clustering, without selecting on judge labels. These are exploratory analyses, not an untouched confirmatory test.

## 1. Two clusters often means one outlier

The raw best-k distribution is **k=2: 21/24; k=3: 2/24; k=8: 1/24**. Eleven configurations split 19+1, always isolating `ctrl_calibrated_hedging`.

For example, false-presupposition L16 selects this split in all 200 bootstrap draws by cluster count. That does not make it two validated behavioral concepts: calibrated hedging has median correlation **−0.296** with the other probes. After removing it, the selected split instead separates `pv_implicit` + `ctrl_evasive_hedging` from the remaining 17 probes.

Signed clustering appropriately separates opposite-firing probes, but opposite firing can reflect opposite ends of a shared axis. The unsigned analysis is a sensitivity check for that possibility, not a replacement for signed behavioral interpretation.

## 2. A substantive middle-layer group survives some important checks

At **ethical L20**, the best partition is 14+6. The six-member group is:

- `pe_implicit`
- `ctrl_warranted_praise`
- `ctrl_genuine_agreement`
- `ctrl_emotional_support`
- `ctrl_calibrated_hedging`
- `ctrl_politeness`

This is descriptively a **support/qualified-response group**, not a pure sycophancy category. It contains benign controls and an inverted-polarity control, and notably **does not include `pe_explicit`**.

The remaining 14 include `general_baseline`, the position probes, both traits probes, explicit emotion validation, and most performance/flattery controls. That larger group should not be treated as internally homogeneous.

Evidence for this example:

- Two clusters recur in **198/200 draws (99%)**. Median adjusted Rand index (ARI) versus the original partition is **1.00**; the descriptive 2.5–97.5 percentile range is **0.271–1.00**, so some resamples do change memberships. Same k does not mean identical memberships.
- Removing turn/length effects leaves the original membership **unchanged**; the adjusted bootstrap selects k=2 in 200/200 draws.
- Removing calibrated hedging leaves the other five members together and preserves the partition of the remaining probes.
- It is not invariant across turns: turn-specific ARI versus the pooled partition ranges **0.047–1.00**. The question-mean partition has ARI **0.795**. Treat this as a useful pooled pattern, not a universal partition across interaction stages.

This example was highlighted after inspecting the results, not chosen as a preregistered primary layer.

[Ethical L20 score correlations](ethical_response_L20_correlation.png) · [Bootstrap co-clustering, k reselected](clustering_stability/ethical_response_L20_bootstrap_selected_k.png) · [Fixed-k=3 sensitivity](clustering_stability/ethical_response_L20_bootstrap_k3.png)

## 3. Richer-looking partitions need caution

| Example | Full-data partition | Bootstrap support for that k | What changes |
|---|---|---:|---|
| Ethical L4 | 8 clusters | 23/200 = 11.5% | Turn/length-adjusted data select 2; raw bootstrap most often selects 2. |
| Debate L20 | 3 clusters, sizes 7+11+2 | 38/200 = 19% | Raw bootstrap selects 2 in 157/200 draws; adjusted full data select 2. |
| Debate L16 | 2 clusters, sizes 15+5 | 194/200 = 97% | Membership is less stable: median raw ARI 0.618, adjusted bootstrap median ARI 0.190. Removing hedging selects 4 clusters. |
| Ethical L20 | 2 clusters, sizes 14+6 | 198/200 = 99% | Original partition survives turn/length adjustment and hedging removal, but changes across individual turns. |

Debate L20's small third group is `pv_implicit` + `ctrl_evasive_hedging`. Its presence is a plausible lead for a premise-acceptance/evasion distinction, **not a stable discovery of a third concept**.

## 4. Turn and length matter, without explaining everything

After adjustment, k=2 is selected in **22/24 configurations**; false-presupposition L24 selects 3 and L28 selects 4. Only **10/24** full partitions are exactly unchanged (ARI=1).

Thus the headline cluster count can remain two even when the members move substantially. In false-presupposition L12, raw and adjusted partitions have ARI **0.122**: the small two-probe outlier group becomes a broader eight-probe group.

These adjustments remove only the modeled associations. They do not resolve topic, lexical, training-prompt, or judge-validity confounds. Response length may itself reflect behavior, so adjustment is a sensitivity analysis rather than a uniquely correct version of the data.

## 5. What this says about the taxonomy and activation space

At ethical L20, in the supplementary fixed-k=3 bootstrap:

- Explicit subjective-position and explicit traits probes co-cluster **100%** of the time (score r=0.710).
- Implicit emotion and emotional-support probes co-cluster **100%** (r=0.733).
- Explicit and implicit emotion probes co-cluster only **4.5%**, despite a positive correlation of **r=0.495**.

This illustrates cross-definition overlap and within-definition heterogeneity. It also illustrates why a cluster boundary is not proof of independence: two separated probes can still correlate substantially. Fixed k=3 forces a partition and does not validate three concepts. The result cannot identify whether explicit/implicit wording in the training instructions caused the pattern.

Across all 24 raw configurations, PC1 explains **27.9–40.7%**, and the first three PCs explain **56.7–70.5%**, of standardized probe-score variance. The participation rank of the score-correlation spectrum is **4.58–7.17**. These are descriptions of the 20 probe outputs, including noise and dataset-specific variation—not estimates of the number of underlying concepts or causal activation dimensions.

The defensible interpretation is: **the probes have shared, context-dependent response patterns, with some recurring groupings; neither eight clean concepts nor a universal single axis is established.** Score clusters alone also cannot distinguish overlapping probe readouts from genuinely inseparable underlying representations.

## Next tests

1. **Replicate the ethical L20 membership on independently generated questions and prompt paraphrases**, fixing the layer, pooling, and partition before seeing the new data. Evaluate both pairwise correlations and membership stability.
2. **Cross behavior with surface form:** matched agreement versus praise, warranted versus unwarranted behavior, and warm versus blunt style. Score all frozen probes under a common generation prompt and independently label behavior. This is more diagnostic of concept mixing than another unrestricted cluster sweep.
3. **Distinguish continuous shared factors from discrete groups:** use held-out covariance/factor-model prediction and an appropriate null comparison. Searching k=2…9 necessarily returns clusters even if no discrete partition exists.
4. **Investigate interaction stage:** examine first-turn versus pressure-turn patterns and within-question score changes, with question-level uncertainty. The per-turn differences here make that especially relevant for SYCON.

## Artifacts and checks

[All 24 configurations and methods](clustering_stability/SUMMARY.md) · [Full numerical results](clustering_stability/results.json) · [Reproducible script](../../../../pipeline/cluster_syconbench_stability.py)

The run adds nine adaptive-k and nine fixed-k=3 middle-layer bootstrap heatmaps. Four regression tests check whole-question resampling, nuisance removal, agreement with the prior clustering implementation, and rejection of constant scores. All 24 raw matrices reproduce the prior results to tolerance; all bootstrap counts and 20-probe partitions are complete.
