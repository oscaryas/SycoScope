# SYCON-Bench-derived OOD results

Frozen 20-probe Llama-3.1-8B suite; fresh base-scenario generations, actual five-turn histories, Claude Sonnet 5 labels.

Configurations and the winning probe are chosen on selection questions; AUCs below use disjoint report questions. CIs bootstrap questions, retaining all related turns.

| Target | Selected probe | Position/layer | Report AUC (95% CI) | Turns / questions |
|---|---|---|---|---|
| debate_failure | ctrl_evasive_hedging | response L4 | 0.600 (0.493–0.703) | 350 / 70 |
| debate_first_failure | ps_explicit | response L0 | 0.463 (0.333–0.602) | 231 / 70 |
| ethical_failure | pt_implicit | response L16 | 0.740 (0.685–0.791) | 700 / 140 |
| ethical_first_failure | pe_implicit | last_prompt L0 | 0.549 (0.417–0.681) | 89 / 43 |
| false_presupposition_failure | ctrl_excitement | response L12 | 0.644 (0.586–0.692) | 695 / 139 |
| false_presupposition_first_failure | ctrl_excitement | response L12 | 0.573 (0.458–0.688) | 130 / 56 |

## Fixed response-average middle-layer views

Each cell uses the probe selected on selection questions at that fixed layer. These are sensitivity views, not additional untouched replications.

| Target | L12 | L16 | L20 |
|---|---|---|---|
| debate_failure | 0.638 (ctrl_calibrated_hedging) | 0.646 (ctrl_calibrated_hedging) | 0.638 (ctrl_calibrated_hedging) |
| debate_first_failure | 0.680 (ctrl_emotional_support) | 0.603 (ctrl_calibrated_hedging) | 0.617 (ctrl_calibrated_hedging) |
| ethical_failure | 0.703 (pt_implicit) | 0.740 (pt_implicit) | 0.746 (pv_implicit) |
| ethical_first_failure | 0.635 (ctrl_calibrated_hedging) | 0.650 (ctrl_calibrated_hedging) | 0.672 (ctrl_calibrated_hedging) |
| false_presupposition_failure | 0.644 (ctrl_excitement) | 0.622 (ctrl_pure_sycophancy) | 0.591 (ctrl_pure_sycophancy) |
| false_presupposition_first_failure | 0.573 (ctrl_excitement) | 0.602 (ctrl_warranted_praise) | 0.600 (ctrl_warranted_praise) |

## Interpretation and artifacts

Current-turn failure includes initial failure. First-failure targets include only turns 2–5 after an entirely successful prior history. Debate failure means not maintaining an assigned stance; it is not necessarily factual error or unjustified agreement.

Response-average activations cover only the current answer. They measure detection/decodability, not prediction before that answer. See per-turn AUCs and the turn-number baseline in details.json; changing failure prevalence over turns can inflate pooled AUC.

The separate setting-specific correlation heatmaps/dendrograms use all available turns and are exploratory. Layers reuse the same questions. Cluster counts are not independent concepts or circuits. The subsequent [cluster analysis](CLUSTER_ANALYSIS.md) adds 200 question-bootstrap draws per configuration, turn/length adjustment, and middle-layer sensitivity checks.

Source reproduction and judge caveats are documented in prompt_probes/SYCONBENCH_EVAL.md. Main matrix: analysis/matrix_syconbench.json. Detailed middle-layer, per-turn, and clustering results: analysis/syconbench/details.json.
