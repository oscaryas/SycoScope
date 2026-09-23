# System-prompt versus question activation variance

400 matched held-out Perez questions × 40 system-prompt conditions (20 pairs, both poles) = 16,000 cached responses. Llama-3.1-8B-Instruct; full-response-mean residual-stream activations.

## Method

At each layer: h[s,q] = grand mean + system mean deviation + question mean deviation + interaction/residual. Sum squared Euclidean deviations across all 4096 raw activation coordinates. System SS = Q × sum_s ||mean_q(h[s,q]) − grand||²; question SS = S × sum_q ||mean_s(h[s,q]) − grand||². Residual SS is the squared norm left after subtracting both main effects. These sum to total SS in this balanced design.

Intervals: 200 question-level bootstrap resamples, retaining all 40 conditions per sampled question and recomputing all means. They describe question-sampling uncertainty conditional on these fixed prompts and realized generations, not generation-seed uncertainty. No F-tests or ANOVA p-values are used.

## Results (% of total activation variation)

| Layer | System [95% interval] | Question [95% interval] | Interaction + noise [95% interval] |
|---|---:|---:|---:|
| response_L00 | 11.93 [11.62, 12.55] | 36.08 [34.92, 37.19] | 51.99 [50.47, 53.17] |
| response_L04 | 12.94 [12.64, 13.57] | 42.46 [41.25, 43.49] | 44.60 [43.23, 45.79] |
| response_L08 | 13.04 [12.72, 13.68] | 43.96 [43.06, 44.85] | 42.99 [41.86, 44.00] |
| response_L12 | 16.15 [15.73, 16.93] | 39.37 [38.54, 40.15] | 44.48 [43.45, 45.38] |
| response_L16 | 18.04 [17.59, 18.84] | 43.79 [42.88, 44.59] | 38.17 [37.12, 39.16] |
| response_L20 | 18.47 [18.08, 19.19] | 45.74 [44.83, 46.60] | 35.79 [34.75, 36.73] |
| response_L24 | 17.11 [16.75, 17.78] | 49.62 [48.64, 50.51] | 33.27 [32.16, 34.25] |
| response_L28 | 18.20 [17.80, 18.93] | 47.64 [46.68, 48.57] | 34.16 [32.93, 35.20] |

## Nonrepetitive complete-question sensitivity

Retain only the 379 questions with no repetitive flag in any condition. This is an outcome-selected sensitivity, not the primary estimate. Refusals, short answers and truncations are retained.

| Layer | System % | Question % | Interaction + noise % |
|---|---:|---:|---:|
| response_L00 | 12.12 | 35.86 | 52.02 |
| response_L04 | 13.12 | 42.25 | 44.63 |
| response_L08 | 13.18 | 43.87 | 42.95 |
| response_L12 | 16.31 | 39.29 | 44.41 |
| response_L16 | 18.22 | 43.65 | 38.13 |
| response_L20 | 18.66 | 45.56 | 35.78 |
| response_L24 | 17.30 | 49.41 | 33.28 |
| response_L28 | 18.41 | 47.43 | 34.16 |

## Interpretation and limits

- This is activation-variance attribution across the chosen conditions, not a percentage of response meaning, and not an estimate of presence-versus-absence of a system prompt. No neutral/no-system condition is present in this original cache. general_baseline is a substantive positive/negative prompt pair, not a neutral assistant.
- Only one generation exists per system–question cell: system×question interaction, sampling noise and unmodeled generation effects cannot be separated. The system main effect alone is not the whole system-prompt effect.
- Response text varies across conditions. Style, content and length changes contribute to these activations. Question effects include topic and other properties of the exact question, not just its semantic intent.
- Raw-coordinate variance weights high-variance activation directions more heavily. This descriptive metric does not establish behaviorally important or causal dimensions. Percentages have a separate denominator at each layer.
- results.json also contains per-pair two-condition decompositions and each condition's contribution to the global system sum of squares. Per-pair percentages have different denominators; they are not an additive breakdown of the global percentages.
- This is the original Perez dataset, not the still-running matched-backend Perez/Dolly dataset-control experiment. No probe fitting, generation, GPU use or API calls were needed.
