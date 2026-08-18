# Pooling variant comparison: full response vs. first-K-tokens vs. first-sentence

All probes are residual-stream-only linear probes on `meta-llama/Meta-Llama-3-8B-Instruct`,
5-fold CV, trained per-category (each dataset capped to its own `min(n_pos, n_neg)`,
not the shared mixture cap) except the last row, which is the combined 4-way mixture
probe (`sycophancy_mixture_residual*`, trained on all 4 categories pooled, forced 50/50).

Three pooling modes, all averaging over the response span (tokens after the
`answer_token_id` chat-template delimiter, i.e. response-only, not prompt+response):

- **mean (full response)** — mean over every response token
- **mean_first5** — mean over just the first 5 response tokens
- **mean_first_sentence** — mean over the response's first sentence (found via an
  incremental decode-and-check-for-`.`/`!`/`?` heuristic; not abbreviation/decimal-aware)

## Same-dataset performance (5-fold CV)

| Probe | Pooling | Best layer | Accuracy | Accuracy 95% CI | AUC-ROC | AUC-ROC 95% CI |
|---|---|---|---|---|---|---|
| sypr | mean (full response) | 31 | 0.960 | [0.950, 0.968] | 0.991 | [0.987, 0.996] |
| sypr | mean_first5 | 22 | 0.956 | [0.945, 0.964] | 0.986 | [0.982, 0.991] |
| sypr | mean_first_sentence | 27 | **0.967** | [0.957, 0.974] | **0.992** | [0.989, 0.995] |
| are_you_sure | mean (full response) | 14 | **0.758** | [0.739, 0.776] | **0.838** | [0.828, 0.848] |
| are_you_sure | mean_first5 | 13 | 0.683 | [0.662, 0.703] | 0.742 | [0.723, 0.758] |
| are_you_sure | mean_first_sentence | 13 | 0.680 | [0.659, 0.700] | 0.740 | [0.731, 0.748] |
| social | mean (full response) | 14 | **0.846** | [0.829, 0.861] | **0.929** | [0.922, 0.936] |
| social | mean_first5 | 14 | 0.641 | [0.620, 0.662] | 0.687 | [0.673, 0.700] |
| social | mean_first_sentence | 11 | 0.774 | [0.756, 0.792] | 0.846 | [0.830, 0.863] |
| truthfulqa | mean (full response) | 13 | **0.774** | [0.757, 0.791] | **0.859** | [0.844, 0.874] |
| truthfulqa | mean_first5 | 15 | 0.754 | [0.736, 0.772] | 0.828 | [0.820, 0.838] |
| truthfulqa | mean_first_sentence | 29 | 0.749 | [0.731, 0.767] | 0.817 | [0.802, 0.829] |
| moral_avg | mean (full response) | 13 | **0.758** | [0.726, 0.788] | **0.840** | [0.805, 0.876] |
| moral_avg | mean_first5 | 31 | 0.716 | [0.683, 0.748] | 0.797 | [0.746, 0.846] |
| moral_avg | mean_first_sentence | 30 | 0.714 | [0.680, 0.745] | 0.785 | [0.740, 0.832] |
| sycophancy_mixture (4-way combined) | mean (full response) | 14 | **0.760** | [0.743, 0.775] | **0.848** | [0.839, 0.858] |
| sycophancy_mixture (4-way combined) | mean_first5 | 15 | 0.673 | [0.655, 0.690] | 0.759 | [0.752, 0.769] |
| sycophancy_mixture (4-way combined) | mean_first_sentence | 13 | 0.699 | [0.682, 0.716] | 0.788 | [0.770, 0.802] |

**Bold** = best of the three variants for that probe. sypr is the only probe where a
truncated-window pooling *beats* the full-response mean; every other probe does best
with the full response.

## Cross-dataset generalization

Only run for the **mean (full response)** variant so far — the `mean_first5` /
`mean_first_sentence` probes above have not been transfer-tested (their weight files
weren't pulled from the training VM). All numbers below are zero-shot: a frozen probe
scored directly on a dataset it was never trained on, no retraining.

### Single-category probe → 4-way mixture

| Probe (frozen, full-response, layer) | Target | Accuracy | Accuracy CI | AUC-ROC | AUC-ROC CI |
|---|---|---|---|---|---|
| sypr (31) | mixture overall | 0.609 | [0.591, 0.627] | 0.638 | [0.617, 0.657] |
| sypr (31) | mixture, sypr rows only (own category) | **0.983** | [0.971, 0.990] | **0.995** | [0.990, 0.999] |
| social (14) | mixture overall | 0.606 | [0.588, 0.623] | 0.619 | [0.598, 0.640] |
| social (14) | mixture, social rows only | 0.872 | [0.845, 0.894] | 0.953 | [0.938, 0.966] |
| truthfulqa (13) | mixture overall | 0.547 | [0.529, 0.565] | 0.624 | [0.603, 0.642] |
| truthfulqa (13) | *(not a mixture category — no own-category cell)* | — | — | — | — |
| are_you_sure (14) | mixture overall | 0.627 | [0.609, 0.645] | 0.670 | [0.651, 0.690] |
| are_you_sure (14) | mixture, are_you_sure rows only | 0.873 | [0.847, 0.896] | 0.949 | [0.934, 0.964] |
| moral_avg (13) | mixture overall | 0.626 | [0.608, 0.644] | 0.643 | [0.623, 0.664] |
| moral_avg (13) | mixture, moral rows only | 0.685 | [0.649, 0.718] | 0.733 | [0.696, 0.769] |

Every probe generalizes strongly to its own category inside the mixture but is close
to chance (or worse) on the mixture overall — each single-category direction is fairly
specific to its own category, not a general "sycophancy" direction.

### Combined mixture probe (layer 14) → each category's full, uncapped dataset (true OOD)

| Target | n (natural prevalence) | Accuracy | Accuracy CI | AUC-ROC | AUC-ROC CI |
|---|---|---|---|---|---|
| sypr | 10,792 (~8.6% pos) | 0.830 | [0.823, 0.837] | **0.932** | [0.924, 0.941] |
| social | 8,804 (~88.7% pos) | 0.751 | [0.742, 0.760] | **0.902** | [0.893, 0.911] |
| are_you_sure | 2,217 (~44.7% pos) | 0.675 | [0.655, 0.694] | 0.754 | [0.733, 0.773] |
| truthfulqa | 3,457 (~32.1% pos) | 0.683 | [0.667, 0.698] | 0.681 | [0.664, 0.698] |
| moral | 1,153 (~32.1% pos) | 0.463 | [0.435, 0.492] | 0.812 | [0.787, 0.838] |

moral's accuracy (0.463) sits *below* its 0.679 majority-class baseline despite a
strong AUC (0.812) — a calibration artifact (the probe's decision threshold is set for
the mixture's forced 50/50 balance, not moral's real ~32%-positive prevalence), not an
absence of signal. AUC is the fairer metric whenever a target's natural class balance
differs sharply from the training balance, which is true for all five targets here.
