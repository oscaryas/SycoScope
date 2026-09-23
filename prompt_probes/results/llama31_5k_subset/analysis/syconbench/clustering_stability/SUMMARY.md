# SYCON-Bench cluster stability: numerical summary

Full-response-average scores; signed Pearson distance 1−r; average linkage; silhouette sweep k=2…9.

Question bootstrap retains all turns of each sampled question. ARI compares each reselected bootstrap partition to the original partition; 1 is identical up to cluster numbering. The bootstrap range is descriptive, not a performance CI.

| Setting / layer | Best k | Sizes | Silhouette | Bootstrap same k | Median ARI | Adjusted k | Raw–adjusted ARI | PC1 | Participation rank |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| debate_response_L00 | 2 | 18 + 2 | 0.265 | 70% | 0.307 | 2 | 1.000 | 33% | 6.41 |
| ethical_response_L00 | 2 | 19 + 1 | 0.371 | 100% | 1.000 | 2 | 0.603 | 35% | 5.52 |
| false_presupposition_response_L00 | 2 | 19 + 1 | 0.387 | 100% | 1.000 | 2 | 1.000 | 36% | 5.44 |
| debate_response_L04 | 2 | 16 + 4 | 0.377 | 94% | 0.764 | 2 | -0.109 | 36% | 5.31 |
| ethical_response_L04 | 8 | 4 + 5 + 3 + 2 + 2 + 1 + 2 + 1 | 0.267 | 12% | 0.323 | 2 | 0.157 | 30% | 6.17 |
| false_presupposition_response_L04 | 3 | 16 + 3 + 1 | 0.382 | 74% | 1.000 | 2 | 0.339 | 40% | 5.08 |
| debate_response_L08 | 2 | 19 + 1 | 0.251 | 64% | 1.000 | 2 | 1.000 | 28% | 7.17 |
| ethical_response_L08 | 2 | 19 + 1 | 0.271 | 91% | 1.000 | 2 | 1.000 | 28% | 6.90 |
| false_presupposition_response_L08 | 2 | 19 + 1 | 0.397 | 100% | 1.000 | 2 | 1.000 | 38% | 5.29 |
| debate_response_L12 | 2 | 15 + 5 | 0.314 | 100% | 0.618 | 2 | 0.618 | 28% | 6.87 |
| ethical_response_L12 | 2 | 15 + 5 | 0.372 | 100% | 1.000 | 2 | 0.603 | 30% | 6.18 |
| false_presupposition_response_L12 | 2 | 18 + 2 | 0.250 | 81% | 0.143 | 2 | 0.122 | 28% | 6.90 |
| debate_response_L16 | 2 | 15 + 5 | 0.295 | 97% | 0.618 | 2 | 1.000 | 32% | 6.22 |
| ethical_response_L16 | 2 | 19 + 1 | 0.405 | 96% | 1.000 | 2 | 1.000 | 41% | 4.58 |
| false_presupposition_response_L16 | 2 | 19 + 1 | 0.354 | 100% | 1.000 | 2 | 0.603 | 35% | 5.71 |
| debate_response_L20 | 3 | 7 + 11 + 2 | 0.301 | 19% | 0.578 | 2 | 0.578 | 31% | 6.31 |
| ethical_response_L20 | 2 | 14 + 6 | 0.367 | 99% | 1.000 | 2 | 1.000 | 36% | 5.19 |
| false_presupposition_response_L20 | 2 | 19 + 1 | 0.315 | 100% | 1.000 | 2 | 0.603 | 31% | 6.53 |
| debate_response_L24 | 2 | 19 + 1 | 0.279 | 72% | 0.190 | 2 | 1.000 | 29% | 6.87 |
| ethical_response_L24 | 2 | 15 + 5 | 0.304 | 99% | 0.382 | 2 | 0.190 | 37% | 5.07 |
| false_presupposition_response_L24 | 2 | 19 + 1 | 0.321 | 95% | 1.000 | 3 | 0.190 | 33% | 6.30 |
| debate_response_L28 | 2 | 15 + 5 | 0.276 | 84% | 0.464 | 2 | 0.603 | 29% | 6.88 |
| ethical_response_L28 | 2 | 16 + 4 | 0.357 | 76% | 1.000 | 2 | 1.000 | 37% | 4.84 |
| false_presupposition_response_L28 | 2 | 19 + 1 | 0.285 | 93% | 1.000 | 4 | 0.067 | 29% | 6.65 |

## Interpretation limits

All available questions are used descriptively, without judge-label selection. Layers reuse questions and are not independent replications. Score correlation describes probe behavior on this dataset, not orthogonality or causal circuits. Best k is conditional on searching 2–9: this procedure cannot establish that any discrete clusters exist.

The adjusted view removes linear effects of turn indicators and log(1 + response tokens), refitted inside each bootstrap sample. It does not eliminate all wording, topic, style, or system-prompt confounds, and may remove genuine behavior-associated signal. Question means average probe scores across turns, not token activations across a conversation.

Unsigned distance 1−|r| is only a same-axis sensitivity check. Participation rank is (sum eigenvalues)²/sum(eigenvalues²) of the score-correlation matrix, not an estimate of the number of sycophancy concepts. Fixed k=3 co-clustering deliberately forces three groups and is not evidence for three concepts.

Reproduce: `.venv/bin/python prompt_probes/pipeline/cluster_syconbench_stability.py --run-name llama31_5k_subset --n-boot 200 --seed 20260916`

Full memberships, correlation matrices, bootstrap consensus, per-turn, unsigned, ablation, and question-mean sensitivities are in [results.json](results.json).
