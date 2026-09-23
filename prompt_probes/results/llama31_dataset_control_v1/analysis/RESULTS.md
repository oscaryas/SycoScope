# Dataset-control pilot results

Fresh matched-backend Perez versus Dolly; 120 training and 80 report questions per dataset, 20 fixed system-prompt pairs. AUROC labels identify generating instructions, not independently judged sycophancy.

| Layer / response average | Perez→Perez | Perez→Dolly | Dolly→Perez | Dolly→Dolly |
|---|---:|---:|---:|---:|
| L0 | 0.919 | 0.645 | 0.741 | 0.645 |
| L4 | 0.946 | 0.649 | 0.814 | 0.654 |
| L8 | 0.959 | 0.692 | 0.856 | 0.697 |
| L12 | 0.968 | 0.727 | 0.917 | 0.783 |
| L16 | 0.989 | 0.877 | 0.987 | 0.888 |
| L20 | 0.988 | 0.853 | 0.970 | 0.865 |
| L24 | 0.979 | 0.819 | 0.944 | 0.837 |
| L28 | 0.980 | 0.799 | 0.945 | 0.835 |

Entries above are medians of off-diagonal cell AUROCs, not pooled AUROC and not concept counts. Inspect full matrices and control polarities before interpretation.

## Neutral-context clustering

| Training probes / layer | Perez neutral best k | Dolly neutral best k | Membership ARI |
|---|---:|---:|---:|
| perez L0 | 2 | 2 | 1.000 |
| dolly L0 | 2 | 2 | 0.525 |
| legacy L0 | 2 | 3 | 0.091 |
| perez L4 | 2 | 2 | 1.000 |
| dolly L4 | 2 | 2 | 0.433 |
| legacy L4 | 2 | 2 | 0.021 |
| perez L8 | 2 | 2 | 1.000 |
| dolly L8 | 2 | 2 | 0.081 |
| legacy L8 | 2 | 3 | 0.255 |
| perez L12 | 2 | 2 | 0.246 |
| dolly L12 | 3 | 2 | 0.117 |
| legacy L12 | 2 | 3 | 0.041 |
| perez L16 | 2 | 2 | 0.603 |
| dolly L16 | 2 | 2 | 0.398 |
| legacy L16 | 2 | 2 | 0.603 |
| perez L20 | 2 | 2 | 0.603 |
| dolly L20 | 2 | 2 | 0.274 |
| legacy L20 | 2 | 2 | 0.603 |
| perez L24 | 2 | 2 | 0.603 |
| dolly L24 | 2 | 2 | 0.719 |
| legacy L24 | 2 | 2 | 1.000 |
| perez L28 | 2 | 2 | 0.525 |
| dolly L28 | 3 | 3 | 0.807 |
| legacy L28 | 2 | 2 | 0.603 |

Question-bootstrap stability and membership arrays are in neutral_clusters.json. Matched comparisons across the mixture of inducing prompt conditions, plus condition-mean-removal sensitivity, are in prompted_clusters.json. Both bases use report questions only. Searching k=2–9 always returns clusters; this does not establish that discrete concepts exist.

System-prompt wording is held fixed, not experimentally removed. A surviving pattern supports dataset robustness, not proof that prompts cause collapse. Paraphrase testing is a distinct follow-up. Original-versus-fresh legacy comparisons additionally change the generation backend and training sample size.
