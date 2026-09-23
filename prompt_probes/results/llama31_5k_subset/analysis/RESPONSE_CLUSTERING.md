# Full-response probe-score clustering

Completed 2026-09-15 for `meta-llama/Llama-3.1-8B-Instruct`.

## Scope and method

- All 20 saved instruction-pair probes, at layers 0/4/8/12/16/20/24/28.
- `response` means the mean activation over the complete assistant response span used by the existing extraction pipeline. The probe at each layer was trained at that same position.
- Each probe scores the same examples within each evaluation basis. Pearson score correlations are clustered with average linkage and signed distance `1 - r`.
- Each JSON retains the existing fixed-five-cluster signed and unsigned views for comparison with prior outputs, plus the signed silhouette sweep over **k = 2 through 9**.
- Dendrograms show the signed-distance tree without imposing a fixed cluster cut. Heatmaps retain the taxonomy ordering.
- No probe retraining, generation, or activation extraction was needed.

| Evaluation basis | Examples | Interpretation |
|---|---:|---|
| SyPR | 1,848 | External praise evaluation |
| AITA moral | 4,441 | External moral evaluation; multiple constituent datasets/labels |
| ELEPHANT | 8,797 | External social evaluation; multiple constituent datasets/labels |
| Cross-cell | 4,000 | Synthetic instruction-condition transfer, retaining inducing instructions |

These are four separately scored bases, not four independent naturalistic benchmarks. Correlations use all cached examples within each basis, including the existing selection and report partitions, matching the earlier clustering method. This is an exploratory comparison, not an untouched confirmatory evaluation. AITA and ELEPHANT are pooled within their respective caches; within-source and within-label analyses remain useful follow-ups.

## Selected cluster count by layer

Each entry is **best k (silhouette)** among the tested counts. These are not the fixed-five-cluster assignments also stored in the files.

| Layer | SyPR | AITA moral | ELEPHANT | Synthetic cross-cell |
|---|---|---|---|---|
| 0 | 4 (0.352) | 2 (0.379) | 2 (0.398) | 2 (0.646) |
| 4 | 2 (0.239) | 2 (0.353) | 2 (0.371) | 2 (0.656) |
| 8 | 2 (0.302) | 3 (0.301) | 2 (0.275) | 2 (0.676) |
| 12 | 2 (0.302) | 2 (0.305) | 2 (0.266) | 2 (0.647) |
| 16 | 4 (0.374) | 2 (0.367) | 2 (0.315) | 2 (0.743) |
| 20 | 2 (0.417) | 2 (0.376) | 2 (0.317) | 2 (0.718) |
| 24 | 9 (0.340) | 2 (0.352) | 6 (0.305) | 2 (0.703) |
| 28 | 7 (0.412) | 3 (0.379) | 6 (0.338) | 2 (0.697) |

Full-response best-k counts: **2: 24/32; 3: 2/32; 4: 2/32; 6: 2/32; 7: 1/32; 9: 1/32**.

Across the three external bases alone, k=2 wins **16/24 (67%)**. Synthetic cross-cell contributes eight additional k=2 results. SyPR at layer 24 selects the upper boundary of the sweep (k=9), so this does not locate a definitive optimum over all possible cluster counts.

## Comparison with the previously analyzed positions

Counts below are across the same four bases and eight layers per position.

| Position | Configurations | Best-k distribution | k=2 frequency |
|---|---:|---|---:|
| `last_prompt` | 32 | 2:19, 3:10, 4:2, 5:1 | 59% |
| `first5` | 32 | 2:25, 3:2, 4:2, 5:2, 6:1 | 78% |
| `response` | 32 | 2:24, 3:2, 4:2, 6:2, 7:1, 9:1 | 75% |

There are now **96 correlation JSON files**, including 32 new full-response files. The new run also adds 32 full-response heatmaps and 32 full-response dendrograms. Repeated layers and positions reuse data and probes; these frequencies are descriptive, not confidence levels or independent replications.

## What the memberships show

### Coarse groups recur, but k=2 can mean an isolated outlier

In synthetic cross-cell, seven of eight response configurations separate `ctrl_calibrated_hedging` alone from the other 19 probes. Layer 4 instead has an 18-versus-2 split. On ELEPHANT, layers 0 and 4 also isolate calibrated hedging from the other 19.

Thus a best count of two cannot automatically be described as two coherent sycophancy concepts. The memberships and correlations within the large group matter.

### ELEPHANT at layer 20 reproduces a broad emotion/support division

At its selected k=2:

- **Emotion/support group (7):** `pe_explicit`, `pe_implicit`, `ctrl_warranted_praise`, `ctrl_genuine_agreement`, `ctrl_emotional_support`, `ctrl_calibrated_hedging`, `ctrl_politeness`.
- **Other probes (13):** the baseline, all position probes, both traits probes, pure sycophancy, obsequiousness, excitement, evasive hedging, affiliation, and inauthenticity.

This is a descriptive partition. It does not imply every member of either group has strongly correlated scores with every other member.

[Layer 20 correlations](score_correlation_response_L20_eval_elephant.json) · [Heatmap](../plots/score_correlation_response_L20_eval_elephant.png) · [Dendrogram](../plots/dendrogram_response_L20_eval_elephant.png)

### Later layers show additional structure

ELEPHANT layer 24 selects six groups:

1. `general_baseline`, `pv_explicit`, `pe_explicit`, `pe_implicit`, `ctrl_emotional_support`, `ctrl_politeness`.
2. `pv_implicit`, `ps_implicit`, `ctrl_evasive_hedging`.
3. `ps_explicit`, `pt_explicit`, `pt_implicit`, `ctrl_pure_sycophancy`, `ctrl_obsequiousness`, `ctrl_inauthenticity`.
4. `ctrl_warranted_praise`, `ctrl_genuine_agreement`.
5. `ctrl_calibrated_hedging`.
6. `ctrl_excitement`, `ctrl_affiliation`.

Layer 28 retains a similar six-group pattern, with `ps_explicit` moving into the implicit-position/evasive-hedging group. SyPR prefers nine groups at layer 24 and seven at layer 28. These results qualify a blanket claim that the taxonomy collapses to two or three groups at every depth.

[ELEPHANT layer 24 dendrogram](../plots/dendrogram_response_L24_eval_elephant.png) · [ELEPHANT layer 28 dendrogram](../plots/dendrogram_response_L28_eval_elephant.png) · [SyPR layer 24 dendrogram](../plots/dendrogram_response_L24_eval_sypr.png) · [SyPR layer 28 dendrogram](../plots/dendrogram_response_L28_eval_sypr.png)

## Interpretation and remaining tests

Full-response averaging preserves coarse shared detection patterns while revealing additional, dataset-dependent structure at later layers. It does **not** establish a universal number of sycophancy concepts or independent activation dimensions. Best-k values and semantic names for groups remain exploratory until example-level bootstrap stability, prompt-paraphrase replication, and independently behavior-labeled evaluation are available.

Full-response scores can also reflect response wording, length, and style. These correlations alone do not distinguish those effects from behavioral specificity. The existing specificity and label-quality concerns are not resolved by adding this analysis.

## Reproduction and cache provenance

Run from the repository root:

```bash
MPLCONFIGDIR=/private/tmp/sycoscope-response-matplotlib \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python prompt_probes/pipeline/analyze_probes.py \
  --run-name llama31_5k_subset \
  --positions response \
  --cluster-only \
  --cluster-basis eval_sypr eval_moral eval_elephant eval_cross_cell
```

The SyPR, moral, and cross-cell activation archives had been moved to `/Volumes/Expansion/sycoscope/results/llama31_5k_subset/`. Local symlinks now reconnect them to their original result directories; the drive must be mounted when rerunning. ELEPHANT's activation archive is local. Archive dimensions match the local indices (1,848 / 4,441 / 4,000 rows respectively, 4,096 features). Recomputed `first5_L20` correlations from all three reconnected archives reproduce the existing saved correlations to numerical tolerance, checking continuity with the earlier analysis.

Validation checked all 32 new matrices for finite, symmetric 20-by-20 correlations and unit diagonals; all k=2..9 partitions contain each probe exactly once; all 32 heatmaps and 32 dendrograms were produced. This run did not recompute unrelated ANOVA, transfer, specificity, or reliability-ceiling analyses.
