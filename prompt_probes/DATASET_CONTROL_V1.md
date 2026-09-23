# Experiment 1: change user-prompt dataset, hold model and instructions fixed

Authorized 2026-09-16. Second-model experiment is not started.

## Frozen pilot design

- Model: meta-llama/Llama-3.1-8B-Instruct, revision `0e9e39f249a16976918f6564b8830bc894c89659`.
- Existing Vast A100 instance: ssh4.vast.ai port 15883. No new instance creation or paid API judging.
- Dataset A: 200 original Perez prompts from the legacy probe suite's **unused 3,000-question pool**. None was used to train the legacy 2,000-question probes. Source-balanced sampling, then fixed 120/80 question split.
- Dataset B: 200 Dolly prompts, 25 from each of eight categories; 15/category train and 10/category report. Context is supplied where present; reference answers are never supplied. Exact normalized duplicates are removed. Prompts outside 20–2,500 characters excluded before sampling.
- Dolly: [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, CC-BY-SA-3.0. The raw source file, source hash, row IDs, and category provenance are retained. Normalized exact overlap between A and B is zero; near-duplicate and pretraining contamination are not ruled out.
- 20 positive/negative prompt pairs. Every instruction string was compared to the original saved generations and matches exactly. No instruction wording is changed in this first stage.
- Each question crosses all 40 inducing conditions plus one no-system-prompt neutral condition: **16,400 responses**.
- Fresh responses for **both A and B on the same GPU/backend**, temperature 0.6, top-p 0.9, maximum 1,024 new tokens, batch size 24, recorded deterministic per-batch seeds. Legacy OpenRouter generations are not treated as the matched-backend A arm.
- Layers 0/4/8/12/16/20/24/28, full-response mean, first-five-token mean, and last-prompt token. Residual block convention: hidden_states[layer+1]. Pool in float32. Maximum combined extraction length 4,096; overlong/empty spans are logged, never silently truncated.

## Analyses

1. Train separate probe suites on A and B, using 120 training questions per arm, training-only scaling, L2 logistic regression C=1, no report-set hyperparameter tuning. All conditions sharing a question stay in the same split.
2. Four 20×20 AUROC matrices per layer/pooling: A→A, A→B, B→A, B→B. Row = trained prompt-pair probe, column = evaluated prompt pair. Also apply the original frozen suite to A/B as a **secondary legacy comparison**.
3. Report a complete-pair/nontruncated sensitivity alongside the primary available paired-response result. Labels identify inducing instructions, not independently validated sycophantic behavior.
4. Full-response score-correlation clustering on both report-question bases: (a) the balanced mixture of inducing conditions; (b) neutral-context responses without an inducing system instruction. Each uses the same 20 probes and signed correlation distance, with k=2…9 and 200 question-bootstrap resamples. Also remove per-condition score means as a descriptive sensitivity in the prompted basis.
5. Compare cluster counts, memberships, co-clustering frequencies, and cross-dataset ARI, with explicit attention to 19+1 calibrated-hedging splits. An unchanged count alone is not replication.
6. Save checkpoints by cell, download generations/activations to the local project with activation SHA256 verification, then fit and analyze locally. GPU work is supervised; local watcher does not keep the GPU artificially busy after extraction.

## Interpretation and follow-up

This is a pilot, not a definitive sample-size-matched comparison to the legacy 1,600-training-question suite. The primary A/B probe fits are matched to each other. Source/category and generation length can change with dataset; Dolly is not designed to elicit every type of sycophancy. Absence of a pattern may reflect fewer behavioral opportunities, not absence of a concept.

The prompted-versus-neutral comparison tests how much the score covariance depends on evaluating a mixture of inducing instructions. It does not manipulate the wording used to train the probes. **A recurring pattern across datasets is not proof that system prompts caused collapse.** A fixed-definition paraphrase arm is a separate follow-up, not launched in this first dataset-control stage.

No new independent behavior judge is run, and no second model is loaded. Natural OOD benchmark results from the existing suite remain distinct from these instruction-derived transfer matrices.

## Locations

- Protocol and exact inputs: `results/llama31_dataset_control_v1/experiment.json`.
- Entry point: `pipeline/dataset_control.py` (`prepare`, `gpu`, `watch`, `analyze`).
- Local status: `results/llama31_dataset_control_v1/workflow_status.json`.
- Outputs: `analysis/transfer.json`, `analysis/prompted_clusters.json`, `analysis/neutral_clusters.json`, `analysis/RESULTS.md` within that run.

The existing HTML report remains unchanged until new results are available; the new control is not presented as completed evidence prematurely.
