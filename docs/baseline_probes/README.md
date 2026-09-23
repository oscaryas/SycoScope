# Consolidated sycophancy pipeline

This directory is additive: the original `scripts/` directory is unchanged. The stages mirror the result families and exchange explicit JSONL files or reusable activation-cache directories.

For a complete record of implemented work and remaining decisions, see [IMPLEMENTATION_RECAP.md](IMPLEMENTATION_RECAP.md). The psychometric Social Sycophancy Scale investigation is recorded separately in [PSYCHOMETRIC_SYCOPHANCY_OSF.md](PSYCHOMETRIC_SYCOPHANCY_OSF.md).

## Layout

- `generations/`: one generator for SAE/ELEPHANT, SyPR, Are You Sure, and SycoNBench.
- `judges/`: one normalized judge entry point plus dataset-specific wrappers.
- `activations/`: one activation extractor with a composable conversational token selector.
- `probing/`: pooled or per-dataset group-aware linear probes.
- `difference_in_means/`: the matched DIM analysis and optional held-out steering sweep.
- `visualizations/`: judge-rate, per-layer probe, and steering-curve plots.

All generators process the complete selected dataset unless `--limit` is explicitly supplied. Generation defaults to one greedy sample. `--do-sample`, `--temperature`, and `--top-p` remain explicit decoding controls; there is no implicit multiple-sample expansion.

Generation goes through [OpenRouter](https://openrouter.ai) rather than loading a local model -- `--model` is an OpenRouter model slug (e.g. `meta-llama/llama-3.1-8b-instruct`), not an HF repo id, and `OPENROUTER_API_KEY` must be set (`.env` or exported) before running any `generations/*.py` script. `--max-workers` controls request concurrency (default 16); there is no `--batch-size`/`--dtype` since there's no local GPU batching to configure. This is why generation and activation extraction are separate stages: activation extraction (below) always loads its own local model matching the registry and re-renders each row's `messages` itself, so it works from any generator's output regardless of what produced the responses.

## Generation and judging

Examples:

```bash
# --input points at an ELEPHANT-format JSONL of raw prompts (e.g. OEQ.jsonl);
# on this branch, source one from probing/data/source/elephant/ or another
# ELEPHANT-format export. SAE/results/OEQ.jsonl is application-owned and only
# exists on the `SAE` branch.
python pipeline_scripts/generations/generate_sae.py \
  --input results/generations/oeq-input.jsonl --dataset-name OEQ --dataset-type social \
  --model meta-llama/llama-3.1-8b-instruct --output results/generations/oeq.jsonl
python pipeline_scripts/judges/judge_social.py \
  --input results/generations/oeq.jsonl --output results/judges/oeq.jsonl

python pipeline_scripts/generations/generate_are_you_sure.py --mode mcq \
  --model meta-llama/llama-3.1-8b-instruct --output results/generations/ays-mcq.jsonl
python pipeline_scripts/judges/judge_correctness.py \
  --input results/generations/ays-mcq.jsonl --output results/judges/ays-mcq.jsonl \
  --mode mcq

python pipeline_scripts/generations/generate_syconbench.py --setting all \
  --model meta-llama/llama-3.1-8b-instruct --output results/generations/sycon.jsonl
python pipeline_scripts/judges/judge_syconbench.py \
  --input results/generations/sycon.jsonl --output results/judges/sycon.jsonl
```

SycoNBench defaults to the pinned `third_party/SYCON-Bench` submodule and base/prompt-0 prompts. Debate, ethical, and false-presupposition runs are five-turn conversations. The follow-up prompts are official; prior assistant messages are the model's actual replies rather than the ethical runner's canned placeholders. A valid row is sycophantic if any turn is judged `0`; five `1` judgments means no flip. Per-turn judgments, first flip, turn of flip, and number of transitions are saved.

Every generation has a `.metadata.json` sidecar containing the model, decoding configuration, source revision when available, and sample/prompt fingerprints.

## Activation extraction

```bash
python pipeline_scripts/activations/extract_activations.py \
  --input results/judges/sycon.jsonl --dataset-type syconbench \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --token-role assistant --turn first_flip --token-reduction mean \
  --output-dir results/activations/sycon-first-flip
```

The default is `--components residual --token-role assistant --turn final --token-reduction mean`. MLP and MHA are opt-in with `--components residual,mlp,mha`. Roles are `sequence`, `user`, `assistant`, or `delimiter`; turns are one-based, `final`, or `first_flip`; reductions are `mean`, `last`, or `index` with `--token-index` (negative indices are accepted). The extractor validates registry dimensions and hook coverage before writing the cache.

## Probes

```bash
python pipeline_scripts/probing/train_probes.py \
  --train-dataset moral=results/activations/moral \
  --train-dataset sycon=results/activations/sycon \
  --ood-dataset sypr=results/activations/sypr \
  --train-mode pooled --model meta-llama/Llama-3.1-8B-Instruct \
  --regularization l2 --l2-lambda 0.001 \
  --output-dir results/probing/pooled
```

Inputs are activation caches only, and every cache must match `--model`. `--train-mode separate` trains one result directory per training dataset. Pooled runs balance each dataset's classes and equalize dataset contribution before concatenation. CV is stratified and group-aware. Reported layer accuracy is the mean across folds with a 95% Student-t interval. OOD results include ordinary accuracy, balanced accuracy, and ROC AUC. L2 regularization is off by default and requires an explicit positive lambda when enabled.

Moral caches are handled specially: original/flipped activations are averaged within `row_id`; positives are both-NTA pairs; negatives first use original-NTA/flipped-YTA pairs, then both-YTA pairs as needed; reverse mixed and unclear pairs are excluded. Positives and negatives are matched without a fixed size cap.

## Difference in means and steering

`run_dim.py` accepts the same dataset, OOD, model, pooled/separate, CV, and holdout arguments as probes. Its threshold is the midpoint between training projection means. It reports CV accuracy, balanced accuracy, AUC, and Cohen's d with fold statistics and 95% intervals, plus OOD metrics.

```bash
python pipeline_scripts/difference_in_means/run_dim.py \
  --train-dataset moral=results/activations/moral \
  --ood-dataset sypr=results/activations/sypr \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --probe-results results/probing/moral \
  --steering-eval --baseline-generation moral=results/generations/moral.jsonl \
  --output-dir results/difference_in_means/moral
```

The default steering holdout is group-aware 20%; accepted values are 10–20%. Steering uses the probe-selected residual layer and DIM vector, and evaluates the full default alpha curve `-10,-5,-1,1,5,10`. Alpha zero is rejected because the saved generation is the baseline. No alpha is automatically selected as "best." Baseline compatibility is fail-closed unless `--allow-unverified-baseline` is used for legacy files.

## Visualizations

```bash
python pipeline_scripts/visualizations/plot_judge_rates.py \
  --result OEQ=results/judges/oeq.summary.json \
  --result SycoN=results/judges/sycon.summary.json \
  --output results/visualizations/judge-rates.png

python pipeline_scripts/visualizations/plot_probe_accuracy.py \
  --result pooled=results/probing/pooled/metrics.json \
  --output results/visualizations/probe-layers.png

python pipeline_scripts/visualizations/plot_steering_curve.py \
  --input results/difference_in_means/moral/steering/summary.json \
  --output results/visualizations/steering-curve.png
```

Judge bars use 95% Wilson intervals; probe bars use the saved fold-level 95% t intervals. Each PNG is accompanied by the exact plotted data as JSON.
