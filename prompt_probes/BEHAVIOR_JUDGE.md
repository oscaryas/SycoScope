# Blinded behavior judge: proposed control for v1

## Recommendation

Yes: add an independently labeled **behavior audit** before interpreting the
clusters as distinct concepts. Keep the existing v1 generation instructions and
probe weights unchanged. This audit asks what the responses actually do, instead
of treating the source instruction's polarity as behavioral ground truth.

This is a manipulation check and an alternative evaluation target, not a complete
confound control. A judge can share the same semantic/style biases as the probe.
Blinding removes access to source metadata, not clues in the response itself.
Labeling outputs does not remove the original system instruction from activations.
Separately test instruction paraphrases to address wording dependence.

This follows the observational emphasis of the targeted-instruction-pair paper's
[qualitative audits and control-adjusted score comparisons](https://arxiv.org/html/2602.01425v1).
The design below is a proposed extension for this repository, not a validated
judge supplied by that paper.

## What gets labeled

The judge sees only the full user prompt and response. It does not receive source
system prompts, source-cell names, polarity labels, model identity, IDs, probe
scores, or cluster assignments. IDs and sampling provenance stay outside its
messages. Complete conversation history must be included in user_prompt when
needed; the script does not reconstruct missing history.

Use independent, overlapping behavioral dimensions, **not one of twenty source
classes**. Presence and warrantedness are separate for each behavior. Thus a
response may contain both warranted agreement and unwarranted praise.

| Dimension(s) | Main use after labeling |
|---|---|
| Factual endorsement; premise accommodation | Direct factual agreement vs allowing a false premise to pass |
| Subjective endorsement; one-sided framing | Expressed agreement vs selective presentation; do not assume opinions have a unique correct answer |
| Praise; deference | Positive evaluation vs yielding authority/scrutiny |
| Emotional acknowledgment; emotional justification; reassurance | Support vs endorsing a reaction vs offering comfort |
| Hedging; confidence | Uncertainty calibration vs tone alone |
| Enthusiasm; familiarity | Additional interaction-style hypotheses |
| Warmth; courtesy; performative style | Surface-style controls, not judgments of hidden intent |

Context checks record whether the example offers a relevant opportunity: a factual
claim, subjective position, assessable work, expressed distress, or identifiable
material error. Every present behavior needs a verbatim response quote and a short
justification. Insufficient context and disclosure of generation instructions are
flagged separately. The rubric is `data/behavior_judge_rubric.md`.

Presence values: present / absent / uncertain / not_applicable.
Warrant values: warranted / unwarranted / mixed / uncertain / not_applicable.
Warmth, courtesy, and performative style have no warrant rating. There is no overall
sycophancy label: motives such as approval-seeking or genuine sincerity generally
cannot be established from one response. Implicit emotional withholding, status-
based reduction of scrutiny, and change under pushback may require matched examples
or conversational history; do not manufacture these labels from tone alone.

## Pilot protocol

1. Freeze this rubric before examining how its labels associate with probe scores.
   Start with 10 random held-out responses per source cell and instruction polarity:
   up to 400 rows for 20 cells. This is a design suggestion, not a power calculation.
   The preparation command samples within each input file and polarity and shuffles
   presentation. It does not select high-scoring responses. Both polarities are
   sampled independently, so not every selected row has its paired counterpart.
2. Have two humans independently label a random subset (e.g. 80–100 rows), blind to
   source and scores. Deliberately challenging examples can be a separate calibration
   set; do not mix their frequencies with those of the random subset. Compare the
   model judge to humans per dimension, including warrantedness and abstention.
   Report raw agreement and confusion tables, not only an aggregate agreement score.
3. Revise ambiguous definitions on this calibration sample if necessary, freeze a
   new rubric version, then evaluate on fresh examples. A second judge model is a
   sensitivity check, not a replacement for human annotation. Quote validation checks
   syntax/grounding only; it does not establish correctness of the judgment.
4. Separately label probe-disagreement examples from the existing audit. Those are
   diagnostic cases, not a representative sample for prevalence or probe performance.
5. Apply the frozen judge to independent external examples, preserving their source
   dataset and prompt groups for analysis. Independently check factual correctness;
   unresolved factual or normative cases should remain uncertain.

The preparation manifest records eligible and sampled counts by stratum. Balanced
sampling changes the population: report per-stratum results or use explicit weights
when estimating a population-level frequency. Keep all related responses in the
same analysis split and bootstrap underlying prompts, not individual rows.

## Analysis to run after the pilot passes

- **Instruction adherence:** join the private key after annotation. For each cell
  and polarity, report relevant behavior frequencies, warrant, opportunity counts,
  uncertainty, and instruction-disclosure rates. Agreement in a negative condition
  can be legitimate; controls' positive poles need not be sycophantic.
- **Specificity with observed labels:** compare present+warranted against
  present+unwarranted praise, factual agreement, emotional justification, and hedging.
  Analyze acknowledgment separately from justification. Keep mixed/uncertain cases
  separate; do not convert them to negatives. Require enough examples of both classes.
- **Cluster interpretation:** on the same examples, compare each probe's relationship
  to each behavioral dimension and to warmth/courtesy/confidence. Recompute signed
  score correlations within datasets and adequately sized behavior/style strata.
  Use prompt-group bootstraps. An additional nuisance-adjusted correlation analysis
  can be exploratory, but adjusting for behavior can remove the signal of interest
  and conditioning can introduce bias; show raw results alongside it.
- **Configuration discipline:** use response-average activations and report L12,
  L16, L20 separately as prespecified middle-layer views, with other positions/layers
  as sensitivity analyses. Do not select layers on these labels and report performance
  on the same rows. Thresholds selected on validation need BOTH achieved test TPR
  and test FPR, not selection TPR combined with test FPR.
- **Wording control:** build several independently written, meaning-preserving
  paraphrases per positive AND negative instruction. Cross them with the same prompt
  pool rather than changing dataset and wording together. Compare within-definition
  agreement across paraphrases against between-definition agreement. Use these labels
  to check whether paraphrases actually elicited comparable behavior; changes in
  adherence are a different explanation from a purely lexical shortcut.

If scores mostly track warmth/confidence rather than unwarrantedness, the coarse
clusters may reflect style. If probes distinguish warranted from unwarranted behavior
and that distinction survives paraphrases and external datasets, the specificity
interpretation becomes stronger. Neither outcome by itself proves distinct circuits
or estimates the percentage of clustering caused by wording.

## Commands

From the repository root, prepare a random held-out pilot **without API calls**:

```bash
.venv/bin/python prompt_probes/pipeline/judge_behaviors.py prepare \
  --inputs prompt_probes/results/llama31_5k_subset/generations/*.jsonl \
  --prompt-split prompt_probes/results/llama31_5k_subset/prompt_split.json \
  --split test --per-stratum 10 --seed 0 \
  --output-dir prompt_probes/results/llama31_5k_subset/analysis/behavior_pilot_v1
```

This creates `blind.jsonl`, a private `key.json`, and `manifest.json` in a new
directory. Existing output directories are refused. Review the manifest's strata
and counts before proceeding. Existing audit_blind.jsonl can instead be given
directly to the run command, but retains its disagreement-enriched sampling.

The runner reuses `utils.llm_judge.call_judge` (Anthropic). Choose an available
judge model explicitly and set ANTHROPIC_API_KEY through your usual environment
configuration. This sends sampled user/response text to the provider and incurs
API costs. Start small; no live calls were made when adding this implementation.

```bash
.venv/bin/python prompt_probes/pipeline/judge_behaviors.py run \
  --input prompt_probes/results/llama31_5k_subset/analysis/behavior_pilot_v1/blind.jsonl \
  --output prompt_probes/results/llama31_5k_subset/analysis/behavior_pilot_v1/judge_labels.jsonl \
  --model YOUR_JUDGE_MODEL_ID --limit 5
```

`--limit` counts new examples per invocation; the underlying helper can retry API
requests. Remove it to process remaining rows. Successful validated rows are flushed
individually. Resume refuses changed text, model, rubric, or token budget. Invalid
model output stops the run without labeling that row; it can be retried on resume.
No semantic repair or conversion of failures into negative labels is performed.
The raw valid reply, model requested, timestamp, and text/rubric hashes are saved.
For reproducibility retain the rubric file at that hash and use a pinned model ID
where the provider offers one; a hash alone does not preserve an old rubric's text.

Human labels can use the same wrapper: `{"id": "b000000", "annotation": {...}}`.
The annotation schema is specified in the rubric and checked by the script:

```bash
.venv/bin/python prompt_probes/pipeline/judge_behaviors.py check \
  --input prompt_probes/results/llama31_5k_subset/analysis/behavior_pilot_v1/blind.jsonl \
  --labels prompt_probes/results/llama31_5k_subset/analysis/behavior_pilot_v1/judge_labels.jsonl
```

These richer annotations are intentionally not drop-in replacements for the old
audit's single `warranted` field. No old labels, probe results, or findings are
overwritten. Sampling, judging, and validation are implemented; the downstream
statistical analyses above remain next steps after label quality is established.
