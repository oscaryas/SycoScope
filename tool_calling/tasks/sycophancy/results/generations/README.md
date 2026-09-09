# Llama-3.1-8B-Instruct generations

Raw generations and LLM-judged outputs from `meta-llama/Llama-3.1-8B-Instruct`
used by the sycophancy probing experiments. Judge labels were produced with
`claude-sonnet-5` (see `../../*_judge.py`).

Files are plain JSONL except the SyPr checkpoint, which is gzipped because the
raw file (104 MB) exceeds GitHub's per-file limit. Read it with
`gzip.open(path, "rt")`.

| Set | File | Rows | Notes |
|---|---|---|---|
| Social sycophancy, SS | `SS_social_sycophancy_judged.jsonl` | 3,777 | ELEPHANT metrics: validation / indirectness / framing |
| Social sycophancy, OEQ | `OEQ_social_sycophancy_judged.jsonl` | 3,027 | same |
| Social sycophancy, AITA-YTA | `AITA-YTA_social_sycophancy_judged.jsonl` | 2,000 | same |
| Social pooled | `social_sycophancy_pooled/pooled.jsonl` | 8,804 | SS + OEQ + AITA-YTA with probe `text`/`label` |
| Moral sycophancy | `AITA-NTA-FLIP_moral_sycophancy_judged.jsonl` | 1,591 | original vs flipped-story verdicts |
| Moral pooled | `moral_sycophancy_pooled/pooled.jsonl` | 1,591 | probe-ready |
| Verdict only (exploratory) | `AITA-NTA-OG_verdict_judged.jsonl`, `AITA-YTA_verdict_judged.jsonl` | 1,591 / 2,000 | NTA / YTA / OTHER |
| SyPr praise | `sypr_praise_llama31_full/checkpoint.jsonl.gz` | 10,794 | 10,800 eligible, 6 skipped; see `summary.json` |
| Are-you-sure, free-form | `are_you_sure_freeform/checkpoint.jsonl` | 1,090 | TriviaQA, turn-2 correctness |
| Are-you-sure, MMLU MC | `mmlu_are_you_sure_full/checkpoint.jsonl` | 450 | multiple choice with CoT |
| Are-you-sure, MC extra | `are_you_sure_mc_extra/checkpoint.jsonl` | 677 | additional MC items |
| Are-you-sure, MCQ merged | `are_you_sure_mcq/mcq.jsonl` | 1,127 | MMLU full + MC extra |
| Are-you-sure pooled | `are_you_sure_pooled/pooled.jsonl` | 2,217 | MCQ + free-form |
| SyConBench | `syconbench/raw_generations.jsonl`, `syconbench/conversations.jsonl` | 500 | authors' published Llama-3.1-8B-Instruct generations, prompt0 condition |
| Sycophancy mixture | `sycophancy_mixture/mixture.jsonl` | 2,960 | 370 per category per label from sypr / are_you_sure / social / moral |

Not included here: `truthfulqa_sycophancyeval/` and `dissociating_sycophancy/`
(generated with `Meta-Llama-3-8B-Instruct`, not 3.1) and the
`mixture_categories_steering*/` folders (steered generations, not baseline).
