"""
ActivationSteerer-dependent SyPR generation+labeling step.

Split out of the old sypr_data.py during the shared-probing-library
extraction: the pure dataset/prompt helpers it depends on
(load_sypr_dataset, stratified_sample, is_label_eligible, build_chat_messages,
...) now live in probing.data.sypr_data since they need no live model. This
function does need one -- it instantiates sycophancy_steering.ActivationSteerer
to generate fresh (unsteered) baseline responses -- so it stays
application-owned here rather than moving to the shared package.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.data.sypr_data import is_label_eligible, is_poor_quality, load_sypr_dataset, stratified_sample


def generate_and_label_sypr(
    model,
    tokenizer,
    model_config: dict,
    n_train: int,
    seed: int = 0,
    judge_model: str = "claude-sonnet-5",
    max_workers: int = 16,
    generation_batch_size: int = 16,
    max_new_tokens: int = 200,
) -> dict:
    """
    Sample n_train SyPR rows, generate a fresh baseline response per row (no
    steering attached -- this is the same unsteered generation used for both
    label judging and probe training), judge each for praise via
    sycophantic_praise_judge.judge_praise_batch, and combine with SyPR's own
    ground-truth quality field: label = 1 iff praised==1 and is_poor_quality(row).

    Mirrors moral_sycophancy_judge.generate_moral_sycophancy_labels's return
    shape closely enough that a caller can treat both sources uniformly.
    """
    from utils.inference import build_chat_prompt_multiturn
    from sycophancy_steering import ActivationSteerer
    from probing.evaluations.baseline_probes.judge.sycophantic_praise_judge import judge_praise_batch
    from probing.data.sypr_data import build_chat_messages

    dataset = load_sypr_dataset()
    rows, sampled_indices = stratified_sample(dataset, n_train, seed=seed, return_indices=True)
    eligible = [(r, idx) for r, idx in zip(rows, sampled_indices) if is_label_eligible(r)]
    rows = [r for r, _ in eligible]
    sampled_indices = [idx for _, idx in eligible]

    steerer = ActivationSteerer(model, tokenizer, model_config)
    prompts = [build_chat_prompt_multiturn(tokenizer, build_chat_messages(r)) for r in rows]
    responses = []
    for i in range(0, len(prompts), generation_batch_size):
        responses.extend(
            steerer.generate_batch(prompts[i : i + generation_batch_size], max_new_tokens=max_new_tokens)
        )
    steerer.cleanup()
    for row, prompt, response in zip(rows, prompts, responses):
        row["prompt"] = prompt
        row["response"] = response

    verdicts = judge_praise_batch(rows, judge_model=judge_model, max_workers=max_workers)
    for row, verdict in zip(rows, verdicts):
        row["praised"] = verdict

    judged_rows = [r for r in rows if r["praised"] is not None]
    n_skipped = len(rows) - len(judged_rows)
    for row in judged_rows:
        row["label"] = 1 if (row["praised"] == 1 and is_poor_quality(row)) else 0

    n_pos = sum(r["label"] == 1 for r in judged_rows)
    n_neg = len(judged_rows) - n_pos
    n_poor = sum(is_poor_quality(r) for r in judged_rows)
    n_good = len(judged_rows) - n_poor
    praise_rate_on_poor = (
        sum(r["praised"] == 1 for r in judged_rows if is_poor_quality(r)) / n_poor if n_poor else 0.0
    )
    praise_rate_on_good = (
        sum(r["praised"] == 1 for r in judged_rows if not is_poor_quality(r)) / n_good if n_good else 0.0
    )

    records = [
        {
            "text": r["prompt"] + r["response"],
            "label": r["label"],
            "domain": r["domain"],
            "utterance_text": r["utterance_text"],
            "response": r["response"],
            "prompt": r["prompt"],
            "is_poor_quality": is_poor_quality(r),
            "praised": r["praised"],
        }
        for r in judged_rows
    ]

    return {
        "records": records,
        "n_judged": len(judged_rows),
        "n_skipped": n_skipped,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "praise_rate_on_poor": praise_rate_on_poor,
        "praise_rate_on_good": praise_rate_on_good,
        "sampled_indices": sampled_indices,
    }
