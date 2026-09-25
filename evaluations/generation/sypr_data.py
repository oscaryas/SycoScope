"""
Loading/sampling/prompt-building for the SyPR benchmark
(huggingface.co/datasets/vennemeyerd/sycophantic-praise, 13,200 rows).

Unlike every other dataset this repo's sycophancy pipeline uses
(AITA-NTA-FLIP/OG/YTA, OEQ, SS -- all ELEPHANT CSVs, extracted to
probing/data/source/elephant/, whose generated responses live in
per-application results/*.jsonl files already containing a model's generated
`response`), SyPR ships no model responses: its README describes them as
"dynamically generated at evaluation time". So the pipeline here is: sample
rows -> build a chat prompt from each row's persona-calibration history + the
utterance being evaluated -> generate a response from the target model ->
judge that response for praise -> combine with SyPR's own ground-truth
quality label to get the final sycophancy label. Everything up through
"build a prompt" lives here, as it depends only on the HF dataset itself.
Generation/judging/labeling (which needs a loaded model, an activation
steerer, and a judge client) is application-owned -- see
tool_calling/tasks/sycophancy/sypr_generation.py for the
generate_and_label_sypr step that consumes these helpers.

Row schema (relevant fields): domain ("gsm8k"/"mmlu_chemistry"/
"mmlu_economics"/"long_form_moral_reasoning"/"pseudo_profundity"),
domain_family ("reasoning"/"social"), utterance_quality (reasoning:
"correct"/"incorrect"; social: "low"/"medium"/"high"),
ground_truth_correctness (bool, reasoning only), prompt_condition
("asked_for_eval"/"unprompted"), conversation_history_json (JSON list of
{"role", "content"} turns -- a persona-calibration dialogue, often empty for
social domains), utterance_json (JSON dict with a "text" field -- the final
utterance to evaluate).
"""

import json

HF_DATASET_ID = "vennemeyerd/sycophantic-praise"

REASONING_DOMAINS = ("gsm8k", "mmlu_chemistry", "mmlu_economics")
SOCIAL_DOMAINS = ("long_form_moral_reasoning", "pseudo_profundity")
ALL_DOMAINS = REASONING_DOMAINS + SOCIAL_DOMAINS


def load_sypr_dataset():
    """Loads the full 13,200-row `train` split. Requires the `datasets` package."""
    from datasets import load_dataset

    return load_dataset(HF_DATASET_ID, split="train")


def is_poor_quality(row: dict) -> bool:
    """
    True iff the utterance being evaluated is actually low-quality, per SyPR's own
    ground truth -- reasoning domains use ground_truth_correctness (poor =
    incorrect), social domains use utterance_quality (poor = "low"; "medium" is
    ambiguous and handled by is_label_eligible below, not here).
    """
    if row["domain_family"] == "reasoning":
        return not row["ground_truth_correctness"]
    return row["utterance_quality"] == "low"


def is_label_eligible(row: dict) -> bool:
    """
    False for social-domain "medium"-quality rows -- ambiguous under the
    praise-AND-poor-quality sycophancy definition (confirmed with the user), so
    they're dropped from the labeled training set entirely rather than forced into
    either class. Reasoning rows (binary correct/incorrect) are always eligible.
    """
    if row["domain_family"] == "reasoning":
        return True
    return row["utterance_quality"] != "medium"


def all_eligible_indices(dataset) -> list:
    """
    Every row index across ALL_DOMAINS that is_label_eligible -- the full
    label-eligible pool (10,800 of the raw 13,200 rows), unsampled. Unlike
    stratified_sample, which draws an approximate n_total via per-bucket
    integer-division truncation, this returns every eligible row exactly
    once -- for callers that want the whole pool (e.g. a full-dataset
    generation run) rather than a balanced subsample. Order matches the
    dataset's own order; callers wanting a fixed random order should shuffle
    with their own seed.
    """
    return [i for i, row in enumerate(dataset) if row["domain"] in ALL_DOMAINS and is_label_eligible(row)]


def _build_quality_buckets(dataset) -> dict:
    """
    Single pass bucketing row indices by (domain, quality_bucket) -- avoids
    dataset.filter()'s repeated full-dataset scans (one per domain x bucket).
    Quality bucket is the raw "correct"/"incorrect" for reasoning domains, or
    normalized "low"/"high" for social domains ("medium" is excluded here
    entirely -- ambiguous under the praise-AND-poor-quality sycophancy
    definition, see is_label_eligible). Shared by stratified_sample and
    sample_poor_quality_heldout so bucket-membership rules can't drift between
    the two -- a row poor_quality_heldout excludes for training must be the
    same set stratified_sample could have drawn from.
    """
    buckets = {}
    for i, row in enumerate(dataset):
        domain = row["domain"]
        if domain not in ALL_DOMAINS:
            continue
        if row["domain_family"] == "social" and row["utterance_quality"] == "medium":
            continue
        bucket_key = (domain, row["utterance_quality"] if row["domain_family"] == "reasoning" else ("low" if row["utterance_quality"] == "low" else "high"))
        buckets.setdefault(bucket_key, []).append(i)
    return buckets


def _row_from_index(dataset, i: int) -> dict:
    """Plain dict (not an HF Dataset row) with conversation_history_json/
    utterance_json already parsed into "conversation_history" (list) /
    "utterance_text" (str)."""
    row = dict(dataset[i])
    row["conversation_history"] = json.loads(row["conversation_history_json"])
    row["utterance_text"] = json.loads(row["utterance_json"])["text"]
    return row


def stratified_sample(dataset, n_total: int, seed: int = 0, return_indices: bool = False):
    """
    Samples ~n_total rows evenly across ALL_DOMAINS, and within each domain evenly
    across its label-eligible quality buckets (reasoning: correct/incorrect; social:
    low/high, "medium" excluded at the source since it would just be discarded
    downstream anyway) -- without this, random sampling from the raw 13,200 rows
    would be dominated by whichever quality level happens to be more common and
    risk starving the minority class before training even begins.

    Returns a list of plain dicts (see _row_from_index). If return_indices=True,
    also returns the raw HF-dataset indices actually sampled -- needed by callers
    that must build a held-out set disjoint from this training sample afterward
    (sample_poor_quality_heldout). Default False preserves every existing
    caller's return shape unchanged.
    """
    import random

    rng = random.Random(seed)
    n_domains = len(ALL_DOMAINS)
    per_domain = max(1, n_total // n_domains)

    buckets = _build_quality_buckets(dataset)

    sampled_indices = []
    for domain in ALL_DOMAINS:
        quality_keys = [k for k in buckets if k[0] == domain]
        per_bucket = max(1, per_domain // max(1, len(quality_keys)))
        for key in quality_keys:
            idx_pool = buckets[key]
            rng.shuffle(idx_pool)
            sampled_indices.extend(idx_pool[:per_bucket])

    rng.shuffle(sampled_indices)
    rows = [_row_from_index(dataset, i) for i in sampled_indices]
    if return_indices:
        return rows, sampled_indices
    return rows


def sample_poor_quality_heldout(dataset, exclude_indices, n_heldout: int, seed: int = 0) -> list:
    """
    Poor-quality-only rows (is_poor_quality == True) not present in
    exclude_indices (pass stratified_sample's sampled_indices from the
    training call) -- mirrors the SyPR notebook's in-domain held-out logic:
    disjoint from training by construction, no domain stratification needed
    (this check only needs "known poor-quality, unseen during training", not
    balanced representation across domains/quality tiers).

    Raises ValueError if fewer than n_heldout eligible rows remain, rather
    than silently truncating -- a smaller-than-requested held-out set would
    change what "n=100" means in a run's SUMMARY.md without saying so.
    """
    import random

    exclude = set(exclude_indices)
    buckets = _build_quality_buckets(dataset)

    candidate_indices = []
    for (domain, quality), idx_pool in buckets.items():
        if quality not in ("incorrect", "low"):
            continue
        candidate_indices.extend(i for i in idx_pool if i not in exclude)

    rng = random.Random(seed)
    rng.shuffle(candidate_indices)
    if len(candidate_indices) < n_heldout:
        raise ValueError(
            f"Only {len(candidate_indices)} disjoint poor-quality rows available "
            f"after excluding {len(exclude)} training rows, need {n_heldout}."
        )
    return [_row_from_index(dataset, i) for i in candidate_indices[:n_heldout]]


def build_chat_messages(row: dict) -> list:
    """
    Persona-calibration history (often empty for social domains) + the final
    utterance to evaluate. Reasoning-domain history has consecutive same-role
    "assistant" turns (grading feedback immediately followed by the next
    practice question, with no intervening "user" turn) -- coalesced here by
    joining content with a newline, since chat templates that enforce strict
    user/assistant alternation (e.g. Llama-3's) raise a TemplateError otherwise.
    """
    raw_messages = list(row["conversation_history"]) + [{"role": "user", "content": row["utterance_text"]}]
    messages = []
    for m in raw_messages:
        if messages and messages[-1]["role"] == m["role"]:
            messages[-1] = {"role": m["role"], "content": messages[-1]["content"] + "\n" + m["content"]}
        else:
            messages.append(dict(m))
    # Reasoning-domain history opens with the "assistant" persona posing practice
    # questions before the human ever speaks -- chat templates that require the
    # conversation to start with "user" (common alternation check) reject that.
    # A minimal synthetic opener preserves the real content unchanged.
    if messages and messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": "Let's begin."})
    return messages
