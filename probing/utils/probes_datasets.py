from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from probing.utils.probes_cache import ActivationCache
from probing.utils.probes_common import read_jsonl


SUPPORTED_DATASETS = {
    "generic", "social", "moral", "sypr", "are_you_sure", "are_you_sure_mcq", "mcq", "syconbench"
}


def _messages(prompt: str, response: str) -> list[dict]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def _pair_outcome(original: str | None, flipped: str | None) -> str:
    if original == "NTA" and flipped == "NTA":
        return "both_nta"
    if original == "NTA" and flipped == "YTA":
        return "original_nta_flipped_yta"
    if original == "YTA" and flipped == "NTA":
        return "original_yta_flipped_nta"
    if original == "YTA" and flipped == "YTA":
        return "both_yta"
    return "unclear"


def normalize_records(path: Path, dataset_type: str, label_field: str = "label") -> list[dict]:
    if dataset_type not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset type {dataset_type!r}")
    rows = read_jsonl(path)
    normalized: list[dict] = []

    if dataset_type == "moral":
        for row in rows:
            group_id = str(row["row_id"])
            og = row.get("original_post_verdict")
            flip = row.get("flipped_story_verdict")
            outcome = _pair_outcome(og, flip)
            if outcome not in {"both_nta", "original_nta_flipped_yta", "both_yta"}:
                continue
            label = int(outcome == "both_nta")
            for side in ("original_post", "flipped_story"):
                prompt = row[f"{side}_prompt"]
                response = row[f"{side}_response"]
                normalized.append({
                    "id": f"{group_id}:{side}", "group_id": group_id,
                    "dataset_type": dataset_type, "label": label, "side": side,
                    "pair_outcome": outcome, "messages": _messages(prompt, response),
                })
        return normalized

    for i, row in enumerate(rows):
        label = row.get(label_field)
        if label not in (0, 1, False, True):
            continue
        source_id = row.get("row_id", row.get("id", row.get("question_id", i)))
        group_id = str(source_id)
        if "messages" in row:
            messages = row["messages"]
        elif "prompt" in row and "response" in row:
            messages = _messages(row["prompt"], row["response"])
        elif "utterance_text" in row and "response" in row:
            messages = _messages(row["utterance_text"], row["response"])
        elif "question" in row and "final_response" in row:
            messages = _messages(row["question"], row["final_response"])
        else:
            messages = None
        normalized.append({
            "id": str(row.get("id", f"{group_id}:{i}")),
            "group_id": group_id,
            "dataset_type": dataset_type,
            "label": int(label),
            "messages": messages,
            "text": row.get("text"),
            "first_flip_turn": row.get("first_flip_turn"),
            "domain": row.get("domain"),
            "source": row,
        })
    return normalized


def _example_axis(component: str) -> int:
    return 2 if component == "mha" else 1


def _take(array: np.ndarray, component: str, indices: np.ndarray) -> np.ndarray:
    return np.take(array, indices, axis=_example_axis(component))


def _average(array: np.ndarray, component: str, indices: list[int]) -> np.ndarray:
    return np.take(array, indices, axis=_example_axis(component)).mean(
        axis=_example_axis(component), keepdims=True
    )


def prepare_cache(cache: ActivationCache, dataset_name: str, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
    """Apply moral pair averaging/matching and prefix source groups."""
    if cache.metadata.get("dataset_type") != "moral":
        records = [dict(r) for r in cache.records]
        groups = np.asarray([f"{dataset_name}:{r['group_id']}" for r in records], dtype=object)
        labels = np.asarray([r["label"] for r in records], dtype=np.int64)
        return {k: np.asarray(v) for k, v in cache.arrays.items()}, labels, groups, records

    by_group: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(cache.records):
        by_group[str(record["group_id"])].append(idx)

    pair_records = []
    pair_arrays: dict[str, list[np.ndarray]] = {k: [] for k in cache.arrays}
    for group_id, indices in sorted(by_group.items()):
        if len(indices) != 2:
            raise ValueError(f"moral pair {group_id} has {len(indices)} sides; expected exactly 2")
        outcome = cache.records[indices[0]]["pair_outcome"]
        pair_records.append({
            "id": group_id, "group_id": group_id, "pair_outcome": outcome,
            "label": int(outcome == "both_nta"), "dataset_type": "moral",
        })
        for component, array in cache.arrays.items():
            pair_arrays[component].append(_average(np.asarray(array), component, indices))

    arrays = {
        component: np.concatenate(chunks, axis=_example_axis(component))
        for component, chunks in pair_arrays.items()
    }
    positives = [i for i, r in enumerate(pair_records) if r["pair_outcome"] == "both_nta"]
    primary = [i for i, r in enumerate(pair_records) if r["pair_outcome"] == "original_nta_flipped_yta"]
    secondary = [i for i, r in enumerate(pair_records) if r["pair_outcome"] == "both_yta"]
    rng = np.random.default_rng(seed)
    rng.shuffle(primary)
    rng.shuffle(secondary)
    negatives = primary + secondary
    n = min(len(positives), len(negatives))
    if n == 0:
        raise ValueError("moral cache has no usable balanced positive/negative pairs")
    rng.shuffle(positives)
    selected = np.asarray(sorted(positives[:n] + negatives[:n]), dtype=int)
    selected_records = [pair_records[i] for i in selected]
    selected_arrays = {k: _take(v, k, selected) for k, v in arrays.items()}
    labels = np.asarray([r["label"] for r in selected_records], dtype=np.int64)
    groups = np.asarray([f"{dataset_name}:{r['group_id']}" for r in selected_records], dtype=object)
    return selected_arrays, labels, groups, selected_records
