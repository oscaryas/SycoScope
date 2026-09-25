#!/usr/bin/env python3
"""OOD scoring of frozen prompt probes on judged five-turn SYCON conversations.

Consumes pipeline_scripts/judges/judge_syconbench.py JSONL, not upstream CSVs
with incomplete generation-history provenance. --prepare-only is CPU/offline.
Prepared records/manifest are written next to --cache
(<stem>.records.jsonl, <stem>.preparation.json); the cache .meta.json records
the records fingerprint and selection split, re-checked when --judged is given.

Usage:
    python -m scorer.eval_syconbench --judged judged.jsonl --cache results/probes/scores/sycon.npz --prepare-only
    python -m scorer.eval_syconbench --judged judged.jsonl --cache results/probes/scores/sycon.npz --extract-only
    python -m scorer.eval_syconbench --probe weights.pkl --cache results/probes/scores/sycon.npz
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from utils import common

SETTINGS = ("debate", "ethical", "false_presupposition")
LABEL_FIELDS = tuple(f"{setting}_{target}" for setting in SETTINGS
                     for target in ("failure", "first_failure"))
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def canonical_model(model):
    return str(model).lower().replace("meta-llama/meta-llama-", "meta-llama/llama-")


def binary(value):
    return type(value) in (int, bool) and value in (0, 1)


def normalize_conversations(rows, model=MODEL):
    """Per-turn labels with question-group IDs and no future-response leakage.

    The upstream judge convention is 1=maintained/rejected, 0=failed.
    first_failure is defined only at turns 2..5 where ALL previous judgments
    are 1; this is the first-failure risk set, not 'ever failed' on every turn.
    """
    records, exclusions, seen = [], [], set()
    for row in rows:
        if canonical_model(row.get("model")) != canonical_model(model):
            raise ValueError(f"generation model mismatch: {row.get('model')!r} vs {model!r}")
        setting = row.get("setting")
        if setting not in SETTINGS:
            raise ValueError(f"unsupported setting: {setting!r}")
        source_id = str(row["id"])
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{source_id}: missing question for grouped splitting")
        # Same question remains grouped across seeds, variants, or repeated IDs.
        group = f"SYCON:{setting}:" + hashlib.sha256(question.strip().encode()).hexdigest()[:20]
        example_base = f"SYCON:{setting}:{source_id}"
        if example_base in seen:
            raise ValueError(f"duplicate conversation id: {example_base}")
        seen.add(example_base)
        messages = row.get("messages")
        judgments = row.get("turn_judgments")
        if not isinstance(messages, list) or not isinstance(judgments, list) or len(judgments) != 5:
            raise ValueError(f"{source_id}: requires full messages and five turn_judgments")
        expected = ["user", "assistant"] * 5
        roles = [m.get("role") for m in messages]
        if roles not in (expected, ["system"] + expected):
            raise ValueError(f"{source_id}: expected five complete alternating turns")
        if any(not isinstance(m.get("content"), str) for m in messages):
            raise ValueError(f"{source_id}: messages must contain string content")
        answers = [m["content"] for m in messages if m["role"] == "assistant"]
        if "responses" in row and row["responses"] != answers:
            raise ValueError(f"{source_id}: responses and messages disagree")
        finishes = row.get("turn_finish_reasons")
        if not isinstance(finishes, list) or len(finishes) != 5:
            raise ValueError(f"{source_id}: requires five turn_finish_reasons")
        if any(v is not None and not binary(v) for v in judgments):
            raise ValueError(f"{source_id}: judgments must be 0, 1, or null")
        assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        for turn, message_index in enumerate(assistant_indices, 1):
            response = messages[message_index]["content"]
            # A failed/truncated earlier answer contaminates subsequent histories.
            if finishes[turn - 1] != "stop" or not response.strip():
                exclusions.append({"id": source_id, "from_turn": turn,
                                   "reason": "incomplete_generation_history"})
                break
            judgment = judgments[turn - 1]
            if judgment is None:
                exclusions.append({"id": source_id, "turn": turn, "reason": "unresolved_judgment"})
                continue
            failure = 1 - int(judgment)
            labels = {field: None for field in LABEL_FIELDS}
            labels[f"{setting}_failure"] = failure
            if turn >= 2 and all(v == 1 for v in judgments[:turn - 1]):
                labels[f"{setting}_first_failure"] = failure
            records.append({
                "example_id": f"{example_base}:turn{turn}", "group_id": group,
                "dataset": f"SYCON:{setting}", "turn": turn,
                "chat_messages": [dict(m) for m in messages[:message_index]],
                "response": response, "prompt_condition": row.get("prompt_condition"),
                "judge_model": row.get("judge_model"), **labels,
            })
    return records, exclusions


def inventory(records):
    return {
        "n_records": len(records),
        "n_groups": len({r["group_id"] for r in records}),
        "by_dataset": dict(Counter(r["dataset"] for r in records)),
        "labels": {field: {
            "positive": sum(r[field] == 1 for r in records),
            "negative": sum(r[field] == 0 for r in records),
            "unlabeled": sum(r[field] is None for r in records),
        } for field in LABEL_FIELDS},
    }


def main():
    # Lazy import keeps normalization usable without model/scoring dependencies.
    from scorer import eval_common
    from scorer.score_probes import load_probe_set, probe_model
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    eval_common.add_common_args(parser)
    parser.set_defaults(model=MODEL, layers=[0, 4, 8, 12, 16, 20, 24, 28], batch_size=1,
                        max_length=8192)
    parser.add_argument("--judged", type=Path, default=None, help="judge_syconbench.py JSONL (needed to prepare/extract).")
    parser.add_argument("--model-path", help="Optional local copy of the same HF checkpoint")
    parser.add_argument("--prepare-only", action="store_true", help="validate/export without loading weights")
    args = parser.parse_args()
    if not 0 < args.selection_frac < 1:
        raise ValueError("selection fraction must be between 0 and 1")
    # Probes must come from the same model the conversations were generated with.
    for name, payload in load_probe_set(args.probe or []).items():
        trained_on = probe_model(payload)
        if trained_on and canonical_model(trained_on) != canonical_model(args.model):
            raise ValueError(f"probe {name} was trained on {trained_on}, not the extraction model {args.model}")
    paths = eval_common.cache_paths(args.cache, args.positions)
    selection_spec = {"frac": args.selection_frac, "seed": args.seed}

    if args.judged is not None:
        rows = common.read_jsonl(args.judged)
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("limit must be positive")
            # Limit whole questions, never select a turn because of its failure label.
            keys = sorted({(r["setting"], r["question"].strip()) for r in rows})
            chosen = set(random.Random(args.seed).sample(keys, min(args.limit, len(keys))))
            rows = [r for r in rows if (r["setting"], r["question"].strip()) in chosen]
        records, exclusions = normalize_conversations(rows, args.model)
        if not records:
            raise ValueError("no usable labeled turns")
        fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
        stem = args.cache.with_suffix("")
        stem.parent.mkdir(parents=True, exist_ok=True)
        prep_path = stem.with_name(stem.name + ".preparation.json")
        if prep_path.exists():
            previous = json.loads(prep_path.read_text())
            if previous["records_sha256"] != fingerprint:
                raise ValueError("prepared data differ; use a separate --cache rather than overwrite")
        manifest = {**inventory(records), "records_sha256": fingerprint,
                    "judged_path": str(args.judged.resolve()), "exclusions": exclusions,
                    "model": args.model, "label_fields": LABEL_FIELDS}
        record_path = stem.with_name(stem.name + ".records.jsonl")
        if record_path.exists():
            saved_fingerprint = hashlib.sha256(json.dumps(common.read_jsonl(record_path), sort_keys=True).encode()).hexdigest()
            if saved_fingerprint != fingerprint:
                raise ValueError("existing prepared records differ")
        else:
            common.write_jsonl(record_path, records)
        if not prep_path.exists():
            prep_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(inventory(records), indent=2))
        if args.prepare_only:
            print("Prepared only: no GPU or API calls.")
            return
        if args.overwrite:
            raise ValueError("use a separate cache for re-extraction; overwrite disabled for this evaluator")
        if not all(p.exists() for p in paths.values()):
            fields = (*LABEL_FIELDS, "group_id", "turn", "prompt_condition")
            eval_common.build_cache(paths, records, LABEL_FIELDS, fields, args, meta_extra={
                "records_sha256": fingerprint, "evaluation_selection": selection_spec,
            })
    elif args.prepare_only or not all(p.exists() for p in paths.values()):
        raise SystemExit("--judged is required to prepare or extract")
    else:
        fingerprint = None

    for path in paths.values():
        meta = eval_common.read_meta(path)
        if fingerprint is not None and meta.get("records_sha256") != fingerprint:
            raise ValueError(f"{path}: cached data provenance mismatch")
        if meta.get("model") and canonical_model(meta["model"]) != canonical_model(args.model):
            raise ValueError(f"{path}: cached model provenance mismatch")
        if meta.get("evaluation_selection", selection_spec) != selection_spec:
            raise ValueError("cached selection split differs")
    if args.extract_only:
        return
    eval_common.score_and_report(
        args, paths, "eval_syconbench", LABEL_FIELDS, selection_spec=selection_spec,
        extra={"note": "Separate setting/turn-failure targets; question-grouped split. "
                       "Response pooling covers current answer only."},
    )


if __name__ == "__main__":
    main()
