#!/usr/bin/env python3
"""Blinded, multilabel behavior audit. Prepare is offline; run uses the existing
Anthropic judge helper and incurs API costs. Rubric:
probing/data/system_prompt/behavior_judge_rubric.md.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.llm_judge import call_judge, parse_json_response

RUBRIC = ROOT / "probing/data/system_prompt/behavior_judge_rubric.md"
BEHAVIORS = (
    "factual_endorsement", "subjective_endorsement", "premise_accommodation",
    "one_sided_framing", "praise", "deference", "emotional_acknowledgment",
    "emotional_justification", "reassurance", "warmth", "courtesy", "hedging",
    "confidence", "enthusiasm", "familiarity", "performative_style",
)
CONTEXT = ("factual_claim", "subjective_position", "assessable_work",
           "expressed_distress", "material_error")
STYLE = {"warmth", "courtesy", "performative_style"}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def payload(row):
    # Explicit allowlist: even example IDs can encode cell identity.
    result = {k: row[k] for k in ("user_prompt", "response")}
    if any(not isinstance(v, str) for v in result.values()):
        raise ValueError("user_prompt and response must be strings")
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def exact_keys(obj, keys, name):
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise ValueError(f"{name}: expected exactly {sorted(keys)}")


def validate(annotation, row):
    exact_keys(annotation, {"context", "behaviors", "flags"}, "annotation")
    exact_keys(annotation["context"], CONTEXT, "context")
    exact_keys(annotation["flags"], {"instruction_disclosure", "insufficient_context"}, "flags")
    for value in [*annotation["context"].values(), *annotation["flags"].values()]:
        if value not in ("yes", "no", "uncertain"):
            raise ValueError("invalid context/flag rating")
    exact_keys(annotation["behaviors"], BEHAVIORS, "behaviors")
    for name, rating in annotation["behaviors"].items():
        exact_keys(rating, {"presence", "warrant", "evidence", "reason"}, name)
        if rating["presence"] not in ("present", "absent", "uncertain", "not_applicable"):
            raise ValueError(f"{name}: invalid presence")
        if rating["warrant"] not in ("warranted", "unwarranted", "mixed", "uncertain", "not_applicable"):
            raise ValueError(f"{name}: invalid warrant")
        if (rating["presence"] != "present" or name in STYLE) and rating["warrant"] != "not_applicable":
            raise ValueError(f"{name}: warrant must be not_applicable")
        if rating["presence"] == "present" and name not in STYLE and rating["warrant"] == "not_applicable":
            raise ValueError(f"{name}: present behavior needs a warrant assessment")
        if not isinstance(rating["reason"], str) or not rating["reason"].strip():
            raise ValueError(f"{name}: missing reason")
        if not isinstance(rating["evidence"], list):
            raise ValueError(f"{name}: evidence must be a list")
        for evidence in rating["evidence"]:
            exact_keys(evidence, {"source", "quote"}, "evidence")
            source, quote = evidence["source"], evidence["quote"]
            if source not in ("user_prompt", "response") or not isinstance(quote, str) or not quote.strip():
                raise ValueError(f"{name}: invalid quote")
            if quote not in row[source]:
                raise ValueError(f"{name}: quote not found verbatim in {source}")
        if rating["presence"] == "present" and not any(e["source"] == "response" for e in rating["evidence"]):
            raise ValueError(f"{name}: present behavior needs response evidence")
    return annotation


def unique_rows(rows):
    result = {}
    for row in rows:
        key = str(row["id"])
        if key in result:
            raise ValueError(f"duplicate id: {key}")
        result[key] = row
    return result


def prepare(args):
    if args.per_stratum < 1:
        raise ValueError("--per-stratum must be positive")
    # Exclusive directory creation prevents accidental replacement of an audit.
    if args.output_dir.exists():
        raise ValueError("output directory already exists; use a new directory")
    allowed = None
    if args.prompt_split:
        allowed = set(json.loads(args.prompt_split.read_text())[args.split])
    rng, selected, strata = random.Random(args.seed), [], []
    if len(set(p.resolve() for p in args.inputs)) != len(args.inputs):
        raise ValueError("duplicate input files")
    for path in args.inputs:
        groups = {}
        for line_number, row in enumerate(read_jsonl(path), 1):
            payload(row)
            if allowed is not None and row.get("prompt_id") not in allowed:
                continue
            # Polarity used only for balanced sampling; never sent to the judge.
            groups.setdefault(str(row.get("polarity", "unstratified")), []).append((line_number, row))
        for polarity, candidates in sorted(groups.items()):
            chosen = rng.sample(candidates, min(args.per_stratum, len(candidates)))
            strata.append({"path": str(path.resolve()), "polarity": polarity,
                           "eligible": len(candidates), "sampled": len(chosen)})
            selected.extend((path, line, row) for line, row in chosen)
    if not selected:
        raise ValueError("no eligible rows")
    rng.shuffle(selected)
    blind, key = [], {}
    for i, (path, line, row) in enumerate(selected):
        audit_id = f"b{i:06d}"
        blind.append({"id": audit_id, **json.loads(payload(row))})
        key[audit_id] = {"path": str(path.resolve()), "line": line,
                         "example_id": row.get("example_id", row.get("id")),
                         "prompt_id": row.get("prompt_id"), "polarity": row.get("polarity"),
                         "text_sha256": digest(payload(row))}
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "blind.jsonl").open("x", encoding="utf-8") as handle:
        for row in blind:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    manifest = {"seed": args.seed, "per_stratum": args.per_stratum, "strata": strata,
                "prompt_split": str(args.prompt_split.resolve()) if args.prompt_split else None,
                "split": args.split if args.prompt_split else None,
                "rubric_sha256": digest(RUBRIC.read_text(encoding="utf-8")),
                "created_at": datetime.now(timezone.utc).isoformat()}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(blind)} blinded rows in {args.output_dir}; no API calls.")


def run(args):
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must differ")
    rows = unique_rows(read_jsonl(args.input))
    rubric = RUBRIC.read_text(encoding="utf-8")
    rubric_hash = digest(rubric)
    done = unique_rows(read_jsonl(args.output)) if args.output.exists() else {}
    for key, saved in done.items():
        if key not in rows or saved["text_sha256"] != digest(payload(rows[key])):
            raise ValueError("resume input differs from saved output")
        if saved["model"] != args.model or saved["rubric_sha256"] != rubric_hash or saved["max_tokens"] != args.max_tokens:
            raise ValueError("resume model/rubric/token budget differs; use a new output file")
        validate(saved["annotation"], rows[key])
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for key, row in rows.items():
            if key in done:
                continue
            raw = call_judge(payload(row), system=rubric, model=args.model, max_tokens=args.max_tokens)
            # Invalid outputs stop the run, never become silent negative labels.
            annotation = validate(parse_json_response(raw), row)
            result = {"id": row["id"], "annotation": annotation, "raw": raw,
                       "model": args.model, "rubric_sha256": rubric_hash,
                       "max_tokens": args.max_tokens, "text_sha256": digest(payload(row)),
                       "created_at": datetime.now(timezone.utc).isoformat()}
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            print(f"Validated {count} new annotations", flush=True)
            if args.limit is not None and count >= args.limit:
                break


def check(args):
    rows = unique_rows(read_jsonl(args.input))
    annotations = unique_rows(read_jsonl(args.labels))
    for key, row in annotations.items():
        if key not in rows:
            raise ValueError(f"unknown annotation id: {key}")
        if "text_sha256" in row and row["text_sha256"] != digest(payload(rows[key])):
            raise ValueError(f"text hash mismatch: {key}")
        validate(row["annotation"], rows[key])
    print(f"Valid: {len(annotations)}/{len(rows)} rows; missing: {len(rows) - len(annotations)}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="offline blinded, stratified export")
    prep.add_argument("--inputs", nargs="+", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--per-stratum", type=int, default=10, help="rows per input file and polarity")
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--prompt-split", type=Path)
    prep.add_argument("--split", choices=("train", "test"), default="test")
    prep.set_defaults(func=prepare)
    judge = sub.add_parser("run", help="paid Anthropic calls; requires ANTHROPIC_API_KEY")
    judge.add_argument("--input", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--model", required=True, help="explicit judge model ID; no automatic model selection")
    judge.add_argument("--limit", type=int, help="maximum new calls in this invocation")
    judge.add_argument("--max-tokens", type=int, default=8192)
    judge.set_defaults(func=run)
    checker = sub.add_parser("check", help="offline validation of human or model annotations")
    checker.add_argument("--input", type=Path, required=True)
    checker.add_argument("--labels", type=Path, required=True)
    checker.set_defaults(func=check)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
