from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import torch

SYCOPHANCY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (SYCOPHANCY_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from probing.utils.baseline_probes_common import json_dump, parse_dataset_spec, read_jsonl, write_jsonl  # noqa: E402
from probing.evaluations.baseline_probes.judge.scoring import score_rows  # noqa: E402


def _metadata_for(path: Path) -> dict | None:
    candidates = [path.with_suffix(".metadata.json"), path.parent / "run_info.json"]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def _source_group(name: str, row: dict) -> str:
    source = row.get("row_id", row.get("id", row.get("question_id", row.get("question"))))
    return f"{name}:{source}"


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_baseline_rows(name: str, rows: list[dict], metadata: dict, allow_unverified: bool) -> None:
    expected = metadata.get("baseline_compatibility")
    if expected is None:
        if allow_unverified:
            return
        raise ValueError(f"baseline {name} lacks sample/prompt compatibility fingerprints")
    ids = [str(row.get("row_id", row.get("id", index))) for index, row in enumerate(rows)]
    payload = [row.get("messages", row.get("prompt", row.get("question"))) for row in rows]
    actual = {
        "n_rows": len(rows), "sample_ids_sha256": _digest(ids),
        "prompt_payload_sha256": _digest(payload),
    }
    if actual != expected:
        raise ValueError(f"baseline {name} rows/prompts differ from its generation metadata")
    row_models = {row.get("model") for row in rows if row.get("model") is not None}
    if row_models and row_models != {metadata.get("model")}:
        raise ValueError(f"baseline {name} contains row-level model metadata inconsistent with its sidecar")


def _generation_prompt(tokenizer, row: dict) -> tuple[str, list[dict] | None]:
    if row.get("messages"):
        messages = list(row["messages"])
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return rendered, messages
    prompt = row.get("prompt", row.get("question"))
    if prompt is None:
        raise ValueError("baseline row lacks messages, prompt, and question")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return rendered, None


def _generate(model, tokenizer, prompts: list[str], decoding: dict) -> list[str]:
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=int(decoding.get("max_length", 1024)), add_special_tokens=False,
    ).to(model.device)
    kwargs = {
        "max_new_tokens": int(decoding.get("max_new_tokens", 150)),
        "do_sample": bool(decoding.get("do_sample", False)),
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if kwargs["do_sample"]:
        kwargs["temperature"] = float(decoding["temperature"])
        kwargs["top_p"] = float(decoding["top_p"])
    with torch.no_grad():
        outputs = model.generate(**inputs, **kwargs)
    input_length = inputs["input_ids"].shape[1]
    return [tokenizer.decode(row[input_length:], skip_special_tokens=True).strip() for row in outputs]


def _generated_rows(tokenizer, baseline_rows: list[dict], responses: list[str]) -> list[dict]:
    generated = []
    for row, response in zip(baseline_rows, responses):
        output = dict(row)
        _, prompt_messages = _generation_prompt(tokenizer, row)
        output["response"] = response
        if prompt_messages is not None:
            output["messages"] = prompt_messages + [{"role": "assistant", "content": response}]
        generated.append(output)
    return generated


def run_steering_evaluation(args, alphas: list[float], output_dir: Path) -> None:
    from utils.model_registry import get_model_config
    from sycophancy_steering import ActivationSteerer
    from utils.model import cleanup, load_model_and_tokenizer

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    layer = metrics.get("probe_selected_residual_layer")
    if layer is None:
        raise ValueError("DIM metrics do not contain a probe-selected residual layer")
    vectors = torch.load(output_dir / "residual_dim_vectors.pt", map_location="cpu")
    vector = vectors[str(layer)]

    holdout_data = json.loads((output_dir / "steering_holdout.json").read_text(encoding="utf-8"))
    holdout_by_name = {
        item["name"].removesuffix("_holdout"): set(item["groups"])
        for item in holdout_data["datasets"]
    }
    targets = []
    for raw_spec in args.baseline_generation:
        spec = parse_dataset_spec(raw_spec)
        metadata = _metadata_for(spec.path)
        if metadata is None and not args.allow_unverified_baseline:
            raise ValueError(
                f"baseline {spec.path} lacks metadata; pass --allow-unverified-baseline to override"
            )
        metadata = metadata or {
            "model": args.model, "dataset_type": "generic",
            "decoding": {"do_sample": False, "max_new_tokens": 150}, "unverified": True,
        }
        if metadata.get("model") != args.model:
            raise ValueError(f"baseline {spec.name} model does not match {args.model}")
        if "decoding" not in metadata and not args.allow_unverified_baseline:
            raise ValueError(f"baseline {spec.name} lacks decoding metadata")
        rows = read_jsonl(spec.path)
        _validate_baseline_rows(spec.name, rows, metadata, args.allow_unverified_baseline)
        if spec.name in holdout_by_name:
            rows = [row for row in rows if _source_group(spec.name, row) in holdout_by_name[spec.name]]
            if not rows:
                raise ValueError(f"baseline {spec.name} contains none of the reserved steering groups")
        targets.append((spec, metadata, rows))

    model, tokenizer = load_model_and_tokenizer(args.model)
    config = get_model_config(args.model)
    steerer = ActivationSteerer(model, tokenizer, config)
    judge_model = "claude-sonnet-5"
    summaries = {}
    try:
        for spec, metadata, rows in targets:
            dataset_type = metadata.get("dataset_type", "generic")
            baseline_summary, baseline_judged = score_rows(rows, dataset_type, judge_model)
            target_dir = output_dir / "steering" / spec.name
            write_jsonl(target_dir / "baseline_judged.jsonl", baseline_judged)
            per_alpha = {}
            prompts = [_generation_prompt(tokenizer, row)[0] for row in rows]
            decoding = metadata.get("decoding", {"do_sample": False, "max_new_tokens": 150})
            for alpha in alphas:
                steerer.attach("residual", int(layer), vector, alpha)
                try:
                    responses = _generate(model, tokenizer, prompts, decoding)
                finally:
                    steerer.cleanup()
                generated = _generated_rows(tokenizer, rows, responses)
                judged_summary, judged = score_rows(generated, dataset_type, judge_model)
                write_jsonl(target_dir / f"alpha_{alpha:+g}.jsonl", judged)
                per_alpha[str(alpha)] = judged_summary
            summaries[spec.name] = {"baseline": baseline_summary, "alphas": per_alpha}
    finally:
        steerer.cleanup()
        cleanup(model, tokenizer)
    json_dump(output_dir / "steering" / "summary.json", summaries)
