"""Shared ELEPHANT (data/source/elephant/*.csv) generation loop for
generate_moral.py, generate_social.py and generate_aita_yta.py.

Merges the former openrouter-only moral_generate_openrouter.py /
social_generate_openrouter.py / generate_sae.py and the local path of
steering/generation/{moral,social}_generate.py. Output schema is the
one the ELEPHANT judges (run_moral_sycophancy_judge_aita.py,
run_social_sycophancy_judge_oeq.py, judge/scoring.py) read:
    {dataset, row_id, prompt_col, prompt, response, sample_idx, model, ...}
where "prompt" is the raw source text (never a chat-templated string), plus
finish_reason / reasoning / system_prompt / messages / backend.

Checkpointed: records are appended per chunk; a rerun of the same command
skips every (row_id, prompt_col) already in --output or in the cap-hit
sidecar, so a crash loses at most one chunk. Cap-hit (finish_reason ==
"length", i.e. INCOMPLETE) generations go to <output>.truncated.jsonl
(checkpoint.jsonl -> checkpoint.truncated.jsonl, the old sidecar name)
instead of the judged checkpoint. Rows whose OpenRouter request failed
outright are not written at all, so a rerun retries them.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from evaluations.generation.common import (
    make_generator, resolve_system_prompt, with_system_prompt, write_metadata,
)

ELEPHANT_FILES = {
    "aita_nta_flip": "AITA-NTA-FLIP.csv",
    "aita_nta_og": "AITA-NTA-OG.csv",
    "aita_yta": "AITA-YTA.csv",
    "oeq": "OEQ.csv",
    "ss": "SS.csv",
}


def _plain(value):
    return value.item() if hasattr(value, "item") else value


def load_elephant_rows(dataset: str, seed: int, limit: int | None, keep_pairs: bool) -> tuple[list[dict], int]:
    """(prompt records in generation order, number of source rows selected).

    keep_pairs=True (moral): shuffle whole row_ids so both prompt columns of
    an AITA-NTA-FLIP row (original_post, flipped_story) stay adjacent and a
    --limit ROW cap keeps complete flip pairs -- the flip judge pairs them by
    row_id. keep_pairs=False (social): shuffle individual records."""
    from utils.datasets import iter_prompts

    all_rows = [{**r, "row_id": _plain(r["row_id"])} for r in iter_prompts(ELEPHANT_FILES[dataset])]
    if not keep_pairs:
        random.Random(seed).shuffle(all_rows)
        if limit is not None:
            all_rows = all_rows[:limit]
        return all_rows, len(all_rows)
    row_ids = sorted({r["row_id"] for r in all_rows}, key=str)
    random.Random(seed).shuffle(row_ids)
    if limit is not None:
        row_ids = row_ids[:limit]
    order = {rid: i for i, rid in enumerate(row_ids)}
    rows = sorted((r for r in all_rows if r["row_id"] in order), key=lambda r: (order[r["row_id"]], r["prompt_col"]))
    return rows, len(row_ids)


def _key(record: dict) -> tuple[str, str]:
    return (str(record["row_id"]), record["prompt_col"])


def _done_keys(*paths: Path) -> set[tuple[str, str]]:
    done = set()
    for path in paths:
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                done.update(_key(json.loads(line)) for line in handle if line.strip())
    return done


def add_chunk_arg(parser) -> None:
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Records generated (and fsync'd to --output) per checkpoint step")


def run_elephant(args, dataset: str, keep_pairs: bool, dataset_type: str) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    truncated_path = output.with_suffix(".truncated.jsonl")
    system_prompt = resolve_system_prompt(args)

    print(f"Loading {ELEPHANT_FILES[dataset]}...")
    rows, n_source_rows = load_elephant_rows(dataset, args.seed, args.limit, keep_pairs)
    done = _done_keys(output, truncated_path)
    pending = [r for r in rows if _key(r) not in done]
    print(f"{n_source_rows} rows -> {len(rows)} generations (seed={args.seed}); "
          f"{len(rows) - len(pending)} already on disk, {len(pending)} pending.")

    generate = make_generator(args)
    n_failed = 0
    for start in range(0, len(pending), args.chunk_size):
        chunk = pending[start : start + args.chunk_size]
        conversations = [with_system_prompt([{"role": "user", "content": r["text"]}], system_prompt) for r in chunk]
        generations = generate(conversations)
        kept, truncated = [], []
        for row, chat, gen in zip(chunk, conversations, generations):
            if gen["finish_reason"] == "request_failed":
                n_failed += 1
                continue
            record = {
                "dataset": row["dataset"], "row_id": row["row_id"], "prompt_col": row["prompt_col"],
                "prompt": row["text"], "response": gen["content"], "sample_idx": 0, "model": args.model,
                "finish_reason": gen["finish_reason"], "reasoning": gen["reasoning"],
                "system_prompt": system_prompt, "backend": args.backend,
                "messages": chat + [{"role": "assistant", "content": gen["content"]}],
            }
            if gen["finish_reason"] == "length":
                truncated.append({**record, "max_new_tokens": args.max_new_tokens})
            else:
                kept.append(record)
        for path, records in ((output, kept), (truncated_path, truncated)):
            if records:
                with path.open("a", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{min(start + len(chunk), len(pending))}/{len(pending)}] {len(kept)} written, "
              f"{len(truncated)} cap-hit set aside this chunk", flush=True)

    n_generated = len(_done_keys(output))
    n_truncated = len(_done_keys(truncated_path))
    if n_failed:
        print(f"WARNING: {n_failed} requests failed and were not written -- rerun the same command to retry them.")
    write_metadata(
        output, args, dataset_type, {"path": f"data/source/elephant/{ELEPHANT_FILES[dataset]}"},
        {"dataset": dataset, "n_rows": n_source_rows, "n_total_generations": len(rows),
         "n_generated": n_generated, "n_cap_hit_set_aside": n_truncated, "n_request_failed": n_failed},
    )
    print(f"Done. {n_generated}/{len(rows)} generations in {output} ({n_truncated} cap-hit in {truncated_path.name}).")
