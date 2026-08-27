#!/usr/bin/env python3
"""
Build the shared pool of user turns that every prompt pair is applied to: the
three sycophancy subsets of Perez et al. (2022) / Anthropic/model-written-evals.

Source rows are first-person bios ending in "I believe X. Do you agree?" plus an
"(A)/(B)" option block. The option block is stripped -- left in, the model answers
with a single letter and the first5/response activation positions have nothing to
measure.

Selection is round-robin over the subsets after a seeded per-subset shuffle, then
cut at --n. That makes --n 20 a prefix of --n 200: smoke runs use real data, and
raising --n appends rather than invalidating generations already produced against
the smaller file.

generate_response.py always reads the default path, so --out is for inspecting a
selection, not for feeding the pipeline; use its --limit-prompts for smoke runs.

Usage:
    python fetch_user_prompts.py --n 200
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402  (sets up REPO_ROOT on sys.path)

# name -> filename in prompt_probes/data/
SUBSETS = {
    "nlp_survey": "sycophancy_on_nlp_survey.jsonl",
    "political_typology": "sycophancy_on_political_typology_quiz.jsonl",
    "philpapers": "sycophancy_on_philpapers2020.jsonl",
}

# Anchored on the blank line + "(A)" that opens the option block. Strict on
# purpose: a miss is counted (n_strip_missed) rather than silently tolerated.
_CHOICES_RE = re.compile(r"\n\s*(?:Choices:\s*)?\n?\s*\(A\)", re.IGNORECASE)
_TRAILING_RE = re.compile(r"\s*(?:Choices:|Answer:)\s*$", re.IGNORECASE)

# Some bios cite the option labels directly ("I agree with the statement in (A)").
# Stripping the block leaves those turns pointing at nothing, so they are dropped.
_MARKER_RE = re.compile(r"\(A\)|\(B\)|Choices:", re.IGNORECASE)


def resolve_subsets(subsets: dict, data_dir: Path) -> dict[str, Path]:
    """Resolve each subset to its local file. A missing file is fatal."""
    out = {}
    for name, local_name in subsets.items():
        local = data_dir / local_name
        if not local.exists():
            raise SystemExit(f"missing subset file: {local}")
        print(f"  {name}: {local.name}")
        out[name] = local
    return out


def strip_choices(question: str) -> tuple[str, bool]:
    """Return (question without the multiple-choice block, whether it fired)."""
    m = _CHOICES_RE.search(question)
    if not m:
        return question.strip(), False
    stripped = question[: m.start()]
    stripped = _TRAILING_RE.sub("", stripped)
    return stripped.strip(), True


def build_user_prompts(
    raw: dict[str, list[dict]],
    n: int,
    seed: int,
    strip: bool,
) -> tuple[list[dict], dict]:
    """Round-robin sample n prompts across subsets; returns (rows, stats).

    prompt_id embeds each row's index in its source file, so ids stay stable
    across runs and across changes to --n.
    """
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = {}
    stats = {
        "n_strip_fired": 0,
        "n_strip_missed": 0,
        "n_marker_in_bio": 0,
        "per_subset": {},
    }

    for name, rows in raw.items():
        prepared = []
        for idx, row in enumerate(rows):
            question = row.get("question") or ""
            if strip:
                text, fired = strip_choices(question)
                stats["n_strip_fired" if fired else "n_strip_missed"] += 1
            else:
                text, fired = question.strip(), False
            if not text:
                continue
            if strip and _MARKER_RE.search(text):
                stats["n_marker_in_bio"] += 1
                continue
            prepared.append(
                {
                    "prompt_id": f"{name}__{idx:05d}",
                    "source": name,
                    "user_prompt": text,
                    "question_raw": question,
                    "answer_matching_behavior": row.get("answer_matching_behavior"),
                    "answer_not_matching_behavior": row.get("answer_not_matching_behavior"),
                }
            )
        rng.shuffle(prepared)
        pools[name] = prepared
        stats["per_subset"][name] = {"n_available": len(prepared)}

    # Round-robin over the pools, cut at n. Pool order is fixed by the seed and
    # nothing is resampled, so --n 20 is a prefix of --n 200.
    order = sorted(pools)
    selected: list[dict] = []
    cursor = {name: 0 for name in order}
    while len(selected) < n:
        progressed = False
        for name in order:
            if len(selected) >= n:
                break
            if cursor[name] < len(pools[name]):
                selected.append(pools[name][cursor[name]])
                cursor[name] += 1
                progressed = True
        if not progressed:
            break

    if len(selected) < n:
        raise SystemExit(
            f"only {len(selected)} prompts survived filtering, wanted {n}. Lower --n."
        )

    for row in selected:
        stats["per_subset"][row["source"]].setdefault("n_selected", 0)
        stats["per_subset"][row["source"]]["n_selected"] += 1
    return selected, stats


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=200, help="Number of user prompts to select.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=common.USER_PROMPTS_PATH)
    parser.add_argument("--data-dir", type=Path, default=common.DATA_DIR,
                        help="Where the sycophancy_on_*.jsonl subsets live.")
    parser.add_argument(
        "--keep-choices",
        action="store_true",
        help="Keep the (A)/(B) option block. Makes responses ~1 token, so the first5 and "
        "response activation positions become degenerate. Off by default.",
    )
    parser.add_argument("--show", type=int, default=3, help="Print this many selected prompts.")
    args = parser.parse_args()

    print(f"Resolving {len(SUBSETS)} Perez sycophancy subsets ...")
    paths = resolve_subsets(SUBSETS, args.data_dir)
    raw = {name: common.read_jsonl(path) for name, path in paths.items()}
    for name, rows in raw.items():
        print(f"  {name}: {len(rows)} rows")

    rows, stats = build_user_prompts(
        raw,
        n=args.n,
        seed=args.seed,
        strip=not args.keep_choices,
    )

    if not args.keep_choices:
        # build_user_prompts already drops these; anything reaching here is a bug.
        leaked = [r["prompt_id"] for r in rows if _MARKER_RE.search(r["user_prompt"])]
        if leaked:
            raise SystemExit(f"option markers survived in {len(leaked)} prompts, e.g. {leaked[:3]}")

    common.write_jsonl(args.out, rows)

    print(f"\nWrote {len(rows)} prompts to {args.out}")
    print(
        f"  strip fired: {stats['n_strip_fired']}, missed: {stats['n_strip_missed']}, "
        f"dropped (marker in bio): {stats['n_marker_in_bio']}"
    )
    print(f"  per subset: {json.dumps(stats['per_subset'])}")

    for row in rows[: args.show]:
        print(f"\n--- {row['prompt_id']} ({row['source']}) ---\n{row['user_prompt']}")


if __name__ == "__main__":
    main()
