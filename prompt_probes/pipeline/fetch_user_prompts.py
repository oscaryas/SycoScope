#!/usr/bin/env python3
"""
Fetch and prepare the user turns the prompt pairs get applied to: the three
sycophancy subsets of Perez et al. (2022) / Anthropic/model-written-evals.

Each row's `question` has the form "Hello, my name is <name>. I am a <bio>...
I believe <position>. Do you agree...?" followed by "Choices:\\n (A) ...\\n
(B) ...". We strip the choices, because keeping them forces one-token answers
and two of the three activation positions ("first 5 response tokens", "entire
response") would then be measuring nothing. The self-description bio is what
makes these turns usable for the person-referent taxonomy cells as well as the
position-referent ones.

Sampling is round-robin over the three subsets after a seeded per-subset
shuffle, then truncated to --n. That makes --n 20 literally the first 20 rows
of --n 200: a smoke test runs on a genuine subset of the real data, and raising
--n later appends prompts instead of invalidating generations already produced
against the smaller file.

Prompts whose rendered length under the *longest* of the 28 system prompts
exceeds --max-prompt-tokens are dropped here, so no later stage ever has to
truncate (a truncated prompt would make the last_prompt position meaningless).

Usage:
    python fetch_user_prompts.py --n 200
    python fetch_user_prompts.py --n 20 --out /tmp/smoke_prompts.jsonl   # smoke
"""
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402  (sets up REPO_ROOT on sys.path)
from utils.inference import build_chat_prompt  # noqa: E402

HF_REPO = "Anthropic/model-written-evals"
# name -> (filename in prompt_probes/data/, path within the HF dataset repo)
SUBSETS = {
    "nlp_survey": ("sycophancy_on_nlp_survey.jsonl", "sycophancy/sycophancy_on_nlp_survey.jsonl"),
    "political_typology": (
        "sycophancy_on_political_typology_quiz.jsonl",
        "sycophancy/sycophancy_on_political_typology_quiz.jsonl",
    ),
    "philpapers": ("sycophancy_on_philpapers2020.jsonl", "sycophancy/sycophancy_on_philpapers2020.jsonl"),
}

# Anchored on the blank-line + "(A)" that opens the option block. Deliberately
# strict: if it doesn't match we want to know, not to silently keep the options.
_CHOICES_RE = re.compile(r"\n\s*(?:Choices:\s*)?\n?\s*\(A\)", re.IGNORECASE)
_TRAILING_RE = re.compile(r"\s*(?:Choices:|Answer:)\s*$", re.IGNORECASE)

# A handful of source rows reference the option labels inside the user's own
# bio ("I agree with the statement in (A)"). Once the option block is stripped
# those turns are incoherent -- they point at something no longer present -- so
# they get dropped as candidates rather than silently kept.
_MARKER_RE = re.compile(r"\(A\)|\(B\)|Choices:", re.IGNORECASE)


def resolve_subsets(subsets: dict, data_dir: Path) -> dict[str, Path]:
    """Prefer the copies in prompt_probes/data/; download only what is missing.

    Local files take priority because the hub is not trustworthy for this
    dataset: hf_hub_download returns byte-identical content for
    sycophancy_on_nlp_survey.jsonl and sycophancy_on_philpapers2020.jsonl
    (both sha256 582860b4..., 9984 rows, all NLP-themed). Taking the hub at its
    word silently yields two copies of the NLP survey and no philosophy data.
    """
    out = {}
    for name, (local_name, hub_path) in subsets.items():
        local = data_dir / local_name
        if local.exists():
            print(f"  {name}: local {local.name}")
        else:
            from huggingface_hub import hf_hub_download

            src = hf_hub_download(repo_id=HF_REPO, filename=hub_path, repo_type="dataset")
            local.write_bytes(Path(src).read_bytes())
            print(f"  {name}: downloaded {hub_path} -> {local.name}  (VERIFY: the hub has served "
                  "duplicate content for these filenames before)")
        out[name] = local
    return out


def assert_distinct(paths: dict[str, Path]) -> dict[str, str]:
    """Fail loudly if two subsets are the same file.

    This exact failure already happened once and was silent: the sampled
    prompts looked like balanced thirds while actually being two-thirds one
    source, and a duplicate user turn appeared under two prompt_ids -- which
    defeats grouped_folds, since the same turn then lands in both train and
    test.
    """
    hashes = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()}
    by_hash: dict[str, list[str]] = {}
    for name, h in hashes.items():
        by_hash.setdefault(h, []).append(name)
    dupes = [names for names in by_hash.values() if len(names) > 1]
    if dupes:
        raise SystemExit(
            f"identical subset files: {dupes}. These must be three distinct datasets. "
            f"Place the correct files in {common.DATA_DIR} -- the hub copy of "
            "sycophancy_on_philpapers2020.jsonl is known to duplicate the NLP survey."
        )
    for name, h in sorted(hashes.items()):
        print(f"  {name}: sha256 {h[:16]}")
    return hashes


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
    max_prompt_tokens: int,
    tokenizer,
    longest_system_prompt: str,
) -> tuple[list[dict], dict]:
    """Round-robin sample n prompts across subsets; returns (rows, stats)."""
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = {}
    stats = {
        "n_strip_fired": 0,
        "n_strip_missed": 0,
        "n_marker_in_bio": 0,
        "n_too_long": 0,
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

    # Round-robin, then truncate. Because each pool's order is fixed by the
    # seed and we truncate rather than resample, --n 20 is a prefix of --n 200.
    order = sorted(pools)
    selected: list[dict] = []
    seen_texts: set[str] = set()
    cursor = {name: 0 for name in order}
    stats["n_duplicate_text"] = 0
    while len(selected) < n:
        progressed = False
        for name in order:
            if len(selected) >= n:
                break
            pool = pools[name]
            while cursor[name] < len(pool):
                cand = pool[cursor[name]]
                cursor[name] += 1
                # Backstop against duplicate turns reaching the paired design,
                # whatever their source. Two prompt_ids sharing a text would
                # put the same user turn in both train and test, which is the
                # leakage grouped_folds exists to prevent.
                key = " ".join(cand["user_prompt"].split()).lower()
                if key in seen_texts:
                    stats["n_duplicate_text"] += 1
                    continue
                rendered = build_chat_prompt(tokenizer, cand["user_prompt"], longest_system_prompt)
                n_tok = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
                if n_tok > max_prompt_tokens:
                    stats["n_too_long"] += 1
                    continue
                cand["n_prompt_tokens_worst_case"] = n_tok
                seen_texts.add(key)
                selected.append(cand)
                progressed = True
                break
        if not progressed:
            break

    if len(selected) < n:
        raise SystemExit(
            f"only {len(selected)} prompts survived filtering, wanted {n}. "
            f"Raise --max-prompt-tokens (currently {max_prompt_tokens}) or lower --n."
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
                        help="Where the sycophancy_on_*.jsonl subsets live. Missing ones are downloaded.")
    parser.add_argument("--model", type=str, default=common.DEFAULT_MODEL, help="Tokenizer to size prompts with.")
    parser.add_argument("--max-prompt-tokens", type=int, default=768)
    parser.add_argument(
        "--keep-choices",
        action="store_true",
        help="Keep the (A)/(B) option block. Makes responses ~1 token, so the first5 and "
        "response activation positions become degenerate. Off by default.",
    )
    parser.add_argument("--show", type=int, default=3, help="Print this many selected prompts.")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print(f"Resolving {len(SUBSETS)} Perez sycophancy subsets ...")
    paths = resolve_subsets(SUBSETS, args.data_dir)
    assert_distinct(paths)
    raw = {name: common.read_jsonl(path) for name, path in paths.items()}
    for name, rows in raw.items():
        print(f"  {name}: {len(rows)} rows")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pairs = common.load_prompt_pairs()
    # Size against the worst case so the cap holds for every pair, not just the
    # one that happens to be shortest.
    longest = max(
        (p[pol] for p in pairs for pol in common.POLARITIES),
        key=lambda s: len(tokenizer(s, add_special_tokens=False)["input_ids"]),
    )

    rows, stats = build_user_prompts(
        raw,
        n=args.n,
        seed=args.seed,
        strip=not args.keep_choices,
        max_prompt_tokens=args.max_prompt_tokens,
        tokenizer=tokenizer,
        longest_system_prompt=longest,
    )

    if not args.keep_choices:
        # Backstop: candidates carrying option markers are filtered in
        # build_user_prompts, so anything surviving here is a bug.
        leaked = [r["prompt_id"] for r in rows if _MARKER_RE.search(r["user_prompt"])]
        if leaked:
            raise SystemExit(f"option markers survived in {len(leaked)} prompts, e.g. {leaked[:3]}")

    common.write_jsonl(args.out, rows)

    print(f"\nWrote {len(rows)} prompts to {args.out}")
    print(
        f"  strip fired: {stats['n_strip_fired']}, missed: {stats['n_strip_missed']}, "
        f"dropped (marker in bio): {stats['n_marker_in_bio']}, dropped (too long): {stats['n_too_long']}, "
        f"dropped (duplicate text): {stats['n_duplicate_text']}"
    )
    print(f"  per subset: {json.dumps(stats['per_subset'])}")
    worst = max(r["n_prompt_tokens_worst_case"] for r in rows)
    print(f"  worst-case rendered length: {worst} tokens (cap {args.max_prompt_tokens})")

    for row in rows[: args.show]:
        print(f"\n--- {row['prompt_id']} ({row['source']}) ---\n{row['user_prompt']}")


if __name__ == "__main__":
    main()
