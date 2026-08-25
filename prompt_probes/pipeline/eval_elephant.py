#!/usr/bin/env python3
"""
STUB. Score ELEPHANT responses with a trained prompt-probe and compare against
the LLM-judge labels.

The evaluation design is deliberately not settled yet -- --label-field has no
default for that reason. What is implemented here is the mechanical part:
loading judged records, rendering them the way they were generated, extracting
the same positions/layers, and applying a saved probe.

Why this is the real test. The probes are trained to predict which system
prompt was in context, and ELEPHANT responses have no system prompt at all.
That distribution shift is deliberate and is exactly the paper's evaluation
setup: Natarajan et al. train on instruction pairs and transfer to ten
instruction-free on-policy datasets, and the size of the resulting gap is their
headline result rather than a flaw to control away. Two consequences:

  * `last_prompt` probes have the least reason to transfer. That position was
    reading system-prompt tokens which are simply absent here.
  * `first5` is the position with a principled claim to transferring, since it
    was the intent measurement rather than the content one.

Also note the judge ceiling: `validation`, `indirectness` and `framing` are
GPT-4o-style judge labels ported to Claude, not ground truth. They cap what any
probe can appear to recover, which is part of why this pipeline trains against
an intervention instead.

Usage (once a label definition is chosen):
    python eval_elephant.py --run-name main --cell general_baseline \\
        --position first5 --layer 16 --label-field validation
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
import get_activations as ga  # noqa: E402
from analyze_probes import apply_probe, load_probes  # noqa: E402
from train_probes import safe_auc  # noqa: E402

JUDGED_DIR = common.SYCOPHANCY_DIR / "results" / "generations"
DEFAULT_JUDGED = [
    JUDGED_DIR / "OEQ_social_sycophancy_judged.jsonl",
    JUDGED_DIR / "SS_social_sycophancy_judged.jsonl",
    JUDGED_DIR / "AITA-YTA_social_sycophancy_judged.jsonl",
]
# TODO(design): the ELEPHANT social-sycophancy judge emits three binary metrics.
# Which one (or which combination) counts as the probe's target is exactly the
# question that was deferred, so it stays an explicit CLI choice.
LABEL_FIELDS = ("validation", "indirectness", "framing")


def load_judged(paths: list[Path], label_field: str) -> list[dict]:
    """Judged ELEPHANT records that carry the requested label."""
    out = []
    for path in paths:
        if not path.exists():
            print(f"  missing: {path}")
            continue
        rows = common.read_jsonl(path)
        kept = [r for r in rows if r.get(label_field) is not None]
        print(f"  {path.name}: {len(kept)}/{len(rows)} rows with '{label_field}'")
        for r in kept:
            out.append(
                {
                    "dataset": r.get("dataset") or path.stem,
                    "row_id": r.get("row_id"),
                    "prompt": r["prompt"],
                    "response": r["response"],
                    "label": int(bool(r[label_field])),
                }
            )
    return out


def build_texts(tokenizer, records: list[dict]) -> list[dict]:
    """Shape records for ga.prepare_records, which renders the chat template.

    system_prompt is None so the text matches how these responses were actually
    generated -- the same convention as build_labeled_text in
    social_sycophancy_judge.py, so the probe sees what the judge scored.
    """
    return [
        {
            **r,
            "user_prompt": r["prompt"],
            "system_prompt": None,
            "example_id": f"{r['dataset']}__{r['row_id']}",
        }
        for r in records
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--cell", type=str, required=True, help="Slug of the probe to evaluate.")
    parser.add_argument("--position", type=str, required=True, choices=common.POSITIONS)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--label-field",
        type=str,
        required=True,
        choices=LABEL_FIELDS,
        help="Judge metric to score against. No default: this is the deferred design decision.",
    )
    parser.add_argument("--judged", type=Path, nargs="+", default=DEFAULT_JUDGED)
    parser.add_argument("--model", type=str, default=common.DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Random subsample (seeded), not a head.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    run_dir = common.resolve_run_dir(args.run_name, create=False)
    probes = load_probes(run_dir, args.cell)
    key = ga.act_key(args.position, args.layer)
    if key not in probes:
        raise SystemExit(f"no probe for {args.cell} at {key}; available: {sorted(probes)}")
    probe = probes[key]

    if args.position == "last_prompt":
        print(
            "NOTE: last_prompt was trained on a position whose signal was the system-prompt\n"
            "tokens themselves. ELEPHANT prompts have no system prompt, so a null result here\n"
            "is the expected outcome rather than evidence about sycophancy representations."
        )

    print(f"Loading judged records (label={args.label_field}) ...")
    records = load_judged(list(args.judged), args.label_field)
    if not records:
        raise SystemExit("no judged records loaded")
    if args.limit:
        # Shuffle before truncating. The files are concatenated in order, so a
        # plain head would take one dataset only, and the judge labels are
        # heavily skewed within a dataset -- a naive --limit 60 yields 59/60
        # positives from OEQ alone and an AUC that means nothing.
        import random

        random.Random(args.seed).shuffle(records)
        records = records[: args.limit]
    label_counts = collections.Counter(r["label"] for r in records)
    ds_counts = collections.Counter(r["dataset"] for r in records)
    print(f"{len(records)} records | labels {dict(label_counts)} | datasets {dict(ds_counts)}")
    if min(label_counts.values(), default=0) < 10:
        print("  WARNING: one class has under 10 examples; the AUC will be very noisy.")

    from utils.model import cleanup as cleanup_model, load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(args.model)
    tokenizer.padding_side = "right"

    # skip, not fail: ELEPHANT prompt lengths are not ours to control, so one
    # long AITA post must not abort the evaluation. The counts are reported
    # because dropping the longest posts is a selection effect.
    prepared, skips, _ = ga.prepare_records(
        build_texts(tokenizer, records), tokenizer, args, on_long_prompt="skip"
    )
    reasons = collections.Counter(s["reason"] for s in skips)
    print(f"{len(prepared)} extractable, {len(skips)} skipped {dict(reasons)}")
    if reasons:
        print(
            "  NOTE: skipped records are dropped, never truncated. Long prompts are "
            "systematically longer AITA posts, so treat this as a selection effect on the "
            "reported AUC and raise --max-prompt-tokens/--max-length if the fraction is large."
        )
    arrays = ga.extract(model, tokenizer, prepared, [args.layer], model.config.hidden_size, args.batch_size)
    cleanup_model(model, tokenizer)

    X = arrays[(args.position, args.layer)]
    y = np.array([p["rec"]["label"] for p in prepared], dtype=int)
    scores = apply_probe(probe, X)

    auc = safe_auc(y, scores)
    out_dir = run_dir / "eval_elephant"
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.cell}_{key}_{args.label_field}"

    per_dataset = {}
    datasets = [p["rec"]["dataset"] for p in prepared]
    for ds in sorted(set(datasets)):
        m = np.array([d == ds for d in datasets], dtype=bool)
        per_dataset[ds] = {"n": int(m.sum()), "n_pos": int((y[m] == 1).sum()), "auc": safe_auc(y[m], scores[m])}

    metrics = {
        "run": args.run_name,
        "cell": args.cell,
        "position": args.position,
        "layer": args.layer,
        "label_field": args.label_field,
        "n": len(y),
        "n_pos": int((y == 1).sum()),
        "n_skipped": len(skips),
        "skip_reasons": dict(reasons),
        "auc_overall": auc,
        "per_dataset": per_dataset,
        "note": "Probes trained with a system prompt in context; ELEPHANT has none. "
        "See module docstring.",
    }
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    common.write_jsonl(
        out_dir / f"scores_{tag}.jsonl",
        [
            {"example_id": p["rec"]["example_id"], "dataset": p["rec"]["dataset"], "label": int(label), "score": float(s)}
            for p, label, s in zip(prepared, y, scores)
        ],
    )

    print(f"\noverall AUC: {auc}")
    for ds, e in per_dataset.items():
        print(f"  {ds:<18} n={e['n']:<6} pos={e['n_pos']:<6} auc={e['auc']}")
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
