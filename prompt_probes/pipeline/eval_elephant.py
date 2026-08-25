#!/usr/bin/env python3
"""
Out-of-distribution evaluation on ELEPHANT. Stage 5, and the primary
measurement of the pipeline.

Why this is primary rather than a follow-up. The in-distribution numbers from
train_probes.py saturate: the first full 8B run hit ~1.000 at all 126 settings,
at every position and layer. That is expected, not a bug. Natarajan et al. hit
the same thing and did not report it -- their section 3.5 notes in-distribution
skyline probes "achieved near-perfect AUC (>= 0.993 on nine of ten datasets),
and are omitted from the main figures for clarity." Every number they publish is
OOD, including the section 4.1 ANOVA (computed on validation-set AUC, not on
training separation) and the section 5.3 clustering ("across all evaluation
samples"). So in-distribution AUC is a ceiling check, and the signal lives here.

This is also why no system-prompt-free extraction context is needed. The probes
were trained with an instruction in context and every position could read it;
ELEPHANT responses have no system prompt at all, so the evaluation itself
removes that confound. Same reason the paper needed no such control.

A-priori matched prediction, not a post-hoc oracle. ELEPHANT's three judge
metrics map onto the taxonomy, so which cell should win which column can be
stated before looking (see MATCHED below). The paper's headline +0.108 was an
oracle over deception types, and their own limitations section concedes the
matched probe ranked top-3 for its own type in only 8 of 16 cases. Reporting the
full 14 x 3 grid makes our version falsifiable in advance.

Two ceilings to keep in mind when reading the output:
  * The labels are ELEPHANT's own judge rubric ported to Claude, not ground
    truth (the paper's benchmarks were human-labelled). Judge noise caps the
    achievable AUC, so a weak result cannot be cleanly attributed to the probe.
  * The labels are heavily skewed -- 83-90% positive, minority class 837-1471 of
    8797 -- so n_pos/n_neg is printed beside every AUC.

Extraction runs once (GPU, ~45-90 min for ~8.8k records) and is cached; every
probe x position x layer x metric combination then scores as a dot product.

Usage:
    # one GPU pass, cached thereafter
    python eval_elephant.py --run-name main --extract-only

    # scoring only, CPU
    python eval_elephant.py --run-name main
"""
import argparse
import json
import random
import sys
from datetime import datetime, timezone
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
LABEL_FIELDS = ("validation", "indirectness", "framing")

# Stated before looking at any result. ELEPHANT's metrics correspond to
# taxonomy cells, so each column has a predicted winner:
#   validation   = unconditional emotional validation -> Person-Emotions
#   framing      = accepting the user's framing       -> Position-Subjective/Implicit
#   indirectness = softened feedback / hedged delivery -> the Implicit person cells
MATCHED = {
    "validation": ("pe_explicit", "pe_implicit"),
    "framing": ("ps_implicit",),
    "indirectness": ("pe_implicit", "pt_implicit"),
}


def load_judged(paths: list[Path]) -> list[dict]:
    """Judged records carrying all three metrics, so the 14x3 grid is
    measured on one common set of rows rather than three different ones."""
    out = []
    for path in paths:
        if not path.exists():
            print(f"  missing: {path}")
            continue
        rows = common.read_jsonl(path)
        kept = [r for r in rows if all(r.get(k) is not None for k in LABEL_FIELDS)]
        print(f"  {path.name}: {len(kept)}/{len(rows)} rows with all three metrics")
        for r in kept:
            out.append(
                {
                    "dataset": r.get("dataset") or path.stem.split("_")[0],
                    "row_id": r.get("row_id"),
                    "user_prompt": r["prompt"],
                    "system_prompt": None,  # matches how these were generated
                    "response": r["response"],
                    "example_id": f"{r.get('dataset') or path.stem.split('_')[0]}__{r.get('row_id')}",
                    **{k: int(bool(r[k])) for k in LABEL_FIELDS},
                }
            )
    return out


def selection_split(records: list[dict], frac: float, seed: int) -> set[str]:
    """example_ids reserved for probe *selection*, never used for reporting.

    The paper's section 3.3 analogue: they held out 50 balanced samples from
    three evaluation datasets to pick the single "Best Average" probe, and never
    reused them. Without this, "the best taxonomy probe" is chosen on the same
    rows it is reported on, which is the oracle they were explicit about.

    Split is stratified by dataset and independent of label field, so all three
    metrics are evaluated on identical rows.
    """
    rng = random.Random(seed)
    by_ds: dict[str, list[str]] = {}
    for r in records:
        by_ds.setdefault(r["dataset"], []).append(r["example_id"])
    selection = set()
    for ds, ids in sorted(by_ds.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        selection.update(ids[: int(round(len(ids) * frac))])
    return selection


def extract_activations(run_dir: Path, records: list[dict], args) -> None:
    """One GPU pass over ELEPHANT; cached as activations.npz + index."""
    from utils.model import cleanup as cleanup_model, load_model_and_tokenizer

    out_dir = run_dir / "eval_elephant"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    tokenizer.padding_side = "right"
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    layers = ga.resolve_layers(n_layers, args.layers, args.layer_fracs)
    print(f"{n_layers} blocks, hidden {hidden_dim}, layers {layers}")

    # skip, not fail: ELEPHANT prompt lengths are not ours to control, so one
    # long AITA post must not abort the run. Counts are reported because
    # dropping the longest posts is a selection effect on the AUC.
    prepared, skips, n_mismatch = ga.prepare_records(records, tokenizer, args, on_long_prompt="skip")
    from collections import Counter

    reasons = Counter(s["reason"] for s in skips)
    print(f"{len(prepared)} extractable, {len(skips)} skipped {dict(reasons)}")
    if reasons:
        print(
            "  NOTE: skipped, never truncated. Long prompts are systematically longer AITA\n"
            "  posts, so treat this as a selection effect and raise --max-prompt-tokens /\n"
            "  --max-length if the fraction is large."
        )
    if n_mismatch:
        print(f"  naive-vs-offsets prompt_len mismatches: {n_mismatch} (offsets value used)")

    arrays = ga.extract(model, tokenizer, prepared, layers, hidden_dim, args.batch_size)
    cleanup_model(model, tokenizer)

    np.savez(
        out_dir / "activations.npz",
        **{ga.act_key(pos, layer): arr for (pos, layer), arr in arrays.items()},
    )
    common.write_jsonl(
        out_dir / "activations_index.jsonl",
        [
            {
                "row": i,
                "example_id": p["rec"]["example_id"],
                "dataset": p["rec"]["dataset"],
                "n_response_tokens": p["resp_end"] - p["prompt_len"],
                **{k: p["rec"][k] for k in LABEL_FIELDS},
            }
            for i, p in enumerate(prepared)
        ],
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "layers": layers,
                "positions": list(common.POSITIONS),
                "n_records": len(prepared),
                "n_skipped": len(skips),
                "skip_reasons": dict(reasons),
                "layer_convention": "hidden_states[layer + 1]",
                "code_version": common.get_code_version(),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "script_args": {k: str(v) for k, v in vars(args).items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"cached -> {out_dir / 'activations.npz'}")


def score_all(run_dir: Path, slugs: list[str], positions, layers, label_fields, selection: set[str]) -> list[dict]:
    """Every (cell, position, layer, metric) x {selection, eval} AUC."""
    out_dir = run_dir / "eval_elephant"
    index = common.read_jsonl(out_dir / "activations_index.jsonl")
    is_sel = np.array([r["example_id"] in selection for r in index], dtype=bool)
    labels = {k: np.array([r[k] for r in index], dtype=int) for k in label_fields}
    probes = {s: load_probes(run_dir, s) for s in slugs}

    rows = []
    with np.load(out_dir / "activations.npz") as z:
        for position in positions:
            for layer in layers:
                key = ga.act_key(position, layer)
                if key not in z:
                    continue
                X = z[key].astype(np.float32)
                assert X.shape[0] == len(index), f"{key}: {X.shape[0]} rows vs index {len(index)}"
                for slug in slugs:
                    probe = probes.get(slug, {}).get(key)
                    if probe is None:
                        continue
                    scores = apply_probe(probe, X)
                    for field in label_fields:
                        y = labels[field]
                        for split_name, mask in (("selection", is_sel), ("eval", ~is_sel)):
                            if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
                                continue
                            rows.append(
                                {
                                    "slug": slug,
                                    "position": position,
                                    "layer": layer,
                                    "label_field": field,
                                    "split": split_name,
                                    "auc": safe_auc(y[mask], scores[mask]),
                                    "n": int(mask.sum()),
                                    "n_pos": int((y[mask] == 1).sum()),
                                    "n_neg": int((y[mask] == 0).sum()),
                                }
                            )
    return rows


def print_matrix(rows: list[dict], position: str, layer: int, label_fields, slugs) -> None:
    """14 x 3 AUC grid on the eval half, with the matched prediction marked."""
    sel = {
        (r["slug"], r["label_field"]): r
        for r in rows
        if r["position"] == position and r["layer"] == layer and r["split"] == "eval"
    }
    present = [s for s in slugs if any((s, f) in sel for f in label_fields)]
    if not present:
        return
    header = "".join(f"{f[:11]:>13}" for f in label_fields)
    print(f"\n  {position}_L{layer:02d}   (* = predicted match; AUC on the eval half)")
    print(f"  {'cell':<26}{header}")
    for slug in present:
        line = f"  {slug:<26}"
        for f in label_fields:
            r = sel.get((slug, f))
            mark = "*" if slug in MATCHED.get(f, ()) else " "
            line += f"{(r['auc'] if r else float('nan')):>12.3f}{mark}"
        print(line)
    first = next(iter(sel.values()))
    print(f"  n={first['n']}  (pos/neg per metric: " + ", ".join(
        f"{f} {sel[(present[0], f)]['n_pos']}/{sel[(present[0], f)]['n_neg']}"
        for f in label_fields if (present[0], f) in sel
    ) + ")")

    # Does the predicted cell actually win its column?
    for f in label_fields:
        col = [(s, sel[(s, f)]["auc"]) for s in present if (s, f) in sel and sel[(s, f)]["auc"] is not None]
        if not col:
            continue
        ranked = sorted(col, key=lambda kv: -kv[1])
        winner, best = ranked[0]
        predicted = MATCHED.get(f, ())
        ranks = {s: i + 1 for i, (s, _) in enumerate(ranked)}
        pred_ranks = {s: ranks[s] for s in predicted if s in ranks}
        verdict = "HIT" if any(r <= 3 for r in pred_ranks.values()) else "miss"
        print(
            f"    {f:<13} best={winner} ({best:.3f})  predicted={list(predicted)} "
            f"ranks={pred_ranks}  top3={verdict}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--judged", type=Path, nargs="+", default=DEFAULT_JUDGED)
    parser.add_argument("--model", type=str, default=common.DEFAULT_MODEL)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--layer-fracs", type=float, nargs="+", default=None)
    parser.add_argument("--positions", type=str, nargs="+", default=None, choices=common.POSITIONS)
    parser.add_argument("--label-fields", type=str, nargs="+", default=list(LABEL_FIELDS), choices=LABEL_FIELDS)
    common.add_cells_arg(parser)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Random subsample (seeded), not a head.")
    parser.add_argument("--selection-frac", type=float, default=0.3, help="Reserved for probe selection.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if the cache exists.")
    args = parser.parse_args()

    run_dir = common.resolve_run_dir(args.run_name, create=False)
    if not run_dir.exists():
        raise SystemExit(f"no such run: {run_dir}")
    out_dir = run_dir / "eval_elephant"

    print("Loading judged ELEPHANT records ...")
    records = load_judged(list(args.judged))
    if not records:
        raise SystemExit("no judged records loaded")
    if args.limit:
        # Seeded subsample, not a head: the files concatenate in order, so a
        # head takes one dataset only.
        random.Random(args.seed).shuffle(records)
        records = records[: args.limit]
    print(f"{len(records)} records across {len(set(r['dataset'] for r in records))} datasets")
    for f in args.label_fields:
        pos = sum(r[f] for r in records)
        print(f"  {f:<13} pos {pos} / neg {len(records) - pos}  (pos rate {pos / len(records):.3f})")

    cache = out_dir / "activations.npz"
    if args.overwrite or not cache.exists():
        extract_activations(run_dir, records, args)
    else:
        print(f"using cached activations at {cache} (--overwrite to redo)")
    if args.extract_only:
        return

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    positions = args.positions or meta["positions"]
    layers = args.layers or meta["layers"]
    slugs = [
        s for s in (args.cells or common.all_slugs(include_neutral=False))
        if (run_dir / "probes" / s / "probes.npz").exists()
    ]
    if not slugs:
        raise SystemExit("no trained probes found -- run train_probes.py first")

    index = common.read_jsonl(out_dir / "activations_index.jsonl")
    selection = selection_split(index, args.selection_frac, args.seed)
    print(f"\nselection split: {len(selection)} rows reserved, {len(index) - len(selection)} for reporting")

    rows = score_all(run_dir, slugs, positions, layers, args.label_fields, selection)
    summary = {
        "schema_version": 1,
        "code_version": common.get_code_version(),
        "n_records": len(index),
        "selection_frac": args.selection_frac,
        "seed": args.seed,
        "matched_prediction": {k: list(v) for k, v in MATCHED.items()},
        "note": "Judge labels, not ground truth, and 83-90% positive. AUC is base-rate "
        "insensitive but the minority class is small; read n_pos/n_neg alongside.",
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for position in positions:
        for layer in layers:
            print_matrix(rows, position, layer, args.label_fields, slugs)

    # Best-average probe, chosen on the selection half and reported on the eval
    # half -- the honest version of "which single probe should you deploy".
    print("\n--- best-average probe (selected on the selection half) ---")
    by_probe: dict[tuple, list[float]] = {}
    for r in rows:
        if r["split"] == "selection" and r["auc"] is not None:
            by_probe.setdefault((r["slug"], r["position"], r["layer"]), []).append(r["auc"])
    if by_probe:
        best = max(by_probe.items(), key=lambda kv: float(np.mean(kv[1])))
        (slug, position, layer), sel_aucs = best
        held = [
            r["auc"] for r in rows
            if r["split"] == "eval" and r["slug"] == slug and r["position"] == position
            and r["layer"] == layer and r["auc"] is not None
        ]
        print(f"  {slug} @ {position}_L{layer:02d}")
        print(f"    selection-half mean AUC {float(np.mean(sel_aucs)):.3f}")
        print(f"    eval-half       mean AUC {float(np.mean(held)):.3f}   <- the reportable number")

    print(f"\n{len(rows)} rows -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
