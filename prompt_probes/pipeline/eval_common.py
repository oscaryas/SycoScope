"""
Shared OOD-evaluation machinery for eval_elephant / eval_are_you_sure / eval_moral:
one cached extraction pass, then a probe x position x layer x label AUC grid.
"""
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import common
import get_activations as ga
from analyze_probes import apply_probe, load_probes
from train_probes import safe_auc

N_NULL_DIRECTIONS = 20
NULL_SLUGS = common.NULL_SLUGS


def split_key(record: dict) -> str:
    """group_id where a record belongs to a pair, else its own example_id."""
    return str(record.get("group_id") or record["example_id"])


def selection_split(records: list[dict], frac: float, seed: int) -> set[str]:
    """Split keys reserved for probe selection, never used for reporting.
    Grouped, so both sides of a pair land on the same side of the split."""
    rng = random.Random(seed)
    by_ds: dict[str, set[str]] = {}
    for r in records:
        by_ds.setdefault(r["dataset"], set()).add(split_key(r))
    selection = set()
    for ds, keys in sorted(by_ds.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        selection.update(keys[: int(round(len(keys) * frac))])
    return selection


def extract_activations(out_dir: Path, records: list[dict], index_fields, args) -> None:
    """One GPU pass; cached as activations.npz + activations_index.jsonl."""
    from utils.model import cleanup as cleanup_model, load_model_and_tokenizer

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    tokenizer.padding_side = "right"
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    layers = ga.resolve_layers(n_layers, args.layers, args.layer_fracs)
    print(f"{n_layers} blocks, hidden {hidden_dim}, layers {layers}")

    from collections import Counter

    prepared, skips = ga.prepare_records(records, tokenizer, args)
    reasons = Counter(s["reason"] for s in skips)
    print(f"{len(prepared)} extractable, {len(skips)} skipped {dict(reasons)}")

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
                **{k: p["rec"].get(k) for k in index_fields},
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


def group_mean(scores: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Collapse to one score and one label per group, for pair-level metrics."""
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    keys = sorted(by_group)
    return (
        np.array([scores[by_group[k]].mean() for k in keys]),
        np.array([y[by_group[k]][0] for k in keys]),
    )


def auc_rows(scores, y, is_sel, datasets, groups, base: dict) -> list[dict]:
    """One row per (split, dataset). Label -1 means the row has no label here."""
    rows = []
    for split, split_mask in (("selection", is_sel), ("eval", ~is_sel)):
        for ds in ("all", *sorted(set(datasets))):
            mask = split_mask & (y >= 0)
            if ds != "all":
                mask = mask & (datasets == ds)
            if mask.sum() == 0:
                continue
            s, yy = scores[mask], y[mask]
            if groups is not None:
                s, yy = group_mean(s, yy, groups[mask])
            if len(np.unique(yy)) < 2:
                continue
            rows.append(
                {
                    **base,
                    "split": split,
                    "dataset": ds,
                    "auc": safe_auc(yy, s),
                    "n": int(len(yy)),
                    "n_pos": int((yy == 1).sum()),
                    "n_neg": int((yy == 0).sum()),
                }
            )
    return rows


def null_rows(X, lengths, labels, is_sel, datasets, group_ids, pair_fields, position, layer, rng) -> list[dict]:
    """Random directions and response length, scored exactly like a probe."""
    directions = rng.normal(size=(N_NULL_DIRECTIONS, X.shape[1])).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    random_scores = X @ directions.T

    rows = []
    for field, y in labels.items():
        groups = group_ids if field in pair_fields else None
        base = {"position": position, "layer": layer, "label_field": field}

        collected: dict[tuple, dict] = {}
        for k in range(N_NULL_DIRECTIONS):
            for r in auc_rows(random_scores[:, k], y, is_sel, datasets, groups, base):
                if r["auc"] is None:
                    continue
                entry = collected.setdefault((r["split"], r["dataset"]), {"row": r, "aucs": []})
                entry["aucs"].append(r["auc"])
        for entry in collected.values():
            for slug, auc in (
                ("null_random_p50", float(np.median(entry["aucs"]))),
                ("null_random_max", float(np.max(entry["aucs"]))),
            ):
                rows.append({**entry["row"], "slug": slug, "auc": auc})

        for r in auc_rows(lengths, y, is_sel, datasets, groups, {**base, "slug": "null_length"}):
            if r["auc"] is not None:
                r["auc"] = max(r["auc"], 1.0 - r["auc"])
            rows.append(r)
    return rows


def score_all(run_dir, out_dir, slugs, positions, layers, label_fields, selection, seed, pair_fields=()) -> list[dict]:
    index = common.read_jsonl(out_dir / "activations_index.jsonl")
    is_sel = np.array([split_key(r) in selection for r in index], dtype=bool)
    datasets = np.array([r["dataset"] for r in index])
    group_ids = np.array([split_key(r) for r in index])
    lengths = np.array([r["n_response_tokens"] for r in index], dtype=np.float32)
    labels = {
        f: np.array([-1 if r.get(f) is None else int(r[f]) for r in index], dtype=int)
        for f in label_fields
    }
    probes = {s: load_probes(run_dir, s) for s in slugs}
    rng = np.random.default_rng(seed)

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
                        rows += auc_rows(
                            scores,
                            labels[field],
                            is_sel,
                            datasets,
                            group_ids if field in pair_fields else None,
                            {"slug": slug, "position": position, "layer": layer, "label_field": field},
                        )
                rows += null_rows(
                    X, lengths, labels, is_sel, datasets, group_ids, pair_fields, position, layer, rng
                )
    return rows


def print_matrix(rows, position, layer, label_fields, slugs, matched=None) -> None:
    """Cell x label AUC grid on the eval half, pooled across datasets."""
    matched = matched or {}
    sel = {
        (r["slug"], r["label_field"]): r
        for r in rows
        if r["position"] == position and r["layer"] == layer
        and r["split"] == "eval" and r["dataset"] == "all"
    }
    present = [s for s in [*slugs, *NULL_SLUGS] if any((s, f) in sel for f in label_fields)]
    if not present:
        return
    header = "".join(f"{f[:11]:>13}" for f in label_fields)
    print(f"\n  {position}_L{layer:02d}   (AUC on the eval half)")
    print(f"  {'cell':<26}{header}")
    for slug in present:
        line = f"  {slug:<26}"
        for f in label_fields:
            r = sel.get((slug, f))
            mark = "*" if slug in matched.get(f, ()) else " "
            line += f"{(r['auc'] if r else float('nan')):>12.3f}{mark}"
        print(line)
    counts = [
        f"{f} {sel[(present[0], f)]['n']} ({sel[(present[0], f)]['n_pos']}/{sel[(present[0], f)]['n_neg']})"
        for f in label_fields if (present[0], f) in sel
    ]
    print("  n pos/neg: " + ", ".join(counts))


def print_by_dataset(rows, position, layer, label_fields, slugs) -> None:
    """Per-dataset AUC for the best pooled cell: pooled sets whose base rates
    differ by dataset reward dataset identity on its own."""
    pooled = [
        r for r in rows
        if r["position"] == position and r["layer"] == layer and r["split"] == "eval"
        and r["dataset"] == "all" and r["slug"] in slugs and r["auc"] is not None
    ]
    datasets = sorted({r["dataset"] for r in rows if r["dataset"] != "all"})
    if not pooled or len(datasets) < 2:
        return
    print(f"\n  {position}_L{layer:02d}   per-dataset (best pooled cell per label)")
    for field in label_fields:
        column = [r for r in pooled if r["label_field"] == field]
        if not column:
            continue
        best = max(column, key=lambda r: r["auc"])
        parts = []
        for ds in datasets:
            match = [
                r for r in rows
                if r["slug"] == best["slug"] and r["position"] == position and r["layer"] == layer
                and r["split"] == "eval" and r["label_field"] == field and r["dataset"] == ds
            ]
            if match and match[0]["auc"] is not None:
                parts.append(f"{ds} {match[0]['auc']:.3f} (n={match[0]['n']})")
        print(f"    {field:<16} {best['slug']:<24} pooled {best['auc']:.3f} | " + "  ".join(parts))


def print_best_average(rows) -> None:
    """Best-average probe, chosen on the selection half, reported on the eval half."""
    print("\n--- best-average probe (selected on the selection half) ---")
    by_probe: dict[tuple, list[float]] = {}
    for r in rows:
        if (r["split"] == "selection" and r["dataset"] == "all"
                and r["slug"] not in NULL_SLUGS and r["auc"] is not None):
            by_probe.setdefault((r["slug"], r["position"], r["layer"]), []).append(r["auc"])
    if not by_probe:
        return
    (slug, position, layer), sel_aucs = max(by_probe.items(), key=lambda kv: float(np.mean(kv[1])))
    held = [
        r["auc"] for r in rows
        if r["split"] == "eval" and r["dataset"] == "all" and r["slug"] == slug
        and r["position"] == position and r["layer"] == layer and r["auc"] is not None
    ]
    print(f"  {slug} @ {position}_L{layer:02d}")
    print(f"    selection-half mean AUC {float(np.mean(sel_aucs)):.3f}")
    if held:
        print(f"    eval-half       mean AUC {float(np.mean(held)):.3f}   <- the reportable number")


def report(out_dir, rows, positions, layers, label_fields, slugs, extra: dict, matched=None) -> None:
    summary = {
        "schema_version": 2,
        "code_version": common.get_code_version(),
        **extra,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for position in positions:
        for layer in layers:
            print_matrix(rows, position, layer, label_fields, slugs, matched)
            print_by_dataset(rows, position, layer, label_fields, slugs)
    print_best_average(rows)
    print(f"\n{len(rows)} rows -> {out_dir / 'summary.json'}")


def resolve_eval_targets(run_dir: Path, out_dir: Path, args):
    """(positions, layers, slugs) for a cached extraction with trained probes."""
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    positions = args.positions or meta["positions"]
    layers = args.layers or meta["layers"]
    slugs = [
        s for s in (args.cells or common.all_slugs(include_neutral=False))
        if (run_dir / "probes" / s / "probes.npz").exists()
    ]
    if not slugs:
        raise SystemExit("no trained probes found -- run train_probes.py first")
    return positions, layers, slugs


def add_common_args(parser) -> None:
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model", type=str, default=common.DEFAULT_MODEL)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--layer-fracs", type=float, nargs="+", default=None)
    parser.add_argument("--positions", type=str, nargs="+", default=None, choices=common.POSITIONS)
    common.add_cells_arg(parser)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Random subsample (seeded), not a head.")
    parser.add_argument("--selection-frac", type=float, default=0.3, help="Reserved for probe selection.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if the cache exists.")
