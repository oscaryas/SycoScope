"""
Shared OOD-evaluation machinery for the scorer/eval_*.py scripts: build an
activation cache once (analyze_probes.cache_activations.extract_records, one
GPU pass), then score pickled probes on it as a probe x position x layer x
label AUC/accuracy grid.

Cache format: probes_core.save_cache (.npz with "L##" per-layer arrays, "y",
"groups", "layers") -- one file per token position, with a
"<cache>.meta.json" sidecar recording "token_position". The eval_* scripts add
per-row extras: "example_id", "dataset", "n_response_tokens" and one
"label__<field>" int array per label field (-1 = no label on this row). A plain
cache_activations.py cache (no label__ arrays) is scored on its "y".

With several --positions, position P's cache is <stem>_<P>.npz next to
--cache; with one, --cache itself.

Probe format: probes_core.pickle_weights payloads (see scorer.score_probes);
each layer is scored at its best_C unless --C is given.
"""
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from utils import common
from analyze_probes import probes_core as core
from analyze_probes.probes_core import safe_auc
from scorer.score_probes import SCORES_DIR, apply_probe, load_probe_set, parse_probe_spec, select_probe

N_NULL_DIRECTIONS = 20
NULL_SLUGS = common.NULL_SLUGS
LABEL_PREFIX = "label__"


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


# ---------------------------------------------------------------------------
# Cache IO
# ---------------------------------------------------------------------------


def cache_paths(cache: Path, positions) -> dict[str, Path]:
    """{position: .npz path}: --cache itself for one position, else
    <stem>_<position><suffix> siblings."""
    cache = Path(cache)
    positions = list(positions)
    if len(positions) == 1:
        return {positions[0]: cache}
    return {p: cache.with_name(f"{cache.stem}_{p}{cache.suffix or '.npz'}") for p in positions}


def meta_path(cache: Path) -> Path:
    return Path(cache).with_name(Path(cache).name + ".meta.json")


def read_meta(cache: Path) -> dict:
    path = meta_path(cache)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def needs_extraction(paths: dict[str, Path], args) -> bool:
    missing = [str(p) for p in paths.values() if not p.exists()]
    if not (args.overwrite or missing):
        print(f"using cached activations {', '.join(str(p) for p in paths.values())} (--overwrite to redo)")
        return False
    if not args.model:
        raise SystemExit(f"cache missing ({missing}); pass --model to extract it")
    return True


def build_cache(paths: dict[str, Path], records: list[dict], label_fields, index_fields, args,
                fix_records=None, meta_extra: dict | None = None) -> list[dict]:
    """One GPU pass over `records` for every position in `paths`, saved as one
    probes_core.save_cache .npz per position. `fix_records(kept_records)` may
    edit labels after over-length rows are dropped (e.g. orphaned pairs).
    Returns the kept records, in cache row order."""
    from collections import Counter

    from analyze_probes.cache_activations import extract_records

    prepared, skips, arrays, layers = extract_records(
        records, args.model, args.layers, None, positions=tuple(paths), average="mean",
        batch_size=args.batch_size, max_length=args.max_length, model_path=getattr(args, "model_path", None),
    )
    recs = [p["rec"] for p in prepared]
    if fix_records is not None:
        fix_records(recs)
    labels = {f: np.array([-1 if r.get(f) is None else int(r[f]) for r in recs], dtype=int) for f in label_fields}
    extra = {
        "example_id": np.array([str(r["example_id"]) for r in recs]),
        "dataset": np.array([str(r["dataset"]) for r in recs]),
        "n_response_tokens": np.array([p["resp_end"] - p["prompt_len"] for p in prepared], dtype=int),
        **{LABEL_PREFIX + f: labels[f] for f in label_fields},
        **{f: np.array(["" if r.get(f) is None else str(r[f]) for r in recs])
           for f in index_fields if f not in label_fields and f not in ("dataset", "example_id")},
    }
    reasons = Counter(s["reason"] for s in skips)
    for position, path in paths.items():
        core.save_cache(path, {layer: arrays[(position, layer)] for layer in layers},
                        y=labels[label_fields[0]], groups=[split_key(r) for r in recs], **extra)
        meta = {
            "model": args.model, "layers": layers, "layer_convention": "hidden_states[layer + 1]",
            "token_position": position, "average": "mean", "label_fields": list(label_fields),
            "n_examples": len(recs), "n_skipped": len(skips), "skip_reasons": dict(reasons),
            "code_version": common.get_code_version(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script_args": {k: str(v) for k, v in vars(args).items()},
            **(meta_extra or {}),
        }
        meta_path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"cached -> {path}")
    return recs


def load_table(path: Path, position: str | None = None) -> dict:
    """A cache as a scoring table: acts per layer, per-row metadata, labels
    ({field: int array, -1 = unlabeled}) and the pseudo-records that
    selection_split needs."""
    cache = core.load_cache(path)
    meta = read_meta(path)
    n = len(cache["y"])
    example_id = cache["example_id"].astype(str) if "example_id" in cache else np.array([str(i) for i in range(n)])
    datasets = cache["dataset"].astype(str) if "dataset" in cache else np.full(n, Path(path).stem)
    groups = cache["groups"].astype(str)
    labels = {k[len(LABEL_PREFIX):]: cache[k].astype(int) for k in cache if k.startswith(LABEL_PREFIX)}
    if not labels:
        labels = {"y": cache["y"].astype(int)}
    lengths = cache["n_response_tokens"].astype(np.float32) if "n_response_tokens" in cache else None
    return {
        "path": Path(path), "meta": meta, "position": meta.get("token_position") or position or "cache",
        "acts": cache["acts"], "layers": cache["layers"], "labels": labels, "lengths": lengths,
        "datasets": datasets, "groups": groups, "example_id": example_id, "cache": cache,
        "records": [{"dataset": d, "group_id": g, "example_id": e} for d, g, e in zip(datasets, groups, example_id)],
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


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
    """One row per (split, dataset). Label -1 means the row has no label here.
    accuracy thresholds the decision function at 0 (the probe's own boundary)."""
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
                    "accuracy": float(((s > 0).astype(int) == yy).mean()),
                    "n": int(len(yy)),
                    "n_pos": int((yy == 1).sum()),
                    "n_neg": int((yy == 0).sum()),
                }
            )
    return rows


def null_rows(X, lengths, labels, is_sel, datasets, group_ids, pair_fields, position, layer, rng) -> list[dict]:
    """Random directions and response length, scored exactly like a probe
    (AUC only: a null has no boundary, so accuracy is None)."""
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
                rows.append({**entry["row"], "slug": slug, "auc": auc, "accuracy": None})

        if lengths is None:
            continue
        for r in auc_rows(lengths, y, is_sel, datasets, groups, {**base, "slug": "null_length"}):
            if r["auc"] is not None:
                r["auc"] = max(r["auc"], 1.0 - r["auc"])
            r["accuracy"] = None
            rows.append(r)
    return rows


def resolve_label_fields(tables: list[dict], label_fields) -> list[str]:
    """The eval's label fields present in every cache; else the cache's own
    labels (e.g. "y" for a plain cache_activations.py cache)."""
    present = [f for f in label_fields if all(f in t["labels"] for t in tables)]
    return present or sorted(set.intersection(*(set(t["labels"]) for t in tables)))


def score_all(probes: dict, tables: list[dict], label_fields, selection, seed, pair_fields=(), C=None) -> list[dict]:
    """Every probe x cached position x layer (probe layers present in the
    cache) x label, plus the random-direction and length nulls."""
    rng = np.random.default_rng(seed)
    rows = []
    for table in tables:
        position = table["position"]
        is_sel = np.array([split_key(r) in selection for r in table["records"]], dtype=bool)
        labels = {f: table["labels"][f] for f in label_fields}
        for layer in table["layers"]:
            X = table["acts"][layer].astype(np.float32)
            assert X.shape[0] == len(is_sel), f"L{layer:02d}: {X.shape[0]} rows vs {len(is_sel)} labels"
            scored = False
            for name, payload in probes.items():
                probe = select_probe(payload, layer, C)
                if probe is None:
                    continue
                scored = True
                scores = apply_probe(probe, X)
                for field in label_fields:
                    rows += auc_rows(
                        scores, labels[field], is_sel, table["datasets"],
                        table["groups"] if field in pair_fields else None,
                        {"slug": name, "position": position, "layer": int(layer), "label_field": field,
                         "C": probe["C"]},
                    )
            if scored:
                rows += null_rows(X, table["lengths"], labels, is_sel, table["datasets"], table["groups"],
                                  pair_fields, position, int(layer), rng)
    if not rows:
        raise SystemExit("no probe layer is present in the cache(s) -- check --layers at extraction")
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


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
    print(f"  {'probe':<26}{header}")
    for slug in present:
        line = f"  {slug:<26}"
        for f in label_fields:
            r = sel.get((slug, f))
            mark = "*" if slug in matched.get(f, ()) else " "
            auc = r["auc"] if r and r["auc"] is not None else float("nan")
            line += f"{auc:>12.3f}{mark}"
        print(line)
    counts = [
        f"{f} {sel[(present[0], f)]['n']} ({sel[(present[0], f)]['n_pos']}/{sel[(present[0], f)]['n_neg']})"
        for f in label_fields if (present[0], f) in sel
    ]
    print("  n pos/neg: " + ", ".join(counts))


def print_by_dataset(rows, position, layer, label_fields, slugs) -> None:
    """Per-dataset AUC for the best pooled probe: pooled sets whose base rates
    differ by dataset reward dataset identity on its own."""
    pooled = [
        r for r in rows
        if r["position"] == position and r["layer"] == layer and r["split"] == "eval"
        and r["dataset"] == "all" and r["slug"] in slugs and r["auc"] is not None
    ]
    datasets = sorted({r["dataset"] for r in rows if r["dataset"] != "all"})
    if not pooled or len(datasets) < 2:
        return
    print(f"\n  {position}_L{layer:02d}   per-dataset (best pooled probe per label)")
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


def report(output: Path, rows, label_fields, slugs, extra: dict, matched=None) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 3,
        "code_version": common.get_code_version(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
        "rows": rows,
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    combos = sorted({(r["position"], r["layer"]) for r in rows})
    for position, layer in combos:
        print_matrix(rows, position, layer, label_fields, slugs, matched)
        print_by_dataset(rows, position, layer, label_fields, slugs)
    print_best_average(rows)
    print(f"\n{len(rows)} rows -> {output}")


def default_output(eval_name: str, cache: Path) -> Path:
    return SCORES_DIR / eval_name / f"{Path(cache).stem}.json"


def score_and_report(args, paths: dict[str, Path], eval_name: str, label_fields, pair_fields=(),
                     extra: dict | None = None, matched=None, selection_spec: dict | None = None) -> list[dict]:
    """Load --probe pickles and the cache(s), score the grid, write the JSON."""
    if not args.probe:
        raise SystemExit("--probe is required unless --extract-only")
    probes = load_probe_set(args.probe)
    tables = [load_table(path, position) for position, path in paths.items()]
    n_rows = {len(t["records"]) for t in tables}
    if len(n_rows) != 1:
        raise SystemExit(f"position caches disagree on row count: {n_rows}")
    fields = resolve_label_fields(tables, label_fields)
    spec = selection_spec or {"frac": args.selection_frac, "seed": args.seed}
    selection = selection_split(tables[0]["records"], spec["frac"], spec["seed"])
    print(f"\nselection split: {len(selection)} keys reserved for probe selection; "
          f"labels {fields}; probes {list(probes)}")
    rows = score_all(probes, tables, fields, selection, args.seed, pair_fields, args.C)
    output = args.output or default_output(eval_name, next(iter(paths.values())))
    report(
        output, rows, fields, list(probes),
        {
            "eval": eval_name,
            "probes": {name: str(path) for name, path in map(parse_probe_spec, args.probe)},
            "caches": {t["position"]: str(t["path"]) for t in tables},
            "n_records": len(tables[0]["records"]),
            "selection_frac": spec["frac"],
            "seed": spec["seed"],
            "C": args.C if args.C is not None else "best_C per layer",
            "pair_fields": list(pair_fields),
            **(extra or {}),
        },
        matched,
    )
    return rows


def add_common_args(parser) -> None:
    parser.add_argument("--probe", nargs="+", default=None,
                        help="Pickled probe(s) from analyze_probes/train_probes.py: path or name=path.")
    parser.add_argument("--cache", type=Path, required=True,
                        help="Activation cache .npz; extracted from this eval's inputs if missing (needs --model).")
    parser.add_argument("--output", type=Path, default=None, help=f"Results JSON (default under {SCORES_DIR}).")
    parser.add_argument("--model", type=str, default=None, help="Model to extract with (only when extracting).")
    parser.add_argument("--positions", type=str, nargs="+", default=["response"], choices=common.POSITIONS)
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Extraction layers (default 0.25/0.5/0.75 depth).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None, help="Random subsample (seeded), not a head.")
    parser.add_argument("--selection-frac", type=float, default=0.3, help="Reserved for probe selection.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--C", type=float, default=None, help="Score at this C (default: each layer's best_C).")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if the cache exists.")
