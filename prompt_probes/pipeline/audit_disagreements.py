#!/usr/bin/env python3
"""
V1 analysis plan, step 5: audit disagreements between probes.

Extends the existing hedging investigation (FINDINGS.md sec. 4, which already
has text-level confirmation of the ctrl_calibrated_hedging inversion) to
praise, agreement, and emotional validation -- the three other priority
comparisons from step 4.

For each probe pair, scores are converted to within-probe percentile rank on
a common example pool (every test-split row from every source cell in that
step-4 comparison row, reusing eval_specificity.ROWS and its selected
(position, layer) per probe so this is consistent with what step 4 already
reported). Large rank disagreements in both directions are sampled, plus a
random comparison sample, per the plan.

Two-phase, blinded export -- run this twice:

  python audit_disagreements.py sample --run-name llama31_5k_subset
      Writes audit_blind.jsonl (id, user_prompt, response only -- no probe
      names, scores, ranks, or source-cell identity) and audit_key.json
      (id -> everything else, kept aside). Read audit_blind.jsonl only.

  python audit_disagreements.py report --run-name llama31_5k_subset \
      --labels audit_labels.jsonl
      Joins hand-assigned labels (from a labels file keyed by id, schema
      below) back against audit_key.json and reports behavior frequencies by
      sample reason (top A>B, top B>A, random), with representative excerpts.

Labels file: one json object per line, {"id": ..., "praise": bool,
"endorsement": bool, "omitted_correction": bool, "warmth": bool,
"hedging": bool, "warranted": true|false|"uncertain", "note": str}. Multiple
behavior labels may be true at once; "warranted" asks whether the labeled
behavior was actually supported by the user's context, not whether it looks
sycophantic on its face.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
import get_activations as ga  # noqa: E402
from analyze_probes import apply_probe, load_probes  # noqa: E402
from eval_specificity import ROWS, POLARITY_LABEL  # noqa: E402
from train_probes import DROP_REASONS  # noqa: E402

SEED = 0
N_PER_DIRECTION = 8  # top disagreements sampled in each direction, per pair
N_RANDOM = 8  # random comparison sample, per pair

# (pair name, probe A, probe B, step-4 row to draw the common example pool
# and each probe's selected (position, layer) from). Hedging is already
# covered in FINDINGS.md sec. 4 with text evidence, not repeated here.
AUDIT_PAIRS = [
    ("praise", "pt_explicit", "ctrl_warranted_praise", "unwarranted_praise"),
    ("agreement", "pv_explicit", "ctrl_genuine_agreement", "unsupported_vs_genuine_agreement"),
    ("emotional_validation", "pe_explicit", "ctrl_emotional_support", "inappropriate_validation_vs_support"),
]


def row_by_name(name: str) -> dict:
    return next(r for r in ROWS if r["name"] == name)


def load_generations(run_dir: Path, slug: str) -> dict:
    path = run_dir / "generations" / f"{slug}.jsonl"
    return {r["example_id"]: r for r in common.read_jsonl(path)}


def load_cell_pool(run_dir, test_prompt_ids, slug, position, layer, drop_degenerate, cache):
    """(X, y, example_ids) for ALL of this cell's held-out test-split rows,
    any polarity, at (position, layer)."""
    key = (slug, position, layer)
    if key not in cache:
        index = ga.load_index(run_dir, slug)
        X = ga.load_acts(run_dir, slug, position, layer)
        assert len(index) == X.shape[0]
        keep = np.ones(len(index), dtype=bool)
        if drop_degenerate:
            bad_prompts = {r["prompt_id"] for r in index if r["degenerate"] in DROP_REASONS}
            keep &= np.array([r["prompt_id"] not in bad_prompts for r in index], dtype=bool)
        keep &= np.array([r["prompt_id"] in test_prompt_ids for r in index], dtype=bool)
        rows = [r for r, k in zip(index, keep) if k]
        y = np.array([r["label"] for r in rows], dtype=int)
        eids = [r["example_id"] for r in rows]
        cache[key] = (X[keep], y, eids)
    return cache[key]


def percentile_rank(scores: np.ndarray) -> np.ndarray:
    """0-100, higher score -> higher percentile, ties averaged."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(len(scores))
    # average ranks for ties
    uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks = avg[inv]
    return 100.0 * ranks / (len(scores) - 1) if len(scores) > 1 else np.zeros(len(scores))


def build_pool(run_dir, row, probe_slug, position, layer, test_prompt_ids, drop_degenerate, cache):
    """Every (example_id, polarity_label) in this row's positive+negative
    sources, with this probe's score at its own (position, layer)."""
    probes = load_probes(run_dir, probe_slug)
    probe = probes[ga.act_key(position, layer)]
    example_ids, polarities, sources, scores = [], [], [], []
    for polarity_list, y_val in ((row["positive"], 1), (row["negative"], 0)):
        for src_slug, src_polarity in polarity_list:
            X, y, eids = load_cell_pool(run_dir, test_prompt_ids, src_slug, position, layer,
                                         drop_degenerate, cache)
            mask = y == POLARITY_LABEL[src_polarity]
            if mask.sum() == 0:
                continue
            s = apply_probe(probe, X[mask])
            sel_eids = [e for e, m in zip(eids, mask) if m]
            example_ids.extend(sel_eids)
            polarities.extend([y_val] * mask.sum())
            sources.extend([f"{src_slug}/{src_polarity}"] * mask.sum())
            scores.extend(s.tolist())
    return example_ids, np.array(polarities), sources, np.array(scores)


def cmd_sample(args):
    run_dir = common.resolve_run_dir(args.run_name, create=False)
    spec_path = run_dir / "analysis" / "specificity.json"
    if not spec_path.exists():
        raise SystemExit(f"run eval_specificity.py first: {spec_path} not found")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    picks = {}  # (row_name, slug) -> (position, layer)
    for row_report in spec["rows"]:
        for slug, e in row_report["probes"].items():
            picks[(row_report["name"], slug)] = (e["position"], e["layer"])

    split = json.loads((run_dir / "prompt_split.json").read_text(encoding="utf-8"))
    test_prompt_ids = set(split["test"])
    drop_degenerate = not args.keep_degenerate
    cache = {}
    rng = random.Random(SEED)

    blind, key = [], {}
    audit_id = 0

    for pair_name, slug_a, slug_b, row_name in AUDIT_PAIRS:
        row = row_by_name(row_name)
        pos_a, lay_a = picks[(row_name, slug_a)]
        pos_b, lay_b = picks[(row_name, slug_b)]
        # The pool spans every source cell in this row (e.g. "praise" draws
        # from pt_explicit, ctrl_obsequiousness, AND ctrl_warranted_praise),
        # not just slug_a -- merge every source's own generations file so
        # every sampled example_id resolves to its actual response text.
        row_slugs = {s for s, _ in row["positive"]} | {s for s, _ in row["negative"]}
        gens = {}
        for s in row_slugs:
            gens.update(load_generations(run_dir, s))

        eids_a, pol_a, src_a, scores_a = build_pool(run_dir, row, slug_a, pos_a, lay_a,
                                                      test_prompt_ids, drop_degenerate, cache)
        eids_b, pol_b, src_b, scores_b = build_pool(run_dir, row, slug_b, pos_b, lay_b,
                                                      test_prompt_ids, drop_degenerate, cache)
        # Common pool: same example_ids scored by both probes (both probes
        # see every source in this row, so the id sets should match exactly;
        # intersect defensively in case of a degenerate-drop mismatch).
        idx_a = {e: i for i, e in enumerate(eids_a)}
        idx_b = {e: i for i, e in enumerate(eids_b)}
        common_ids = [e for e in eids_a if e in idx_b]
        if len(common_ids) < len(eids_a):
            print(f"  [{pair_name}] {len(eids_a) - len(common_ids)} ids dropped (present for "
                  f"{slug_a} only, degenerate-filter mismatch)")

        pct_a_full = percentile_rank(scores_a)
        pct_b_full = percentile_rank(scores_b)
        pct_a = np.array([pct_a_full[idx_a[e]] for e in common_ids])
        pct_b = np.array([pct_b_full[idx_b[e]] for e in common_ids])
        diff = pct_a - pct_b  # positive: A reads this as more sycophantic than B does

        order_a_gt_b = np.argsort(-diff)[:N_PER_DIRECTION]
        order_b_gt_a = np.argsort(diff)[:N_PER_DIRECTION]
        remaining = [i for i in range(len(common_ids))
                     if i not in set(order_a_gt_b.tolist()) | set(order_b_gt_a.tolist())]
        random_idx = rng.sample(remaining, min(N_RANDOM, len(remaining)))

        for reason, idxs in (("A_higher", order_a_gt_b), ("B_higher", order_b_gt_a), ("random", random_idx)):
            for i in idxs:
                eid = common_ids[i]
                gen = gens.get(eid)
                if gen is None:
                    continue
                blind.append({
                    "id": audit_id,
                    "user_prompt": gen["user_prompt"],
                    "response": gen["response"],
                })
                key[str(audit_id)] = {
                    "pair": pair_name, "reason": reason, "example_id": eid,
                    "source": src_a[idx_a[eid]],
                    f"pct_{slug_a}": round(float(pct_a[i]), 1),
                    f"pct_{slug_b}": round(float(pct_b[i]), 1),
                    "diff_a_minus_b": round(float(diff[i]), 1),
                    "probe_a": slug_a, "probe_b": slug_b,
                }
                audit_id += 1
        print(f"[{pair_name}] pool={len(common_ids)}  sampled {len(order_a_gt_b)}+{len(order_b_gt_a)}+{len(random_idx)}")

    rng.shuffle(blind)  # sampling reason/pair order itself would leak information
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "audit_blind.jsonl").write_text(
        "\n".join(json.dumps(b, ensure_ascii=False) for b in blind), encoding="utf-8"
    )
    (out_dir / "audit_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"\n-> {out_dir / 'audit_blind.jsonl'} ({len(blind)} rows, shuffled)")
    print(f"-> {out_dir / 'audit_key.json'} (do not read before labeling)")


def cmd_report(args):
    run_dir = common.resolve_run_dir(args.run_name, create=False)
    out_dir = run_dir / "analysis"
    key = json.loads((out_dir / "audit_key.json").read_text(encoding="utf-8"))
    labels = {str(json.loads(l)["id"]): json.loads(l) for l in Path(args.labels).read_text(encoding="utf-8").splitlines() if l.strip()}

    missing = set(key) - set(labels)
    if missing:
        print(f"WARNING: {len(missing)} sampled ids have no label ({sorted(missing)[:10]}...)")

    behaviors = ("praise", "endorsement", "omitted_correction", "warmth", "hedging")
    for pair_name, slug_a, slug_b, row_name in AUDIT_PAIRS:
        print(f"\n=== {pair_name}  ({slug_a} vs {slug_b}) ===")
        for reason in ("A_higher", "B_higher", "random"):
            ids = [i for i, k in key.items() if k["pair"] == pair_name and k["reason"] == reason and i in labels]
            if not ids:
                continue
            n = len(ids)
            print(f"  {reason} (n={n}):")
            for b in behaviors:
                rate = sum(1 for i in ids if labels[i].get(b)) / n
                print(f"    {b:<20} {rate:.2f}")
            warranted_counts = {}
            for i in ids:
                w = labels[i].get("warranted")
                warranted_counts[str(w)] = warranted_counts.get(str(w), 0) + 1
            print(f"    warranted breakdown: {warranted_counts}")
            label_a = "A" if reason == "A_higher" else ("B" if reason == "B_higher" else None)
            if label_a:
                print(f"    (A_higher means {slug_a} scores this higher than {slug_b} does)")
            # one representative excerpt with a note, if any
            noted = [i for i in ids if labels[i].get("note")]
            if noted:
                i = noted[0]
                print(f"    excerpt (id {i}, {key[i]['source']}): {labels[i]['note'][:200]}")

    out_dir.joinpath("audit_report_raw.json").write_text(
        json.dumps({"key": key, "labels": labels}, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--run-name", type=str, required=True)
    p_sample.add_argument("--keep-degenerate", action="store_true")

    p_report = sub.add_parser("report")
    p_report.add_argument("--run-name", type=str, required=True)
    p_report.add_argument("--labels", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "sample":
        cmd_sample(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
