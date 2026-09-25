#!/usr/bin/env python3
"""
V1 analysis plan, step 2: the probe-by-evaluation-target performance matrix.

Reads an activation cache already built by scorer/eval_sypr.py /
eval_moral.py / eval_social_sycophancy.py / eval_syconbench.py (one .npz per
position, see eval_common) and pickled probes -- no GPU, no model weights,
nothing re-extracted. --target picks the label/pair-field spec. For each
label_field:

  1. Select (position, layer) per probe on the *selection* split (mean AUC),
     never on the eval split.
  2. Score every probe on the *eval* split at its selected (position, layer).
  3. Where the label_field is in PAIR_FIELDS (a pair-level label, e.g. AITA
     both_nta -- the SAME label covers both sides of a conflict), group-average
     scores within each pair before scoring, same as eval_common.auc_rows().
  4. Bootstrap the eval-half AUC by resampling *groups* (the same key used at
     extraction time: group_id if the record has one, else example_id) with
     replacement, so related rows always move together.
  5. Report each taxonomy probe's paired bootstrap AUC delta against
     a probe named general_baseline (if given) and against the "best single
     probe" (also selected on the selection split, across all given probes).

None of the three current eval targets (SyPR, AITA, ELEPHANT) has matched
positive/negative *responses* to the same scenario -- that contrastive-pair
structure exists only in the probe training data, where transfer_matrix() in
misc/score_probes_legacy.py reports paired win rate. If a future eval target
does have matched pairs, that metric belongs here as a new function, not
retrofitted onto PAIR_FIELDS (which here means "shared pair-level label").

This is a synthesis pass over already-computed per-row scores; it does not
change any evaluation logic in eval_common.py.

Usage:
    python -m scorer.eval_matrix --target moral --cache results/probes/scores/moral.npz \
        --probe pv_implicit=pv.pkl general_baseline=gb.pkl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import common  # noqa: E402
from analyze_probes.probes_core import safe_auc  # noqa: E402
from scorer.eval_common import (  # noqa: E402
    cache_paths, group_mean, load_table, resolve_label_fields, selection_split, split_key,
)
from scorer.score_probes import SCORES_DIR, apply_probe, load_probe_set, select_probe  # noqa: E402

TARGETS = {
    "sypr": {"label_fields": ("sycophantic_praise",), "pair_fields": ()},
    "moral": {
        "label_fields": ("nta_when_yta", "unwarranted_nta", "both_nta"),
        "pair_fields": ("both_nta",),
    },
    "elephant": {
        "label_fields": None,  # discovered from the cache's label__* arrays
        "pair_fields": (),
    },
    "syconbench": {
        "label_fields": tuple(
            f"{setting}_{target}"
            for setting in ("debate", "ethical", "false_presupposition")
            for target in ("failure", "first_failure")
        ),
        "pair_fields": (),  # labels vary by turn; never average them per conversation
    },
}

N_BOOT = 2000
SEED = 0

# Rows always shown when present, in this order, before any remaining taxonomy cells.
HEADLINE_SLUGS = (
    "pv_explicit", "pv_implicit", "ps_explicit", "ps_implicit",
    "pt_explicit", "pt_implicit", "pe_explicit", "pe_implicit",
    "general_baseline",
    "ctrl_pure_sycophancy", "ctrl_obsequiousness", "ctrl_excitement",
    "ctrl_affiliation", "ctrl_inauthenticity", "ctrl_politeness",
    "ctrl_warranted_praise", "ctrl_genuine_agreement",
    "ctrl_emotional_support", "ctrl_evasive_hedging", "ctrl_calibrated_hedging",
)


def bootstrap_auc(scores, y, group_ids, n_boot=N_BOOT, seed=SEED):
    """AUC + percentile CI, resampled at the group level (with replacement)."""
    rng = np.random.default_rng(seed)
    groups = np.array(group_ids)
    uniq = np.unique(groups)
    by_group = {g: np.where(groups == g)[0] for g in uniq}
    point = safe_auc(y, scores)
    if point is None or len(uniq) < 4:
        return {"auc": point, "ci_lo": None, "ci_hi": None, "n_boot": 0}
    boots = []
    for _ in range(n_boot):
        sample_groups = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_group[g] for g in sample_groups])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        a = safe_auc(yy, scores[idx])
        if a is not None:
            boots.append(a)
    if len(boots) < 50:
        return {"auc": point, "ci_lo": None, "ci_hi": None, "n_boot": len(boots)}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auc": point, "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": len(boots)}


def bootstrap_delta(scores_a, scores_b, y, group_ids, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap CI for AUC(a) - AUC(b) on the same rows/labels."""
    rng = np.random.default_rng(seed)
    groups = np.array(group_ids)
    uniq = np.unique(groups)
    by_group = {g: np.where(groups == g)[0] for g in uniq}
    pa, pb = safe_auc(y, scores_a), safe_auc(y, scores_b)
    if pa is None or pb is None or len(uniq) < 4:
        return {"delta": None, "ci_lo": None, "ci_hi": None, "n_boot": 0}
    point = pa - pb
    boots = []
    for _ in range(n_boot):
        sample_groups = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_group[g] for g in sample_groups])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        aa, bb = safe_auc(yy, scores_a[idx]), safe_auc(yy, scores_b[idx])
        if aa is not None and bb is not None:
            boots.append(aa - bb)
    if len(boots) < 50:
        return {"delta": point, "ci_lo": None, "ci_hi": None, "n_boot": len(boots)}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": point, "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": len(boots)}


def select_position_layer(tables, selection, payload, label_field, group_field: bool = False, C=None):
    """Best (position, layer) for this probe on the selection split, by AUC.
    Never touches the eval split. For a pair-level label (group_field=True,
    e.g. AITA both_nta), scores are averaged within each group before AUC --
    that label describes the pair, not either individual response, matching
    eval_common.auc_rows()."""
    best = None
    for table in tables:
        is_sel = np.array([split_key(r) in selection for r in table["records"]], dtype=bool)
        y_all = table["labels"][label_field]
        mask_sel = is_sel & (y_all >= 0)
        if mask_sel.sum() == 0 or len(np.unique(y_all[mask_sel])) < 2:
            continue
        for layer in table["layers"]:
            probe = select_probe(payload, layer, C)
            if probe is None:
                continue
            s = apply_probe(probe, table["acts"][layer][mask_sel])
            yy = y_all[mask_sel]
            if group_field:
                s, yy = group_mean(s, yy, table["groups"][mask_sel])
            a = safe_auc(yy, s)
            if a is not None and (best is None or a > best[0]):
                best = (a, table["position"], int(layer))
    if best is None:
        return None
    return {"selection_auc": best[0], "position": best[1], "layer": best[2]}


def run_target(probes: dict, paths: dict, target_key: str, n_boot: int, out_path: Path, C=None):
    spec = TARGETS[target_key]
    tables = [load_table(path, position) for position, path in paths.items()]
    if len({len(t["records"]) for t in tables}) != 1:
        raise SystemExit("position caches disagree on row count")
    label_fields = resolve_label_fields(tables, spec["label_fields"] or ())
    pair_fields = spec["pair_fields"]
    by_position = {t["position"]: t for t in tables}

    records = tables[0]["records"]
    split_spec = tables[0]["meta"].get("evaluation_selection", {"frac": 0.3, "seed": SEED})
    selection = selection_split(records, split_spec["frac"], split_spec["seed"])
    group_ids_all = np.array([split_key(r) for r in records])
    is_sel = np.array([split_key(r) in selection for r in records], dtype=bool)

    report = {"target": target_key, "caches": {t["position"]: str(t["path"]) for t in tables},
              "probes": sorted(probes), "n_boot": n_boot, "fields": {}}

    for field in label_fields:
        y_all = tables[0]["labels"][field]
        eval_mask = (~is_sel) & (y_all >= 0)
        if eval_mask.sum() == 0 or len(np.unique(y_all[eval_mask])) < 2:
            print(f"  {field}: no usable eval-split labels, skipping")
            continue
        group_field = field in pair_fields
        y_eval_raw = y_all[eval_mask]
        groups_eval_raw = group_ids_all[eval_mask]

        picks, eval_scores = {}, {}
        for slug, payload in probes.items():
            pick = select_position_layer(tables, selection, payload, field, group_field=group_field, C=C)
            if pick is None:
                continue
            probe = select_probe(payload, pick["layer"], C)
            s = apply_probe(probe, by_position[pick["position"]]["acts"][pick["layer"]][eval_mask])
            if group_field:
                # both_nta etc: the label describes the pair, not either response
                # individually -- average within each group before scoring, same
                # as eval_common.auc_rows(). y is identical within a group so the
                # label survives the average unchanged.
                s, _ = group_mean(s, y_eval_raw, groups_eval_raw)
            picks[slug] = pick
            eval_scores[slug] = s

        if group_field:
            y_eval, _ = group_mean(y_eval_raw, y_eval_raw, groups_eval_raw)
            y_eval = y_eval.round().astype(int)
            # Already one row per group (pair) after averaging -- an ordinary
            # per-row bootstrap over these rows already resamples at the
            # scenario level, since each row now *is* one scenario/pair.
        else:
            y_eval = y_eval_raw
        # Turn-level labels can vary within a conversation. Preserve those groups
        # for bootstrap resampling without averaging away turn-specific outcomes.
        groups_eval = np.arange(len(y_eval)) if group_field else groups_eval_raw

        if not eval_scores:
            print(f"  {field}: no probes scoreable")
            continue

        # Best single probe selected on the selection split, across all probes.
        best_slug = max(picks, key=lambda s: picks[s]["selection_auc"])

        field_report = {"n_eval": int(eval_mask.sum()), "n_eval_groups": int(len(np.unique(groups_eval_raw))),
                         "n_pos": int((y_eval == 1).sum()),
                         "n_neg": int((y_eval == 0).sum()), "best_single_probe": best_slug,
                         "probes": {}}

        gb_scores = eval_scores.get("general_baseline")
        for slug in sorted(eval_scores, key=lambda s: HEADLINE_SLUGS.index(s) if s in HEADLINE_SLUGS else 999):
            s = eval_scores[slug]
            boot = bootstrap_auc(s, y_eval, groups_eval, n_boot=n_boot)
            entry = {"position": picks[slug]["position"], "layer": picks[slug]["layer"],
                     "selection_auc": round(picks[slug]["selection_auc"], 4),
                     "accuracy": float(((s > 0).astype(int) == y_eval).mean()), **boot}
            if gb_scores is not None and slug != "general_baseline":
                entry["vs_general_baseline"] = bootstrap_delta(s, gb_scores, y_eval, groups_eval, n_boot=n_boot)
            if slug != best_slug:
                entry["vs_best_single_probe"] = bootstrap_delta(s, eval_scores[best_slug], y_eval, groups_eval, n_boot=n_boot)
            field_report["probes"][slug] = entry

        if group_field:
            # both_nta pairs (original_post vs flipped_story) share one label per
            # pair rather than being matched positive/negative responses to the
            # same scenario, so a within-pair win rate isn't defined here -- see
            # eval_moral.py's PAIR_FIELDS docstring. Note it rather than silently
            # report nothing.
            field_report["note"] = (
                "pair-level label (both_nta): scores were group-averaged (original_post, "
                "flipped_story) before AUC/bootstrap, per eval_common.auc_rows(); no matched "
                "positive/negative response pair exists here, so paired win rate is not reported."
            )

        report["fields"][field] = field_report
        print(f"\n  === {target_key}.{field}  (n_eval={field_report['n_eval']}, "
              f"pos/neg {field_report['n_pos']}/{field_report['n_neg']}, best_single={best_slug}) ===")
        for slug in sorted(field_report["probes"], key=lambda s: -field_report["probes"][s]["auc"]):
            e = field_report["probes"][slug]
            ci = f"[{e['ci_lo']:.3f},{e['ci_hi']:.3f}]" if e["ci_lo"] is not None else "[n/a]"
            print(f"    {slug:<26} AUC {e['auc']:.3f} {ci}  @ {e['position']}_L{e['layer']:02d}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", nargs="+", required=True, help="Pickled probes: path or name=path.")
    parser.add_argument("--cache", type=Path, required=True, help="The eval_*.py cache for --target.")
    parser.add_argument("--positions", type=str, nargs="+", default=["response"], choices=common.POSITIONS)
    parser.add_argument("--target", type=str, required=True, choices=list(TARGETS))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--C", type=float, default=None, help="Default: each layer's best_C.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.output or SCORES_DIR / "matrix" / f"matrix_{args.target}.json"
    run_target(load_probe_set(args.probe), cache_paths(args.cache, args.positions), args.target, args.n_boot,
               out_path, args.C)


if __name__ == "__main__":
    main()
