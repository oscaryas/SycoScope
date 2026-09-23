#!/usr/bin/env python3
"""Fixed-definition system-prompt paraphrase control on the Dolly arm.

This follow-up uses the exact 200 Dolly questions and 120/80 train/report split
from ``llama31_dataset_control_v1``.  The original responses and activations are
reused; only a definition-preserving paraphrase of each of the 20 positive and
negative system prompts is newly generated.  The analysis fits fresh probes on
both wordings and reports the fully crossed transfer design:

    original -> original       original -> paraphrase
    paraphrase -> original     paraphrase -> paraphrase

Because dataset, questions, split, model revision, and decoding configuration
are held fixed, loss of transfer on the off-diagonal wording arms is evidence
of lexical/template sensitivity.  Labels still identify inducing instructions,
not independently judged behavior.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
import dataset_control as dc  # noqa: E402
from analyze_probes import apply_probe  # noqa: E402
from train_probes import fit_probe, safe_auc  # noqa: E402


RUN = "llama31_prompt_wording_control_v1"
BASE_RUN = "llama31_dataset_control_v1"
PARAPHRASE_PATH = common.DATA_DIR / "sycophancy_probe_prompt_pairs_paraphrase_v1.json"
WORDINGS = ("original", "paraphrase")


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def lexical_jaccard(left: str, right: str) -> float:
    a, b = token_set(left), token_set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def validate_pairs(original: list[dict], paraphrase: list[dict]) -> list[dict]:
    """Require a one-to-one semantic-cell mapping and genuinely changed text."""
    if len(original) != len(paraphrase):
        raise ValueError("paraphrase set must contain exactly the original cells")
    rows = []
    for old, new in zip(original, paraphrase):
        identity = ("cell", "type", "slug", "pair_index")
        if any(old[key] != new[key] for key in identity):
            raise ValueError(f"cell identity/order changed: {old['cell']} vs {new['cell']}")
        stats = {"cell": old["cell"], "slug": old["slug"]}
        for polarity in common.POLARITIES:
            if dc.normalize(old[polarity]) == dc.normalize(new[polarity]):
                raise ValueError(f"unchanged paraphrase: {old['slug']} / {polarity}")
            stats[polarity + "_jaccard"] = lexical_jaccard(old[polarity], new[polarity])
            stats[polarity + "_old_chars"] = len(old[polarity])
            stats[polarity + "_new_chars"] = len(new[polarity])
        rows.append(stats)
    return rows


def prepare(run: Path, base_run: Path) -> None:
    base_manifest = base_run / "experiment.json"
    if not base_manifest.exists():
        raise FileNotFoundError(f"missing completed dataset-control manifest: {base_manifest}")
    base = json.loads(base_manifest.read_text(encoding="utf-8"))
    original = base["pairs"]
    paraphrase = common.load_prompt_pairs(PARAPHRASE_PATH)
    lexical = validate_pairs(original, paraphrase)
    prompts = [row for row in base["prompts"] if row["dataset"] == "dolly"]
    counts = Counter(row["split"] for row in prompts)
    if counts != {"train": 120, "report": 80}:
        raise ValueError(f"unexpected Dolly split in base experiment: {dict(counts)}")

    base_hash = dc.digest(base)
    experiment = {
        "model": base["model"],
        "model_revision": base["model_revision"],
        "seed": base["seed"],
        "dataset": "dolly",
        "prompts": prompts,
        "pairs": paraphrase,
        "layers": base["layers"],
        "generation": base["generation"],
        "analysis": base["analysis"],
        "base_run": str(base_run.resolve()),
        "base_experiment_hash": base_hash,
        "prompt_variant": "definition_preserving_paraphrase_v1",
        "prompt_source": str(PARAPHRASE_PATH.resolve()),
        "lexical_comparison": lexical,
        "n_generation_target": len(prompts) * len(paraphrase) * 2,
        "note": (
            "System-prompt wording control on the alternate Dolly dataset. The original arm is "
            "reused from the matched-backend base run; this run generates only paraphrases. "
            "Labels identify instructions, not independently judged behavior."
        ),
    }
    dc.save_new(run / "experiment.json", experiment)
    common.write_jsonl(run / "user_prompts.jsonl", prompts)
    summary = {
        "prepared": str(run),
        "dataset": "dolly",
        "split": dict(counts),
        "cells": len(paraphrase),
        "new_generations": experiment["n_generation_target"],
        "mean_prompt_jaccard": float(
            np.mean(
                [row[polarity + "_jaccard"] for row in lexical for polarity in common.POLARITIES]
            )
        ),
    }
    print(json.dumps(summary, indent=2))


def _load_cell(root: Path, slug: str, key: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    index = common.read_jsonl(root / "activations" / f"{slug}_index.jsonl")
    with np.load(root / "activations" / f"{slug}.npz") as arrays:
        X = arrays[key].astype(np.float32)
    if len(index) != len(X):
        raise ValueError(f"index/activation mismatch: {root.name}/{slug}/{key}")
    counts = Counter((row["dataset"], row["prompt_id"]) for row in index)
    paired = np.array([counts[row["dataset"], row["prompt_id"]] == 2 for row in index])
    dolly = np.array([row["dataset"] == "dolly" for row in index])
    keep = paired & dolly
    return X[keep], np.array([row["label"] for row in index])[keep], [
        row for row, selected in zip(index, keep) if selected
    ]


def _save_probe(path: Path, scaler, clf) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = clf.coef_[0] / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    direction = raw / (np.linalg.norm(raw) + 1e-12)
    np.savez_compressed(
        path,
        coef=clf.coef_[0],
        intercept=clf.intercept_[0],
        mean=scaler.mean_,
        scale=scaler.scale_,
        direction_raw=direction,
    )
    return {
        "coef": clf.coef_[0],
        "intercept": clf.intercept_[0],
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "direction_raw": direction,
    }


def _median(matrix: list[list[float]], diagonal: bool) -> float:
    values = np.asarray(matrix, dtype=float)
    mask = np.eye(values.shape[0], dtype=bool)
    return float(np.nanmedian(values[mask if diagonal else ~mask]))


def analyze(run: Path, base_run: Path) -> None:
    exp = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    base = json.loads((base_run / "experiment.json").read_text(encoding="utf-8"))
    if exp["base_experiment_hash"] != dc.digest(base):
        raise ValueError("base experiment changed after the wording control was prepared")
    names = [pair["slug"] for pair in exp["pairs"]]
    roots = {"original": base_run, "paraphrase": run}
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    transfers: dict[str, dict] = {}
    directions: dict[tuple[str, str, str], np.ndarray] = {}

    for layer in exp["layers"]:
        for position in common.POSITIONS:
            key = f"{position}_L{layer:02d}"
            print("Analysis", key, flush=True)
            heldout: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, list[dict]]] = {}
            probes = {}
            for wording in WORDINGS:
                for slug in names:
                    X, y, rows = _load_cell(roots[wording], slug, key)
                    train = np.array([row["split"] == "train" for row in rows])
                    report = np.array([row["split"] == "report" for row in rows])
                    if train.sum() < 200 or report.sum() < 100:
                        raise ValueError(f"insufficient paired examples: {wording}/{slug}/{key}")
                    scaler, clf = fit_probe(X[train], y[train], exp["seed"], 1.0, 2000)
                    if int(clf.n_iter_[0]) >= 2000:
                        raise ValueError(f"probe did not converge: {wording}/{slug}/{key}")
                    probe_path = run / "fitted_probes" / wording / slug / f"{key}.npz"
                    probe = _save_probe(probe_path, scaler, clf)
                    probes[wording, slug] = probe
                    directions[wording, slug, key] = probe["direction_raw"]
                    heldout[wording, slug] = (X[report], y[report], [
                        row for row, selected in zip(rows, report) if selected
                    ])

            for train_wording in WORDINGS:
                for test_wording in WORDINGS:
                    values, complete_values = [], []
                    for probe_slug in names:
                        row_values, clean_values = [], []
                        for target_slug in names:
                            X, y, rows = heldout[test_wording, target_slug]
                            scores = apply_probe(probes[train_wording, probe_slug], X)
                            incomplete = {
                                item["prompt_id"] for item in rows if item["finish_reason"] != "stop"
                            }
                            clean = np.array([item["prompt_id"] not in incomplete for item in rows])
                            row_values.append(safe_auc(y, scores))
                            clean_values.append(safe_auc(y[clean], scores[clean]))
                        values.append(row_values)
                        complete_values.append(clean_values)
                    result_key = f"{train_wording}_to_{test_wording}_{key}"
                    transfers[result_key] = {
                        "train_wording": train_wording,
                        "test_wording": test_wording,
                        "dataset": "dolly",
                        "position": position,
                        "layer": layer,
                        "cells": names,
                        "auc": values,
                        "complete_pairs_auc": complete_values,
                        "median_same_cell_auc": _median(values, diagonal=True),
                        "median_cross_cell_auc": _median(values, diagonal=False),
                    }
            (output / "transfer.json").write_text(json.dumps(transfers, indent=2) + "\n")

    cosines = []
    for layer in exp["layers"]:
        for position in common.POSITIONS:
            key = f"{position}_L{layer:02d}"
            for slug in names:
                left, right = directions["original", slug, key], directions["paraphrase", slug, key]
                cosines.append(
                    {
                        "slug": slug,
                        "position": position,
                        "layer": layer,
                        "cosine": float(np.dot(left, right)),
                    }
                )
    (output / "direction_cosines.json").write_text(json.dumps(cosines, indent=2) + "\n")

    lines = [
        "# System-prompt wording control",
        "",
        "Fresh probes on the Dolly dataset (120 training and 80 report questions), comparing the original 20 system-prompt pairs with definition-preserving paraphrases. AUROC labels identify the inducing instruction, not independently judged behavior.",
        "",
        "| Layer / response average | Original→original | Original→paraphrase | Paraphrase→original | Paraphrase→paraphrase |",
        "|---|---:|---:|---:|---:|",
    ]
    for layer in exp["layers"]:
        values = []
        for source, target in (
            ("original", "original"),
            ("original", "paraphrase"),
            ("paraphrase", "original"),
            ("paraphrase", "paraphrase"),
        ):
            result = transfers[f"{source}_to_{target}_response_L{layer:02d}"]
            values.append(f"{result['median_same_cell_auc']:.3f}")
        lines.append(f"| L{layer} | " + " | ".join(values) + " |")
    lines += [
        "",
        "The table reports the median diagonal AUROC: each probe is evaluated on the same semantic cell under the target wording. Full 20×20 matrices, off-diagonal medians, and complete-response-pair sensitivity are in `transfer.json`.",
        "",
        "Raw-space direction cosines between each original/paraphrase probe pair are in `direction_cosines.json`. High within-wording AUROC with weak cross-wording AUROC or low direction cosine indicates wording sensitivity rather than a stable cell-level direction.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    dc.save_new(
        run / "analysis_complete.json",
        {
            "experiment_hash": dc.digest(exp),
            "n_transfer_matrices": len(transfers),
            "n_direction_cosines": len(cosines),
        },
    )


def watch(run: Path, base_run: Path, host: str, port: str, identity: str) -> None:
    """Download completed shards, verify them, then run the CPU analysis."""
    experiment = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    names = [pair["slug"] for pair in experiment["pairs"]]
    ssh = [
        "ssh",
        "-i",
        identity,
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-p",
        port,
        host,
    ]
    scp = [
        "scp",
        "-i",
        identity,
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-P",
        port,
    ]
    remote = f"/workspace/syconbench_eval/prompt_probes/results/{run.name}"

    def command(shell_command: str) -> str:
        for attempt in range(5):
            try:
                result = subprocess.run(
                    ssh + [shell_command], check=True, text=True, stdout=subprocess.PIPE
                )
                return result.stdout.strip()
            except subprocess.CalledProcessError as exc:
                if exc.returncode != 255 or attempt == 4:
                    raise
                print("Transient SSH failure; retrying in 30 seconds.", flush=True)
                time.sleep(30)
        raise AssertionError("unreachable")

    pending = set(names)
    while pending:
        status = command("supervisorctl status prompt_wording_control_v1 || test $? -eq 3")
        print(status, flush=True)
        for name in names:
            if name not in pending:
                continue
            marker_path = f"{remote}/activations/{name}_complete.json"
            if command(f"test -f {marker_path} && echo yes || echo no") != "yes":
                continue
            relatives = [
                f"generations/{name}.jsonl",
                f"activations/{name}.npz",
                f"activations/{name}_index.jsonl",
                f"activations/{name}_skips.json",
                f"activations/{name}_complete.json",
            ]
            for relative in relatives:
                local = run / relative
                local.parent.mkdir(parents=True, exist_ok=True)
                if local.exists():
                    continue
                temporary = local.with_name(local.name + ".download")
                subprocess.run(scp + [f"{host}:{remote}/{relative}", str(temporary)], check=True)
                temporary.rename(local)
            marker = json.loads((run / f"activations/{name}_complete.json").read_text())
            cache = run / f"activations/{name}.npz"
            if hashlib.sha256(cache.read_bytes()).hexdigest() != marker["sha256"]:
                raise ValueError(f"download checksum mismatch: {name}")
            if dc.digest(common.read_jsonl(run / f"generations/{name}.jsonl")) != marker["generation_hash"]:
                raise ValueError(f"generation hash mismatch: {name}")
            index = common.read_jsonl(run / f"activations/{name}_index.jsonl")
            if len(index) != marker["n_rows"]:
                raise ValueError(f"index length mismatch: {name}")
            with np.load(cache) as arrays:
                if len(arrays.files) != len(experiment["layers"]) * len(common.POSITIONS):
                    raise ValueError(f"missing activation configurations: {name}")
                for key in arrays.files:
                    values = arrays[key]
                    if values.shape != (len(index), 4096) or not np.isfinite(values).all():
                        raise ValueError(f"invalid activation array: {name}/{key}")
            pending.remove(name)
            print("Downloaded and verified", name, flush=True)
        (run / "workflow_status.json").write_text(
            json.dumps(
                {
                    "stage": "generating_extracting" if pending else "analyzing",
                    "remaining_cells": sorted(pending),
                },
                indent=2,
            )
            + "\n"
        )
        if pending and not any(state in status for state in ("RUNNING", "STARTING")):
            raise RuntimeError(f"GPU job stopped with unfinished cells: {status}")
        if pending:
            time.sleep(45)
    analyze(run, base_run)
    (run / "workflow_status.json").write_text(
        json.dumps({"stage": "complete", "report": str(run / "analysis/RESULTS.md")}, indent=2)
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "gpu", "analyze", "watch"))
    parser.add_argument("--run-name", default=RUN)
    parser.add_argument("--base-run-name", default=BASE_RUN)
    parser.add_argument("--model-path", default="/workspace/syconbench_model")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--host", default="root@ssh4.vast.ai")
    parser.add_argument("--port", default="15883")
    parser.add_argument("--identity", default="/Users/oscar/.ssh/vast_vla")
    args = parser.parse_args()
    run = common.resolve_run_dir(args.run_name)
    base_run = common.resolve_run_dir(args.base_run_name, create=False)
    if args.stage == "prepare":
        prepare(run, base_run)
    elif args.stage == "gpu":
        dc.gpu(run, args.model_path, args.smoke)
    elif args.stage == "analyze":
        analyze(run, base_run)
    else:
        try:
            watch(run, base_run, args.host, args.port, args.identity)
        except Exception as exc:
            (run / "workflow_status.json").write_text(
                json.dumps({"stage": "error", "error": str(exc)}, indent=2) + "\n"
            )
            raise


if __name__ == "__main__":
    main()
