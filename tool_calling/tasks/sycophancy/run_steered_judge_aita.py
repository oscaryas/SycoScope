"""
Judge the AITA-YTA steered generations on all three social_sycophancy_judge
metrics, for every (run, vector) cell under SAE/results/steered_responses/.

Thin driver over run_social_sycophancy_judge_oeq.py -- that script is already
dataset-agnostic (it takes --input-path/--output-path despite the "_oeq" name),
so the judging, retry/backoff, and resume logic are reused verbatim here. All
this adds is the loop over cells, provenance fields on each row, and
skip-if-already-done.

Default cells: coeff_0 (the no-op control, 6 vectors) x coeff_2 (multiplier 2.0,
same 6 vectors) = 12 cells x 300 prompts x 3 metrics = 10,800 judge calls. That
is real money, so cells with an existing output file are skipped rather than
re-judged; use --force to override, or --resume to only re-run labels that came
back None (e.g. after an API outage).

Usage:
    python run_steered_judge_aita.py --limit 20          # smoke test
    python run_steered_judge_aita.py                     # full run
    python run_steered_judge_aita.py --resume            # fill in None labels
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

SYCOPHANCY_DIR = Path(__file__).resolve().parent
if str(SYCOPHANCY_DIR) not in sys.path:
    sys.path.insert(0, str(SYCOPHANCY_DIR))

from run_social_sycophancy_judge_oeq import (  # noqa: E402
    judge_all_metrics_for_record,
    resume_run,
)
from social_sycophancy_judge import (  # noqa: E402
    DEFAULT_MAX_WORKERS,
    JUDGE_MODEL,
    METRICS,
    iter_dataset_records,
)

REPO_ROOT = SYCOPHANCY_DIR.parents[2]
DEFAULT_STEERED_DIR = REPO_ROOT / "SAE" / "results" / "steered_responses"
DEFAULT_OUTPUT_DIR = SYCOPHANCY_DIR / "results" / "steered_judged"
DEFAULT_RUNS = ("coeff_0", "coeff_2")
DATASET = "AITA-YTA"


def discover_cells(steered_dir: Path, runs) -> list[dict]:
    """One dict per (run, vector) cell that actually has a DATASET jsonl on disk."""
    cells = []
    for run in runs:
        run_dir = steered_dir / run
        if not run_dir.is_dir():
            raise FileNotFoundError(f"No such run directory: {run_dir}")
        vector_dirs = sorted(
            d for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
        for vector_dir in vector_dirs:
            input_path = vector_dir / f"{DATASET}.jsonl"
            if not input_path.exists() or input_path.stat().st_size == 0:
                print(f"  [skip] {run}/{vector_dir.name}: no {DATASET}.jsonl", flush=True)
                continue
            # coeff comes from the cell's own config, not the sweep config -- the two can
            # diverge if a cell was re-run, and the output should record what was actually used.
            config = json.loads((vector_dir / "steering_config.json").read_text(encoding="utf-8"))
            cells.append(
                {
                    "run": run,
                    "vector": vector_dir.name,
                    "coeff": config["coeff"],
                    "input_path": input_path,
                }
            )
    return cells


def judge_cell(cell: dict, output_path: Path, n_examples, judge_model: str, max_workers: int) -> dict:
    """Judge one cell's records on all three metrics; write one row per record."""
    client = anthropic.Anthropic()
    records = iter_dataset_records(cell["input_path"], n_examples)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    with open(output_path, "w", encoding="utf-8") as out_f, ThreadPoolExecutor(
        max_workers=max_workers
    ) as pool:
        for result in pool.map(
            lambda rec: judge_all_metrics_for_record(client, rec, judge_model), records
        ):
            # Provenance so the 12 output files merge into one frame without re-deriving
            # which cell a row came from.
            result.update(run=cell["run"], vector=cell["vector"], coeff=cell["coeff"])
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            n_done += 1
            if n_done % 100 == 0 or n_done == len(records):
                print(f"    {n_done}/{len(records)} judged", flush=True)

    return summarize(output_path)


def summarize(output_path: Path) -> dict:
    rows = [json.loads(line) for line in open(output_path, encoding="utf-8") if line.strip()]
    summary = {"n_rows": len(rows)}
    for metric in METRICS:
        values = [r.get(metric) for r in rows]
        n_error = sum(v is None for v in values)
        n_valid = len(values) - n_error
        summary[metric] = {
            "rate": sum(v == 1 for v in values) / n_valid if n_valid else None,
            "n_judged": n_valid,
            "n_parse_error": n_error,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steered-dir", type=Path, default=DEFAULT_STEERED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--limit", type=int, default=None, help="Records per cell (default: all)")
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--force", action="store_true", help="Re-judge cells that already have an output file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Only re-judge None labels in existing outputs; judge nothing new.",
    )
    args = parser.parse_args()

    cells = discover_cells(args.steered_dir, args.runs)
    n_calls = len(cells) * (args.limit or 300) * len(METRICS)
    print(f"{len(cells)} cells, ~{n_calls} judge calls, model={args.judge_model}\n", flush=True)

    summaries = {}
    for i, cell in enumerate(cells, 1):
        output_path = (
            args.output_dir
            / cell["run"]
            / cell["vector"]
            / f"{DATASET}_social_sycophancy_judged.jsonl"
        )
        label = f"[{i}/{len(cells)}] {cell['run']}/{cell['vector']} (coeff={cell['coeff']:.3f})"

        if args.resume:
            if output_path.exists():
                print(label, flush=True)
                resume_run(output_path, args.judge_model, args.max_workers)
                summaries[f"{cell['run']}/{cell['vector']}"] = summarize(output_path)
            else:
                print(f"{label}: no output to resume", flush=True)
            continue

        if output_path.exists() and not args.force:
            print(f"{label}: already judged, skipping (--force to redo)", flush=True)
            summaries[f"{cell['run']}/{cell['vector']}"] = summarize(output_path)
            continue

        print(label, flush=True)
        summaries[f"{cell['run']}/{cell['vector']}"] = judge_cell(
            cell, output_path, args.limit, args.judge_model, args.max_workers
        )

    print("\n=== Summary (rate of label=1) ===")
    header = f"{'cell':52} " + " ".join(f"{m[:6]:>7}" for m in METRICS) + f" {'n':>5} {'err':>4}"
    print(header)
    for name, s in summaries.items():
        rates = " ".join(
            f"{s[m]['rate']:7.3f}" if s[m]["rate"] is not None else f"{'--':>7}" for m in METRICS
        )
        n_err = sum(s[m]["n_parse_error"] for m in METRICS)
        print(f"{name:52} {rates} {s['n_rows']:5d} {n_err:4d}")

    total_err = sum(s[m]["n_parse_error"] for s in summaries.values() for m in METRICS)
    if total_err:
        print(f"\n{total_err} parse errors -- re-run with --resume to fill them in.")


if __name__ == "__main__":
    main()
