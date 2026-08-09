"""
Standalone runner: judges every sample_idx=0 record in an OEQ-style jsonl
file (SAE/results/OEQ.jsonl by default) on all three social_sycophancy_judge
metrics (validation, indirectness, framing) and writes one row per record
with all three labels to a jsonl output.

This is judge-only -- unlike generate_social_sycophancy_labels() in
social_sycophancy_judge.py, it does not build the tokenizer-formatted "text"
field used for probe training, so it needs no HF tokenizer/model, no GPU.
Each judge call is an independent Claude API round-trip; concurrency is the
only thing that matters for wall-clock time.

Usage:
    python run_social_sycophancy_judge_oeq.py
    python run_social_sycophancy_judge_oeq.py --n-examples 100 --max-workers 24
    python run_social_sycophancy_judge_oeq.py --resume   # only re-judge rows with a None label
                                                          # in the existing --output-path (e.g.
                                                          # after an API billing/credits outage)
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import httpx

SYCOPHANCY_DIR = Path(__file__).resolve().parent.parent
if str(SYCOPHANCY_DIR) not in sys.path:
    sys.path.insert(0, str(SYCOPHANCY_DIR))

from social_sycophancy_judge import (  # noqa: E402
    DEFAULT_MAX_WORKERS,
    DEFAULT_RESULTS_DIR,
    JUDGE_MODEL,
    METRICS,
    iter_dataset_records,
    judge_metric,
)

DEFAULT_OUTPUT_PATH = SYCOPHANCY_DIR / "results" / "generations" / "OEQ_social_sycophancy_judged.jsonl"

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds; exponential backoff


def judge_metric_with_retry(client, prompt: str, response: str, metric: str, model: str):
    """judge_metric, but survives transient network errors instead of killing the whole batch.

    A single flaky connection (DNS blip, timeout, 5xx/429) inside
    ThreadPoolExecutor.map would otherwise propagate out of pool.map() and
    abort every in-flight and pending call. Retries with backoff on
    connection/rate-limit/server errors; returns None (same "skip" contract
    as judge_metric's own parse-failure case) once retries are exhausted.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return judge_metric(client, prompt, response, metric, model)
        except (anthropic.APIConnectionError, httpx.ConnectError, httpx.TimeoutException):
            pass
        except anthropic.APIStatusError as e:
            if e.status_code not in (429, 500, 502, 503, 529):
                return None
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BASE_DELAY * (2**attempt))
    return None


def judge_all_metrics_for_record(client, rec: dict, model: str, metrics=METRICS) -> dict:
    """Run the given metrics for one record; each label is 0/1 or None on parse/connection failure."""
    labels = {
        metric: judge_metric_with_retry(client, rec["prompt"], rec["response"], metric, model)
        for metric in metrics
    }
    return {
        "row_id": rec["row_id"],
        "prompt_col": rec["prompt_col"],
        "dataset": rec.get("dataset"),
        "prompt": rec["prompt"],
        "response": rec["response"],
        **labels,
    }


def resume_run(output_path: Path, judge_model: str, max_workers: int):
    """Re-judge only the metrics that are None in an existing output file, in place.

    Reuses prompt/response already stored in output_path (no need to re-read the
    original input jsonl) -- only re-runs the specific missing metric(s) per row,
    not all three, to avoid re-paying for labels that already succeeded.
    """
    if not output_path.exists():
        print(f"--resume given but {output_path} doesn't exist -- nothing to resume.", flush=True)
        return

    rows = [json.loads(line) for line in open(output_path, encoding="utf-8")]
    to_fix = [(i, row) for i, row in enumerate(rows) if any(row.get(m) is None for m in METRICS)]
    if not to_fix:
        print(f"No None labels in {output_path} -- nothing to resume.", flush=True)
        return

    print(f"Resuming: {len(to_fix)}/{len(rows)} rows have a missing label.", flush=True)
    client = anthropic.Anthropic()

    n_done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                judge_all_metrics_for_record,
                client,
                row,
                judge_model,
                metrics=[m for m in METRICS if row.get(m) is None],
            )
            for _, row in to_fix
        ]
        for (i, _), future in zip(to_fix, futures):
            result = future.result()
            rows[i].update({m: result[m] for m in METRICS if m in result})
            n_done += 1
            if n_done % 50 == 0 or n_done == len(to_fix):
                print(f"  {n_done}/{len(to_fix)} re-judged", flush=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for row in rows:
            out_f.write(json.dumps(row) + "\n")

    still_null = sum(1 for row in rows for m in METRICS if row.get(m) is None)
    print(f"\nUpdated {output_path}. Remaining None labels across all metrics: {still_null}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_RESULTS_DIR / "OEQ.jsonl")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n-examples", type=int, default=None, help="Limit to first N records (default: all)")
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Only re-judge rows/metrics that are None in an existing --output-path, instead of a full run.",
    )
    args = parser.parse_args()

    if args.resume:
        resume_run(args.output_path, args.judge_model, args.max_workers)
        return

    client = anthropic.Anthropic()
    dataset_records = iter_dataset_records(args.input_path, args.n_examples)
    print(f"Loaded {len(dataset_records)} records from {args.input_path}", flush=True)
    print(f"Judging metrics {METRICS} with model={args.judge_model}, max_workers={args.max_workers}", flush=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    with open(args.output_path, "w", encoding="utf-8") as out_f, ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as pool:
        for result in pool.map(
            lambda rec: judge_all_metrics_for_record(client, rec, args.judge_model),
            dataset_records,
        ):
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            n_done += 1
            if n_done % 100 == 0 or n_done == len(dataset_records):
                print(f"  {n_done}/{len(dataset_records)} judged", flush=True)

    print(f"\nWrote {n_done} records to {args.output_path}", flush=True)

    # Summary: label=1 rate and parse-failure rate per metric
    rows = [json.loads(line) for line in open(args.output_path, encoding="utf-8")]
    print("\n=== Summary ===")
    for metric in METRICS:
        values = [r[metric] for r in rows]
        n_error = sum(v is None for v in values)
        n_valid = len(values) - n_error
        rate = sum(v == 1 for v in values) / n_valid if n_valid else 0.0
        print(f"{metric}: rate(label=1)={rate:.2%}  n_judged={n_valid}  n_parse_error={n_error}")


if __name__ == "__main__":
    main()
