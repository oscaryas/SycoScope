"""
Incrementally extend an existing OEQ/SS/AITA-YTA social-sycophancy judged.jsonl
after --input-path (the generation checkpoint) has grown -- e.g. a 200-row
sample scaled up to 1200 rows. Only judges (row_id, prompt_col) pairs not
already present in --output-path, then appends them, instead of re-judging
(and re-paying for) every row the way run_social_sycophancy_judge_oeq.py's
default mode does.

Usage:
    python incremental_judge_social.py \
        --input-path .../oeq/checkpoint.jsonl \
        --output-path .../oeq/judged.jsonl
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYCOPHANCY_DIR = Path(__file__).resolve().parent.parent
if str(SYCOPHANCY_DIR) not in sys.path:
    sys.path.insert(0, str(SYCOPHANCY_DIR))

from social_sycophancy_judge import DEFAULT_MAX_WORKERS, JUDGE_MODEL, METRICS, iter_dataset_records  # noqa: E402

import anthropic  # noqa: E402
import httpx  # noqa: E402
import time  # noqa: E402

from social_sycophancy_judge import judge_metric  # noqa: E402

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0


def judge_metric_with_retry(client, prompt: str, response: str, metric: str, model: str):
    """Same retry contract as run_social_sycophancy_judge_oeq.py's version."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    args = parser.parse_args()

    already_judged = set()
    if args.output_path.exists():
        with open(args.output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                already_judged.add((str(row["row_id"]), row["prompt_col"]))
    print(f"{len(already_judged)} (row_id, prompt_col) pairs already judged in {args.output_path}", flush=True)

    all_records = iter_dataset_records(args.input_path)
    new_records = [r for r in all_records if (str(r["row_id"]), r["prompt_col"]) not in already_judged]
    print(f"{len(all_records)} total records in {args.input_path} -- {len(new_records)} new to judge.", flush=True)

    if not new_records:
        print("Nothing new to judge.")
    else:
        client = anthropic.Anthropic()
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        n_done = 0
        with open(args.output_path, "a", encoding="utf-8") as out_f, ThreadPoolExecutor(
            max_workers=args.max_workers
        ) as pool:
            for result in pool.map(
                lambda rec: judge_all_metrics_for_record(client, rec, args.judge_model),
                new_records,
            ):
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                n_done += 1
                if n_done % 100 == 0 or n_done == len(new_records):
                    print(f"  {n_done}/{len(new_records)} judged", flush=True)
        print(f"\nAppended {n_done} new records to {args.output_path}", flush=True)

    # Final summary over the FULL merged output file.
    rows = [json.loads(line) for line in open(args.output_path, encoding="utf-8")]
    print(f"\n=== Summary ({len(rows)} total judged rows) ===")
    for metric in METRICS:
        values = [r[metric] for r in rows]
        n_error = sum(v is None for v in values)
        n_valid = len(values) - n_error
        rate = sum(v == 1 for v in values) / n_valid if n_valid else 0.0
        print(f"{metric}: rate(label=1)={rate:.2%}  n_judged={n_valid}  n_parse_error={n_error}")


if __name__ == "__main__":
    main()
