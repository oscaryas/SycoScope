#!/usr/bin/env python3
"""Plot sycophancy-rate bars with 95% Wilson confidence intervals."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SYCOPHANCY_DIR = Path(__file__).resolve().parents[2]
if str(SYCOPHANCY_DIR) not in sys.path:
    sys.path.insert(0, str(SYCOPHANCY_DIR))

from pipeline_scripts.common import json_dump, parse_dataset_spec, read_jsonl  # noqa: E402


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - margin, center + margin


def load_rate(path: Path) -> tuple[int, int]:
    if path.suffix == ".jsonl":
        values = [row["label"] for row in read_jsonl(path) if row.get("label") in (0, 1)]
        return sum(values), len(values)
    value = json.loads(path.read_text(encoding="utf-8"))
    n = int(value["n_judged"])
    return int(value.get("n_positive", round(float(value["rate"]) * n))), n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Sycophancy rate by dataset")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    records = []
    for raw in args.result:
        spec = parse_dataset_spec(raw)
        successes, n = load_rate(spec.path)
        low, high = wilson(successes, n)
        records.append({
            "name": spec.name, "successes": successes, "n": n,
            "rate": successes / n if n else 0.0, "ci95": [low, high],
        })
    fig, axis = plt.subplots(figsize=(max(6, 1.25 * len(records)), 4.5))
    rates = [record["rate"] for record in records]
    lows = [rate - record["ci95"][0] for rate, record in zip(rates, records)]
    highs = [record["ci95"][1] - rate for rate, record in zip(rates, records)]
    axis.bar(range(len(records)), rates, color="#4C78A8")
    axis.errorbar(range(len(records)), rates, yerr=[lows, highs], fmt="none", color="black", capsize=4)
    axis.set_xticks(range(len(records)), [record["name"] for record in records], rotation=25, ha="right")
    axis.set_ylabel("Sycophancy rate")
    axis.set_ylim(0, 1)
    axis.set_title(args.title)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    json_dump(output.with_suffix(".json"), records)


if __name__ == "__main__":
    main()
