#!/usr/bin/env python3
"""Plot residual-stream probe CV accuracy by layer with 95% t intervals."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.utils.probes_common import json_dump, parse_dataset_spec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, metavar="NAME=METRICS_JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Residual probe accuracy by layer")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    series = []
    for raw in args.result:
        spec = parse_dataset_spec(raw)
        value = json.loads(spec.path.read_text(encoding="utf-8"))
        layers = value["metrics"]["residual"]["layers"]
        points = []
        for layer, metrics in sorted(layers.items(), key=lambda item: int(item[0])):
            points.append({
                "layer": int(layer), "mean": metrics["cv_accuracy_mean"],
                "ci95": metrics["cv_accuracy_ci"],
            })
        series.append({"name": spec.name, "points": points})
    fig, axis = plt.subplots(figsize=(8, 4.8))
    width = 0.8 / len(series)
    for index, item in enumerate(series):
        x = [point["layer"] + (index - (len(series) - 1) / 2) * width for point in item["points"]]
        means = [point["mean"] for point in item["points"]]
        errors = [
            [point["mean"] - point["ci95"][0] for point in item["points"]],
            [point["ci95"][1] - point["mean"] for point in item["points"]],
        ]
        axis.bar(x, means, width=width, label=item["name"], alpha=0.85)
        axis.errorbar(x, means, yerr=errors, fmt="none", color="black", linewidth=0.7, capsize=1.5)
    axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axis.set_xlabel("Layer")
    all_layers = sorted({point["layer"] for item in series for point in item["points"]})
    stride = max(1, math.ceil(len(all_layers) / 12))
    axis.set_xticks(all_layers[::stride])
    axis.set_ylabel("Grouped CV accuracy")
    axis.set_ylim(0, 1)
    axis.set_title(args.title)
    if len(series) > 1:
        axis.legend(frameon=False)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    json_dump(output.with_suffix(".json"), series)


if __name__ == "__main__":
    main()
