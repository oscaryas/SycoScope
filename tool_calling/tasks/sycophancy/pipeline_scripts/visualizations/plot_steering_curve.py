#!/usr/bin/env python3
"""Plot baseline and nonzero-alpha steering sycophancy-rate curves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Steering summary.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Steering response curve")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    values = json.loads(Path(args.input).read_text(encoding="utf-8"))
    fig, axis = plt.subplots(figsize=(7, 4.5))
    source = []
    for name, result in values.items():
        points = sorted(
            ({"alpha": float(alpha), "rate": summary["rate"]} for alpha, summary in result["alphas"].items()),
            key=lambda point: point["alpha"],
        )
        axis.plot([p["alpha"] for p in points], [p["rate"] for p in points], marker="o", label=name)
        axis.axhline(result["baseline"]["rate"], linestyle="--", linewidth=1, alpha=0.6)
        source.append({"name": name, "baseline_rate": result["baseline"]["rate"], "points": points})
    axis.set_xlabel("Steering alpha (0 is saved baseline, not regenerated)")
    axis.set_ylabel("Sycophancy rate")
    axis.set_ylim(0, 1)
    axis.set_title(args.title)
    if len(values) > 1:
        axis.legend(frameon=False)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps(source, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
