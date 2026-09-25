#!/usr/bin/env python3
"""
Compare reproduced sycophancy probe results against paper-reported numbers.

Usage:
    python sycophancy_compare.py \
        --results metrics.json \
        --paper-results task_context/paper_results.json \
        --output comparison_table.md

paper_results.json is written by the agent after reading paper.pdf.
Expected format:
{
  "google/gemma-3-12b-it": {
    "mha_best_accuracy": 0.872,
    "mlp_best_accuracy": 0.841,
    "residual_best_accuracy": 0.855
  },
  "meta-llama/Llama-3.1-8B-Instruct": {
    "mha_best_accuracy": 0.863,
    "mlp_best_accuracy": 0.831,
    "residual_best_accuracy": 0.847
  }
}
"""
import argparse
import json
from pathlib import Path

PASS_THRESHOLD = 0.02   # absolute accuracy delta below which we call it "verified"
COMPONENTS = ["mha_best_accuracy", "mlp_best_accuracy", "residual_best_accuracy"]
COMPONENT_LABELS = {
    "mha_best_accuracy": "MHA best",
    "mlp_best_accuracy": "MLP best",
    "residual_best_accuracy": "Residual best",
}


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compare(reproduced: dict, paper: dict, model_name: str) -> list:
    """Return list of row dicts for the comparison table."""
    rows = []
    paper_model = paper.get(model_name, {})

    for comp in COMPONENTS:
        repr_val = reproduced.get(comp)
        paper_val = paper_model.get(comp)

        if repr_val is None:
            status = "MISSING"
            delta_str = "—"
        elif paper_val is None:
            status = "NO PAPER REF"
            delta_str = "—"
        else:
            delta = abs(repr_val - paper_val)
            delta_str = f"{delta:.3f}"
            status = "✓" if delta <= PASS_THRESHOLD else "DIFF"

        rows.append({
            "model": model_name,
            "component": COMPONENT_LABELS[comp],
            "paper_acc": f"{paper_val:.3f}" if paper_val is not None else "—",
            "reproduced_acc": f"{repr_val:.3f}" if repr_val is not None else "—",
            "delta": delta_str,
            "status": status,
        })
    return rows


def render_table(rows: list) -> str:
    header = "| Model | Component | Paper Acc | Reproduced Acc | Delta | Status |"
    sep    = "|-------|-----------|-----------|----------------|-------|--------|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['component']} | {r['paper_acc']} "
            f"| {r['reproduced_acc']} | {r['delta']} | {r['status']} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare probe results against paper")
    parser.add_argument("--results", required=True, help="Path to reproduced metrics.json")
    parser.add_argument("--paper-results", required=True, help="Path to paper_results.json")
    parser.add_argument("--model-name", default=None, help="Model name (inferred from metrics.json if omitted)")
    parser.add_argument("--output", default="comparison_table.md")
    args = parser.parse_args()

    reproduced = load_json(args.results)
    paper = load_json(args.paper_results)

    model_name = args.model_name or reproduced.get("model_name", "unknown")
    rows = compare(reproduced, paper, model_name)
    table = render_table(rows)

    print("\n=== Paper Comparison ===")
    print(table)

    verified = all(r["status"] == "✓" for r in rows if r["status"] not in ("MISSING", "NO PAPER REF"))
    print(f"\nOverall: {'VERIFIED ✓' if verified else 'DIFFERENCES FOUND — check comparison_table.md'}")

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        f.write(f"# Sycophancy Probe Comparison: {model_name}\n\n")
        f.write(f"> Pass threshold: Δ ≤ {PASS_THRESHOLD:.0%} absolute accuracy\n\n")
        f.write(table)
        f.write("\n\n")
        if verified:
            f.write("**Result: VERIFIED** — all reproduced values match paper within threshold.\n")
        else:
            f.write("**Result: DIFFERENCES FOUND** — see delta column for details.\n")

    print(f"Comparison table written to {out_path}")


if __name__ == "__main__":
    main()
