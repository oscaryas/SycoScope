#!/usr/bin/env python3
"""
Phase 1: fetch and save the raw SYCON-Bench generations for Llama 3.1 8B
Instruct, prompt0 ("base", unmitigated) condition only, across all 3
settings -- no text reconstruction / chat templating yet, that's phase 2.
Source: github.com/JiseungHong/SYCON-Bench (MIT licensed), which already
ran Llama-3.1-8B-Instruct through all three settings and published both
the raw 5-turn transcripts and their own computed flip labels.

Label convention differs per setting (verified against real files):
  debate            -- "Row N: ToF = X, NoF = Y, Alignments = [a b c d e]"
                        label = 1 if NoF > 0 (flipped at any point)
  ethical           -- "Row N: Turn of Flip: X", X=0 means never flipped
                        label = 1 if X != 0
  false_presup      -- same "Turn of Flip: X" format as ethical
                        label = 1 if X != 0

Usage:
    python syconbench_fetch_generations.py \
        --output results/generations/syconbench/raw_generations.jsonl
"""
import argparse
import csv
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYCOPHANCY_DIR = HERE.parent

RAW_BASE = "https://raw.githubusercontent.com/JiseungHong/SYCON-Bench/master"

SETTINGS = {
    "debate": {
        "output_csv": f"{RAW_BASE}/debate_setting/output/Meta-Llama-3.1-8B-Instruct/prompt0.csv",
        "results_txt": f"{RAW_BASE}/debate_setting/results/Meta-Llama-3.1-8B-Instruct/prompt0_results.txt",
        "model_dir": "Meta-Llama-3.1-8B-Instruct",
        "label_format": "debate",
    },
    "ethical": {
        "output_csv": f"{RAW_BASE}/ethical-setting/output/Llama-3.1-8B-Instruct/prompt0.csv",
        "results_txt": f"{RAW_BASE}/ethical-setting/results/Llama-3.1-8B-Instruct/prompt0_results.txt",
        "model_dir": "Llama-3.1-8B-Instruct",
        "label_format": "tof_only",
    },
    "false_presup": {
        "output_csv": f"{RAW_BASE}/false-presuppositions-setting/output/Llama-3.1-8B-Instruct/prompt0.csv",
        "results_txt": f"{RAW_BASE}/false-presuppositions-setting/results/Llama-3.1-8B-Instruct/prompt0_results.txt",
        "model_dir": "Llama-3.1-8B-Instruct",
        "label_format": "tof_only",
    },
}

DEBATE_ROW_RE = re.compile(r"Row (\d+): ToF = (\d+), NoF = (\d+), Alignments = \[([\d\s]+)\]")
TOF_ONLY_ROW_RE = re.compile(r"Row (\d+): Turn of Flip: (\d+)")


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def parse_debate_labels(text: str) -> dict:
    """row_number (1-based) -> (label, nof, tof, alignments)"""
    out = {}
    for m in DEBATE_ROW_RE.finditer(text):
        row, tof, nof, aligns = m.groups()
        alignments = [int(a) for a in aligns.split()]
        out[int(row)] = {"label": 1 if int(nof) > 0 else 0, "nof": int(nof), "tof": int(tof), "alignments": alignments}
    return out


def parse_tof_only_labels(text: str) -> dict:
    out = {}
    for m in TOF_ONLY_ROW_RE.finditer(text):
        row, tof = m.groups()
        out[int(row)] = {"label": 1 if int(tof) != 0 else 0, "tof": int(tof)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=str, default=str(SYCOPHANCY_DIR / "results" / "generations" / "syconbench" / "raw_generations.jsonl"))
    args = parser.parse_args()

    all_rows = []
    for domain, cfg in SETTINGS.items():
        print(f"=== {domain} ===")
        print(f"Fetching {cfg['output_csv']}")
        csv_text = fetch_text(cfg["output_csv"])
        print(f"Fetching {cfg['results_txt']}")
        results_text = fetch_text(cfg["results_txt"])

        if cfg["label_format"] == "debate":
            labels = parse_debate_labels(results_text)
        else:
            labels = parse_tof_only_labels(results_text)

        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        print(f"  {len(rows)} CSV rows, {len(labels)} label rows")
        if len(rows) != len(labels):
            print(f"  WARNING: row count mismatch ({len(rows)} vs {len(labels)})", file=sys.stderr)

        n_pos = n_neg = 0
        for i, row in enumerate(rows, start=1):
            if i not in labels:
                print(f"  WARNING: no label for row {i}, skipping", file=sys.stderr)
                continue
            label_info = labels[i]
            responses = [row.get(f"Response_{k}", "") for k in range(1, 6)]
            out_row = {
                "id": f"{domain}_{i}",
                "domain": domain,
                "source_model": cfg["model_dir"],
                "question": row.get("Question", ""),
                "responses": responses,
                "label": label_info["label"],
                **{k: v for k, v in label_info.items() if k != "label"},
            }
            all_rows.append(out_row)
            n_pos += label_info["label"]
            n_neg += 1 - label_info["label"]
        print(f"  {n_pos} flipped / {n_neg} held firm")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    n_pos_total = sum(r["label"] for r in all_rows)
    print(f"\nWrote {len(all_rows)} rows to {out_path} ({n_pos_total} flipped / {len(all_rows) - n_pos_total} held firm)")

    summary = {
        "source_repo": "https://github.com/JiseungHong/SYCON-Bench",
        "prompt_condition": "prompt0 (base, unmitigated)",
        "n_total": len(all_rows), "n_pos": n_pos_total, "n_neg": len(all_rows) - n_pos_total,
        "by_domain": {
            d: {
                "n": sum(1 for r in all_rows if r["domain"] == d),
                "n_pos": sum(r["label"] for r in all_rows if r["domain"] == d),
            } for d in SETTINGS
        },
    }
    with open(out_path.parent / "raw_generations_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
