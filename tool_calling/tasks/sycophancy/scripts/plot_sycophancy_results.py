"""
Builds a summary bar chart + aggregate JSON from the judged result jsonl
files in results/ (produced by run_social_sycophancy_judge_oeq.py and
run_moral_sycophancy_judge_aita.py). Re-derives every number straight from
the raw per-record jsonl files rather than hardcoding them, so it stays
correct if a file is re-judged/resumed.

Three panels, since the underlying metrics measure different things and
mixing them on one axis would be an apples-to-oranges comparison:
  A. Social sycophancy (validation/indirectness/framing rate) -- OEQ, SS,
     AITA-YTA, all judged with the same ELEPHANT sycophancy_scorers.py judge.
  B. Moral sycophancy (AITA-NTA-FLIP) -- paired-verdict outcome breakdown,
     ELEPHANT's own compute_rates() categories.
  C. Single-response verdict distribution (AITA-NTA-OG, AITA-YTA) -- not a
     metric from the paper, kept as an exploratory comparison; consistent
     NTA/YTA/OTHER color mapping with panel B for reading continuity.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
GENERATIONS_DIR = RESULTS_DIR / "generations"

# Okabe-Ito colorblind-safe categorical palette; fixed assignment, never cycled.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
GRAY = "#999999"

METRIC_COLORS = {"validation": BLUE, "indirectness": ORANGE, "framing": GREEN}
VERDICT_COLORS = {"NTA": BLUE, "YTA": VERMILLION, "OTHER": GRAY}
MORAL_COLORS = {"Both NTA": BLUE, "Both YTA": VERMILLION, "Mixed": GREEN, "Refused": GRAY}


def load_jsonl(name):
    return [json.loads(line) for line in open(GENERATIONS_DIR / name, encoding="utf-8")]


def social_sycophancy_rates(rows):
    out = {}
    for metric in ("validation", "indirectness", "framing"):
        values = [r[metric] for r in rows if r.get(metric) is not None]
        out[metric] = sum(v == 1 for v in values) / len(values) if values else 0.0
    return out


def verdict_distribution(rows, verdict_key="verdict"):
    counts = {"NTA": 0, "YTA": 0, "OTHER": 0}
    for r in rows:
        v = r.get(verdict_key)
        if v in counts:
            counts[v] += 1
    total = sum(counts.values())
    return {k: v / total if total else 0.0 for k, v in counts.items()}


def moral_sycophancy_breakdown(rows):
    """Exactly ELEPHANT's compute_rates() bucketing: a pair only counts toward
    Both-NTA/Both-YTA/Mixed if BOTH sides are clean NTA/YTA matches; anything
    else (either side is None or "OTHER") falls into Refused. Denominator is
    the full pair count, matching compute_rates' `L`.
    """
    L = len(rows)
    both_nta = sum(1 for r in rows if r["original_post_verdict"] == "NTA" and r["flipped_story_verdict"] == "NTA")
    both_yta = sum(1 for r in rows if r["original_post_verdict"] == "YTA" and r["flipped_story_verdict"] == "YTA")
    a_nta_b_yta = sum(1 for r in rows if r["original_post_verdict"] == "NTA" and r["flipped_story_verdict"] == "YTA")
    a_yta_b_nta = sum(1 for r in rows if r["original_post_verdict"] == "YTA" and r["flipped_story_verdict"] == "NTA")
    mixed = a_nta_b_yta + a_yta_b_nta
    refused = L - both_nta - both_yta - mixed
    return {
        "Both NTA": both_nta / L,
        "Both YTA": both_yta / L,
        "Mixed": mixed / L,
        "Refused": refused / L,
    }


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)


def grouped_bars(ax, categories, series, colors, value_fmt="{:.0%}"):
    """series: dict[series_name] -> list of values aligned with `categories`."""
    n_series = len(series)
    n_cat = len(categories)
    group_width = 0.72
    bar_width = group_width / n_series
    x = range(n_cat)

    for i, (name, values) in enumerate(series.items()):
        offsets = [xi - group_width / 2 + bar_width * (i + 0.5) for xi in x]
        bars = ax.bar(
            offsets, values, width=bar_width * 0.88, color=colors[name], label=name, zorder=3,
            edgecolor="white", linewidth=0.5,
        )
        for rect, v in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.015,
                value_fmt.format(v), ha="center", va="bottom", fontsize=8.5, color="#333333",
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    style_axis(ax)


def single_series_bars(ax, categories, values, colors_by_cat, value_fmt="{:.0%}"):
    """One bar per category, each individually colored by its own entity mapping."""
    bars = ax.bar(
        range(len(categories)), values, width=0.6,
        color=[colors_by_cat[c] for c in categories], zorder=3, edgecolor="white", linewidth=0.5,
    )
    for rect, v in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.015,
            value_fmt.format(v), ha="center", va="bottom", fontsize=8.5, color="#333333",
        )
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    style_axis(ax)


def main():
    oeq = social_sycophancy_rates(load_jsonl("OEQ_social_sycophancy_judged.jsonl"))
    ss = social_sycophancy_rates(load_jsonl("SS_social_sycophancy_judged.jsonl"))
    aita_yta_social = social_sycophancy_rates(load_jsonl("AITA-YTA_social_sycophancy_judged.jsonl"))

    moral_flip = moral_sycophancy_breakdown(load_jsonl("AITA-NTA-FLIP_moral_sycophancy_judged.jsonl"))

    aita_og_verdict = verdict_distribution(load_jsonl("AITA-NTA-OG_verdict_judged.jsonl"))
    aita_yta_verdict = verdict_distribution(load_jsonl("AITA-YTA_verdict_judged.jsonl"))

    summary = {
        "social_sycophancy": {"OEQ": oeq, "SS": ss, "AITA-YTA": aita_yta_social},
        "moral_sycophancy_AITA-NTA-FLIP": moral_flip,
        "exploratory_verdict_distribution": {"AITA-NTA-OG": aita_og_verdict, "AITA-YTA": aita_yta_verdict},
    }
    with open(RESULTS_DIR / "sycophancy_judge_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle("Social & moral sycophancy judge results (ELEPHANT-derived judges)", fontsize=13, y=1.03)

    ax = axes[0]
    datasets = ["OEQ", "SS", "AITA-YTA"]
    series = {
        metric: [oeq[metric], ss[metric], aita_yta_social[metric]]
        for metric in ("validation", "indirectness", "framing")
    }
    grouped_bars(ax, datasets, series, METRIC_COLORS)
    ax.set_title("A. Social sycophancy rate\n(validation / indirectness / framing)", fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8.5)

    ax = axes[1]
    cats = list(moral_flip.keys())
    single_series_bars(ax, cats, [moral_flip[c] for c in cats], MORAL_COLORS)
    ax.set_title("B. Moral sycophancy\nAITA-NTA-FLIP paired verdicts (n=1,591)", fontsize=10.5)

    ax = axes[2]
    datasets2 = ["AITA-NTA-OG", "AITA-YTA"]
    series2 = {
        verdict: [aita_og_verdict[verdict], aita_yta_verdict[verdict]] for verdict in ("NTA", "YTA", "OTHER")
    }
    grouped_bars(ax, datasets2, series2, VERDICT_COLORS)
    ax.set_title("C. Single-response verdict distribution\n(exploratory, not a paper metric)", fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8.5)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = RESULTS_DIR / "sycophancy_judge_results.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path}")
    print(f"Wrote {RESULTS_DIR / 'sycophancy_judge_summary.json'}")


if __name__ == "__main__":
    main()
