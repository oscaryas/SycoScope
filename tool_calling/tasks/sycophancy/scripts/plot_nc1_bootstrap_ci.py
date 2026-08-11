"""
Plots the bootstrap CI on NC1 (see bootstrap_nc1_pipeline.py) for the three
groupings (category, label, category_label) by residual layer.

Three panels, one per grouping -- their NC1 scales differ by orders of
magnitude (category ~0.4-0.9, label up to ~200+), so a shared linear axis
would flatten two of the three into invisibility. Point estimate as a
marker, 95% bootstrap percentile CI as an error bar. Where the point
estimate falls outside its own CI (a real bootstrap-bias artifact discussed
in the run, not a bug) it's marked with a hollow/differently-colored point
so it doesn't read as a plotting error.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PROBING_DIR = RESULTS_DIR / "probing"

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"

GROUPING_STYLE = {
    "category": {"color": BLUE, "title": "category (C=4)", "yscale": "linear"},
    "label": {"color": VERMILLION, "title": "sycophancy label (C=2)", "yscale": "log"},
    "category_label": {"color": GREEN, "title": "category x label (C=8)", "yscale": "log"},
}


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)


def main():
    data = json.load(open(PROBING_DIR / "nc1_bootstrap_ci.json"))
    layers = data["layers"]
    n_boot = data["n_bootstrap"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"NC1 neural collapse: bootstrap 95% CI by layer (n_bootstrap={n_boot})",
        fontsize=12.5, y=1.03,
    )

    for ax, (grouping, style) in zip(axes, GROUPING_STYLE.items()):
        by_layer = data["results"][grouping]
        points = [by_layer[str(l)]["point_estimate"] for l in layers]
        ci_lo = [by_layer[str(l)]["ci_lower_2.5"] for l in layers]
        ci_hi = [by_layer[str(l)]["ci_upper_97.5"] for l in layers]

        yerr_lo = [max(0, p - lo) for p, lo in zip(points, ci_lo)]
        yerr_hi = [max(0, hi - p) for p, hi in zip(points, ci_hi)]

        outside_ci = [p < lo or p > hi for p, lo, hi in zip(points, ci_lo, ci_hi)]

        ax.errorbar(
            layers, points, yerr=[yerr_lo, yerr_hi],
            fmt="o-", color=style["color"], ecolor=style["color"], elinewidth=1.5,
            capsize=4, markersize=6, linewidth=1.5, zorder=3,
        )
        # Mark point estimates that fall outside their own bootstrap CI --
        # a real resampling-bias artifact (repeated points under with-
        # replacement resampling slightly shrink within-class variance),
        # not a plotting error -- flagged so it doesn't look like one.
        for l, p, outside in zip(layers, points, outside_ci):
            if outside:
                ax.scatter([l], [p], facecolors="none", edgecolors="black", s=110, linewidths=1.5, zorder=4)

        ax.set_yscale(style["yscale"])
        ax.set_xlabel("Residual layer")
        ax.set_ylabel("NC1")
        ax.set_title(style["title"], fontsize=11)
        style_axis(ax)

    fig.text(
        0.5, -0.02,
        "Hollow-ringed points: point estimate falls outside its own bootstrap CI (resampling-bias artifact, see run notes)",
        ha="center", fontsize=8.5, color="#555555",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = PROBING_DIR / "nc1_bootstrap_ci_plot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
