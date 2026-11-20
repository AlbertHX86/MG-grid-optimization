"""Render the annual AC figures from the report's local, verified summaries."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "results" / "annual_ac"
FIGURES = HERE / "figures"

# Palette and print typography used by the rest of the planning study.
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECONDARY, GRID = "#0b0b0b", "#52514e", "#dedcd6"
REPORT_STYLE = {
    "font.family": "serif", "font.size": 8.5, "pdf.fonttype": 42,
    "text.color": INK, "axes.edgecolor": SECONDARY, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.axisbelow": True, "xtick.color": SECONDARY, "ytick.color": SECONDARY,
}

NORMAL_CASES = [
    "p5_2035_reference", "p5_2050_reference", "p5_2050_bess_at_load",
    "p6_2035_sequential", "p6_2035_siting", "p6_2050_sequential", "p6_2050_siting",
]
OUTAGE_CASES = NORMAL_CASES[3:]


def read_summary(name, case_order):
    data = pd.read_csv(DATA / name).set_index("case", verify_integrity=True)
    if set(data.index) != set(case_order) or not data["hours"].eq(8760).all():
        raise ValueError(f"Unexpected case set or hour coverage in {name}")
    return data.loc[case_order]


def labels(axis, rows, plans, headings):
    axis.set_axis_off()
    for y, name in headings:
        axis.text(0, y, name, weight="bold", va="center", ha="left", fontsize=8.8)
    for y, name in zip(rows, plans):
        axis.text(0.04, y, name, va="center", ha="left", fontsize=8.4)


def draw_series(ax, rows, values, color, marker, title, unit, xmax, ticks, tick_labels=None):
    values = np.asarray(values, dtype=np.int64)
    ax.set_xlim(0, xmax)
    ax.set_xticks(ticks, labels=tick_labels)
    if tick_labels is None:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda n, pos: f"{n:,.0f}"))
    ax.tick_params(axis="x", labelsize=7.7, length=2.8, pad=2.4)
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.spines["bottom"].set_linewidth(.6)
    ax.set_title(title, loc="left", fontsize=9.2, weight="bold", pad=5)
    ax.set_xlabel(unit, fontsize=8.0, labelpad=3)
    ax.hlines(rows, 0, values, colors=color, linewidth=1.7, alpha=.63)
    ax.scatter(values, rows, c=color, s=19, marker=marker, zorder=3, clip_on=False)
    for value, yy in zip(values, rows):
        ax.annotate(f"{value:,}", (value, yy), xytext=(5.0, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8.0, color=INK)


def export(fig, stem):
    fig.savefig(FIGURES / (stem + ".pdf"), metadata={"CreationDate": None})
    fig.savefig(FIGURES / (stem + ".png"), dpi=250)
    plt.close(fig)


def normal_figure():
    normal = read_summary("n0_summary.csv", NORMAL_CASES)
    rows = np.array([.85, 1.62, 2.39, 4.02, 4.79, 5.56, 6.33])
    fig, axes = plt.subplots(1, 3, figsize=(6.50, 2.25),
                             gridspec_kw={"width_ratios": [1.05, 1.70, 1.85]})
    fig.subplots_adjust(left=.012, right=.986, top=.886, bottom=.192, wspace=.18)
    for ax in axes:
        ax.set_ylim(6.72, -.34)
    labels(axes[0], rows, ["2035  Reference", "2050  Reference", "2050  BESS at load",
                           "2035  Reference", "2035  Joint siting", "2050  Reference", "2050  Joint siting"],
           [(0, "Phase 5"), (3.23, "Phase 6")])
    draw_series(axes[1], rows, normal["new_low_hours"], BLUE, "o",
                "New low voltage", "Hours", 740, [0, 200, 400, 600])
    draw_series(axes[2], rows, normal["new_high_hours"], ORANGE, "s",
                "New high voltage", "Hours", 10100, [0, 2000, 4000, 6000, 8000])
    export(fig, "annual_ac_n0")


def outage_figure():
    outages = read_summary("n1_summary.csv", OUTAGE_CASES)
    rows = np.array([.72, 1.42, 2.97, 3.67])
    fig, axes = plt.subplots(1, 4, figsize=(6.50, 2.11),
                             gridspec_kw={"width_ratios": [.70, 1.38, 1.36, 1.74]})
    fig.subplots_adjust(left=.012, right=.98, top=.82, bottom=.23, wspace=.22)
    for ax in axes:
        ax.set_ylim(4.09, -.15)
    labels(axes[0], rows, ["Reference", "Joint siting", "Reference", "Joint siting"],
           [(0, "2035"), (2.23, "2050")])
    draw_series(axes[1], rows, outages["hours_ac_new_snapshot"], GREEN, "D",
                "New thermal overload", "Hours", 5950, [0, 2000, 4000])
    draw_series(axes[2], rows, outages["hours_voltage_low_new_n0"], BLUE, "o",
                "New low voltage", "Hours", 1450, [0, 500, 1000])
    draw_series(axes[3], rows, outages["voltage_high_new_n0"], ORANGE, "s",
                "New high voltage", "Bus–outage–hours", 1030000,
                [0, 250000, 500000, 750000], ["0", "250k", "500k", "750k"])
    export(fig, "annual_ac_n1")
    table_rows = []
    for case, row in outages.iterrows():
        _, year, method = case.split("_")
        name = {"sequential": "Reference", "siting": "Joint siting"}[method]
        table_rows.append(
            f"{year} & {name} & {row['ac_new_snapshot']:,.0f} & "
            f"{row['voltage_low_new_n0']:,.0f} & {100*row['ac_new_snapshot_peak_ratio']:.2f} & "
            f"{row['min_vm_pu']:.4f} & {row['pf_failed']:.0f} " + r"\\"
        )
    (HERE / "tables" / "annual_ac_n1_rows.tex").write_text("\n".join(table_rows) + "\n", encoding="utf-8")


def main():
    with plt.rc_context(REPORT_STYLE):
        normal_figure()
        outage_figure()


if __name__ == "__main__":
    main()
