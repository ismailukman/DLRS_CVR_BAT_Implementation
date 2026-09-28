#!/usr/bin/env python
r"""Per-region HC-vs-SCI effects, grouped by resting-state network.

The most informative view of the 216-parcel result: every parcel is plotted
as a point, organised by the network it belongs to, so the reader sees all
216 regions at once, where they sit relative to the significance threshold,
and whether the effects cluster by network.

Why this rather than a Manhattan plot or a brain map: the x-axis carries
meaning (network membership) instead of an arbitrary ordering, the spread
within each network is visible, and the fact that nothing clears the
threshold is immediate rather than inferred.

The quantity plotted is the covariate-adjusted (age + motion) group t
statistic.  t = beta/SE is dimensionless, so CVR and BAT can share one axis
even though their native units and display ranges differ (+/-0.8 vs +/-1.2
a.u.); their adjusted DIFFERENCES could not.

Reads   figure/region216_stats.csv   (written by region_ancova_216.py)
Writes  figure/region216_by_network.png

Needs no access to the external data volume.

Run: /opt/anaconda3/envs/cvr_bat/bin/python cohort_scripts/region216_network_figure.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats as sp_stats

HERE = os.path.dirname(os.path.abspath(__file__))
# This script lives in src/analysis, so the project root is two levels up.
PROJ = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(PROJ, "figure")

ORDER = ["Visual", "Somatomotor", "DorsAttn", "VentAttn/Sal",
         "Limbic", "Control/FP", "Default", "Subcortex"]
SHORT = {"Visual": "Visual", "Somatomotor": "Somato-\nmotor",
         "DorsAttn": "Dorsal\nattention", "VentAttn/Sal": "Salience/\nvent attn",
         "Limbic": "Limbic", "Control/FP": "Control/\nfronto-par",
         "Default": "Default\nmode", "Subcortex": "Sub-\ncortex"}
COLORS = {"Visual": "#781286", "Somatomotor": "#4682B4",
          "DorsAttn": "#00760E", "VentAttn/Sal": "#C43AFA",
          "Limbic": "#8FBF5A", "Control/FP": "#E69422",
          "Default": "#CD3E4E", "Subcortex": "#555555"}


def load():
    rows = [r for r in csv.DictReader(open(os.path.join(FIG, "region216_stats.csv")))
            if r["testable"] == "1"]
    if not rows:
        raise SystemExit("no testable rows in region216_stats.csv")
    return rows


def main():
    rows = load()
    dof = int(float(rows[0]["dof"]))
    t_crit = float(sp_stats.t.ppf(0.975, dof))          # uncorrected p<0.05
    n_tot = len({r["region"] for r in rows})
    print(f"{n_tot} regions, dof={dof}, uncorrected |t| threshold = {t_crit:.2f}")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.4), sharex=True)

    n_sig_total = 0
    for ax, metric in zip(axes, ("CVR", "BAT")):
        sel = [r for r in rows if r["metric"] == metric]
        n_sig = sum(float(r["adj_p_fdr"]) < 0.05 for r in sel)
        n_sig_total += n_sig
        n_unc = sum(abs(float(r["adj_t"])) > t_crit for r in sel)

        # threshold bands first, so the points sit on top
        ax.axhspan(-t_crit, t_crit, color="0.90", zorder=0)
        ax.axhline(0, color="0.45", lw=1.0, zorder=1)
        for sgn in (1, -1):
            ax.axhline(sgn * t_crit, color="#7F8C8D", lw=1.1, ls="--", zorder=1)

        all_t = np.array([float(r["adj_t"]) for r in sel])
        lim = max(np.abs(all_t).max(), t_crit) * 1.12
        ax.set_ylim(-lim * 1.06, lim * 1.20)   # headroom for the % labels

        rng = np.random.default_rng(0)
        pos, pct = [], []
        for i, net in enumerate(ORDER):
            vals = np.array([float(r["adj_t"]) for r in sel if r["network"] == net])
            if vals.size == 0:
                continue
            pos.append(i)
            pct.append(100.0 * (vals > 0).mean())
            bp = ax.boxplot(vals, positions=[i], widths=0.52, patch_artist=True,
                            showfliers=False, zorder=2,
                            medianprops=dict(color="black", lw=1.6),
                            whiskerprops=dict(color="0.4"),
                            capprops=dict(color="0.4"))
            for b in bp["boxes"]:
                b.set_facecolor(COLORS[net])
                b.set_alpha(0.35)
                b.set_edgecolor("0.3")
            x = rng.normal(i, 0.075, size=vals.size)
            ax.scatter(x, vals, s=26, color=COLORS[net], edgecolors="0.25",
                       linewidths=0.4, alpha=0.9, zorder=3)
            ax.text(i, 0.012, f"n={vals.size}", ha="center", va="bottom",
                    fontsize=7.5, color="0.42",
                    transform=ax.get_xaxis_transform())

        ax.set_ylabel(f"{metric}\nadjusted $t$  (SCI $-$ HC)", fontsize=11,
                      fontweight="bold")
        ax.set_xlim(-0.6, len(ORDER) - 0.4)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)

        # Descriptive only: the share of parcels pointing the same way.  These
        # parcels are spatially correlated, so this is NOT a test and is
        # labelled as such in the caption.
        for i, p in zip(pos, pct):
            ax.text(i, 0.975, f"{p:.0f}%", ha="center", va="top",
                    fontsize=8.5, fontweight="bold", color="0.25",
                    transform=ax.get_xaxis_transform())

        ax.set_title(f"{n_unc} of {len(sel)} parcels exceed the uncorrected "
                     f"$p<0.05$ threshold "
                     f"(chance expectation {0.05 * len(sel):.0f}); "
                     f"{'none' if n_sig == 0 else n_sig} survive FDR",
                     fontsize=10, color="0.25", pad=4)

    axes[1].set_xticks(range(len(ORDER)))
    axes[1].set_xticklabels([SHORT[n] for n in ORDER], fontsize=9.5)

    handles = [
        Line2D([], [], color="#7F8C8D", ls="--", lw=1.1,
               label=f"uncorrected $p<0.05$ threshold  ($|t|={t_crit:.2f}$, "
                     f"df$=${dof})"),
        Line2D([], [], marker="s", color="0.6", markerfacecolor="0.90",
               markersize=11, ls="none",
               label="shaded band: below that threshold"),
        Line2D([], [], marker="o", color="0.3", markerfacecolor="0.6",
               markersize=6, ls="none", label="one parcel"),
    ]
    axes[0].legend(handles=handles, fontsize=8.5, ncol=3, frameon=False,
                   loc="lower center", bbox_to_anchor=(0.5, 1.055))

    fig.suptitle("Per-region HC-vs-SCI effect by resting-state network: "
                 f"every one of the {n_tot} parcels, age- and motion-adjusted",
                 fontsize=13.5, fontweight="bold", y=0.995)

    if n_sig_total == 0:
        banner = ("No parcel survives FDR correction (q < 0.05) for either metric. "
                  "Bold percentages are the share of parcels in that network with "
                  "SCI higher - descriptive only, not a test.")
        colour = "#8B1A1A"
    else:
        banner = (f"{n_sig_total} parcel(s) survive FDR correction (q < 0.05). "
                  "Bold percentages are the share with SCI higher (descriptive).")
        colour = "#1A5276"
    fig.text(0.5, 0.012, banner, ha="center", fontsize=9.5, color=colour,
             fontweight="bold")

    fig.tight_layout(rect=[0, 0.035, 1, 0.945])
    out = os.path.join(FIG, "region216_by_network.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
