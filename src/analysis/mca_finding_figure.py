#!/usr/bin/env python
r"""The study's one positive finding, given its own figure.

Reviewer-driven: the abstract is titled around the age-adjusted MCA CVR
difference, yet in the previous figure set that result was a single dot
among thirty in a forest plot, while two of three figures were devoted to
null regional landscapes.  The finding needs to be visible, and so does the
reason it only appears after adjustment.

Panel A  MCA-territory CVR per participant, by group.  The raw distributions,
         with the unadjusted and adjusted p values side by side so the reader
         sees that adjustment strengthens rather than weakens the effect.

Panel B  The mechanism, which was previously text-only: MCA CVR falls with
         age, and the SCI group is older, so the unadjusted comparison is
         confounded in the direction that HIDES the group difference.  Lines
         are the fitted ANCOVA (common age slope, evaluated at mean motion),
         i.e. the actual model, not two independent per-group regressions.

Everything is recomputed here from figure/ancova_per_subject.csv rather than
copied from the text, so the figure cannot drift from the statistics.

Writes figure/mca_finding.png
Needs no access to the external data volume.

Run: /opt/anaconda3/envs/cvr_bat/bin/python cohort_scripts/mca_finding_figure.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
FIG = os.path.join(PROJ, "figure")

HC_COLOR, SCI_COLOR = "#4DBEEE", "#FF6B6B"
REGION = "MCA-total|CVR"


def load():
    path = os.path.join(FIG, "ancova_per_subject.csv")
    rows = list(csv.DictReader(open(path)))
    g = np.array([1.0 if r["group"] == "SCI" else 0.0 for r in rows])
    age = np.array([float(r["age"]) for r in rows])
    mot = np.array([float(r["motion"]) for r in rows])
    y = np.array([float(r[REGION]) for r in rows])
    return g, age, mot, y


def ancova(y, g, age, mot):
    X = np.column_stack([np.ones(len(y)), g, age, mot])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    mse = float(resid @ resid) / dof
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * mse)
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    return beta, se, t, p, dof


def main():
    g, age, mot, y = load()
    beta, se, t, p, dof = ancova(y, g, age, mot)
    hc, sci = y[g == 0], y[g == 1]
    n_hc, n_sci = len(hc), len(sci)
    sp = np.sqrt(((n_hc - 1) * hc.var(ddof=1) + (n_sci - 1) * sci.var(ddof=1))
                 / (n_hc + n_sci - 2))
    d = (sci.mean() - hc.mean()) / sp
    p_unadj = stats.ttest_ind(hc, sci, equal_var=False)[1]
    peta2 = t[1] ** 2 / (t[1] ** 2 + dof)
    print(f"adjusted group p={p[1]:.4f}  d={d:+.2f}  partial eta2={peta2:.3f}")
    print(f"unadjusted p={p_unadj:.4f}   age p={p[2]:.4f} (slope {beta[2]:+.5f}/yr)")
    print(f"mean age: HC {age[g==0].mean():.1f}, SCI {age[g==1].mean():.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))

    # ---------------- Panel A: the effect --------------------------------
    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, (vals, lab, col) in enumerate([(hc, f"HC\n(n={n_hc})", HC_COLOR),
                                          (sci, f"SCI\n(n={n_sci})", SCI_COLOR)]):
        bp = ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=1.8),
                        whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"))
        for b in bp["boxes"]:
            b.set_facecolor(col); b.set_alpha(0.45); b.set_edgecolor("0.3")
        ax.scatter(rng.normal(i, 0.055, size=vals.size), vals, s=46, color=col,
                   edgecolors="0.2", linewidths=0.5, zorder=3, alpha=0.95)
        ax.scatter([i], [vals.mean()], marker="D", s=62, color="black", zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"HC\n(n={n_hc})", f"SCI\n(n={n_sci})"], fontsize=11)
    ax.set_ylabel("MCA-territory mean CVR (a.u.)", fontsize=11)
    ax.axhline(0, color="0.75", lw=0.9, ls=":")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_title("A.  MCA CVR is higher in SCI", fontsize=12, fontweight="bold",
                 loc="left")
    txt = (f"unadjusted  $p$ = {p_unadj:.3f}  (n.s.)\n"
           f"age + motion adjusted  $p$ = {p[1]:.3f}\n"
           f"Cohen $d$ = {d:+.2f},  $\\eta^2_p$ = {peta2:.2f}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9.5, bbox=dict(boxstyle="round,pad=0.45", fc="white",
                                    ec="0.7", alpha=0.95))

    # ---------------- Panel B: why adjustment matters --------------------
    ax = axes[1]
    xs = np.linspace(age.min() - 1, age.max() + 1, 50)
    mot_bar = mot.mean()
    for gv, lab, col in [(0.0, "HC", HC_COLOR), (1.0, "SCI", SCI_COLOR)]:
        m = g == gv
        ax.scatter(age[m], y[m], s=48, color=col, edgecolors="0.2",
                   linewidths=0.5, alpha=0.95, label=lab, zorder=3)
        # the ANCOVA's own fitted line: common age slope, group offset
        ax.plot(xs, beta[0] + beta[1] * gv + beta[2] * xs + beta[3] * mot_bar,
                color=col, lw=2.2, zorder=2)
    ax.set_xlabel("Age (years)", fontsize=11)
    ax.set_ylabel("MCA-territory mean CVR (a.u.)", fontsize=11)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="upper right", frameon=True)
    ax.set_title("B.  Why the raw test understates it", fontsize=12,
                 fontweight="bold", loc="left")
    txt = (f"MCA CVR falls with age\n"
           f"({beta[2]:+.4f} a.u./yr, $p$ = {p[2]:.3f})\n"
           f"and SCI are older "
           f"({age[g==1].mean():.1f} vs {age[g==0].mean():.1f} yr),\n"
           f"which masks the group difference.\n"
           f"Lines: fitted ANCOVA (common slope).")
    ax.text(0.03, 0.03, txt, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.45", fc="white",
                                  ec="0.7", alpha=0.95))

    fig.suptitle("The one group difference the study detects: "
                 "middle cerebral artery territory CVR",
                 fontsize=13.5, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005,
             "Pre-specified anatomical ROI, tested on its own and not part of "
             "the exploratory regional family. CVR is in arbitrary units "
             "(the model's 5.tanh output), so only relative values carry meaning.",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=[0, 0.03, 1, 0.945])

    out = os.path.join(FIG, "mca_finding.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
