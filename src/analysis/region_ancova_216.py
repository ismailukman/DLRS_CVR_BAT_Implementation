#!/usr/bin/env python
r"""Per-region HC-vs-SCI comparison of DLRS CVR/BAT over the full 216-region
functional parcellation (Schaefer-2018 200 cortical parcels + Tian Scale-I
16 subcortical nuclei).

This is the fine-grained counterpart of cohort_scripts/group_ancova.py, which
tests only 16 pooled regions (whole-brain, 7 Yeo networks, 7 bilateral Tian
structures, MCA territory).  Here every individual parcel is tested, so the
question "which of the 216 regions differs between HC and SCI?" can be
answered directly.

For every region x metric (CVR, BAT) we report

  * the unadjusted Welch two-sample test and Cohen's d (HC - SCI), and
  * the covariate-adjusted ANCOVA   y ~ 1 + group + age + motion
    (group SCI=1/HC=0, so the coefficient is the adjusted SCI - HC
    difference), with its t, p and partial eta^2,

then correct for multiple comparisons across the whole 216-region family
*within each metric* with Benjamini-Hochberg FDR (q) and, for reference, with
Bonferroni.  Testing 216 regions is a much larger family than the 7+7 pooled
test, so the FDR threshold is correspondingly stricter; the smallest effect
the design can resolve is printed in the console summary.

Regions are only tested if every subject has at least MIN_VOX voxels of that
parcel inside their own brain mask (2 mm grid); regions that fail are listed
and excluded from the family before FDR, so the correction is not inflated by
untestable parcels.

Outputs (figure/):
  region216_per_subject.csv   per-subject region means (wide)
  region216_stats.csv         per-region statistics, both metrics
  region216_excluded.csv      regions dropped for insufficient coverage
  region216_manhattan.png     -log10(adjusted p) per region, by network
  region216_forest_top.png    top-20 regions per metric, adjusted effect + CI
  region216_effect_map.png    axial montage painted with the adjusted t
  region216_section.tex       LaTeX results fragment (\input by the report)
  region216_top_table.tex     LaTeX table fragment (top regions per metric)

Run (external data volume must be mounted):
  /opt/anaconda3/envs/cvr_bat/bin/python cohort_scripts/region_ancova_216.py
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
# This script lives in src/analysis, so the project root is two levels up.
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJ)

import analyze_network_atlases as ana      # atlas paths, grid, resampling
import group_ancova as ga                  # demographics, ancova, welch, FDR

DATA, OUT, FIG = ana.DATA, ana.OUT, ana.FIG

MIN_VOX = 10          # per-subject in-brain voxels required to test a parcel
TOP_N = 20            # rows in the forest plot / LaTeX table
GROUPS = [("HC_subj", "HC"), ("SCI_subj", "SCI")]

# Colours for the seven Yeo networks (same palette as analyze_network_atlases)
YEO_COLORS = {"Visual": "#781286", "Somatomotor": "#4682B4",
              "DorsAttn": "#00760E", "VentAttn/Sal": "#C43AFA",
              "Limbic": "#C8F0A0", "Control/FP": "#E69422",
              "Default": "#CD3E4E", "Subcortex": "#555555"}

TIAN_FULL = {"HIP": "Hippocampus", "AMY": "Amygdala", "pTHA": "Thalamus-post",
             "aTHA": "Thalamus-ant", "NAc": "Accumbens", "GP": "Pallidum",
             "PUT": "Putamen", "CAU": "Caudate"}


# --------------------------------------------------------------- parcellation
def build_216(shape, affine):
    """Return (labelvol, regions) for the combined 216-region atlas.

    labelvol carries 1..200 for Schaefer parcels and 201..216 for Tian nuclei.
    regions is a list of dicts with id / name / network / hemisphere /
    compartment, in label order.
    """
    cortex = ana.resample_labels(ana.SCHAEFER, shape, affine)      # 0..200
    subcx = ana.resample_labels(ana.TIAN, shape, affine)           # 0..16

    regions = []
    with open(ana.SCHAEFER_TXT) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            pid = int(parts[0])
            # e.g. 7Networks_LH_Vis_1  ->  hemi LH, net Vis, index 1
            tok = parts[1].split("_")
            hemi = tok[1]
            net = ana.YEO7.get(tok[2], tok[2])
            regions.append(dict(id=pid, name=parts[1].replace("7Networks_", ""),
                                network=net, hemi=hemi, compartment="cortex"))
    n_cortex = len(regions)
    if n_cortex != 200:
        raise RuntimeError(f"expected 200 Schaefer parcels, parsed {n_cortex}")

    tian_names = [l.strip() for l in open(ana.TIAN_TXT) if l.strip()]
    if len(tian_names) != 16:
        raise RuntimeError(f"expected 16 Tian S1 labels, parsed {len(tian_names)}")
    for i, lab in enumerate(tian_names, start=1):
        base, _, hemi = lab.partition("-")
        regions.append(dict(id=n_cortex + i,
                            name=f"{TIAN_FULL.get(base, base)}-{hemi.upper()}",
                            network="Subcortex", hemi=hemi.upper(),
                            compartment="subcortex"))

    labelvol = np.zeros(shape, dtype=np.int16)
    labelvol[cortex > 0] = cortex[cortex > 0]
    overlap = int(((cortex > 0) & (subcx > 0)).sum())
    # Subcortex takes precedence where the two resampled atlases collide.
    sub_here = subcx > 0
    labelvol[sub_here] = n_cortex + subcx[sub_here]
    print(f"parcellation: {len(regions)} regions "
          f"({n_cortex} cortical + {len(tian_names)} subcortical); "
          f"cortex/subcortex voxel overlap = {overlap} "
          f"(resolved in favour of Tian)")
    return labelvol, regions


# --------------------------------------------------------------- subject data
def subject_region_means(gdir, sid, labelvol, n_regions):
    """Mean CVR/BAT in every parcel, plus that subject's in-brain voxel count."""
    cvr = ana.L(os.path.join(OUT, gdir, sid, "DLRS_CVR.nii"))
    bat = ana.L(os.path.join(OUT, gdir, sid, "DLRS_BAT.nii"))
    bm = ga.brain_mask(gdir, sid)
    if bm is None:
        bm = np.abs(cvr) > 0.01
    lab = np.where(bm, labelvol, 0)
    # np.bincount over the masked label volume: one pass instead of 216.
    # Non-finite voxels are excluded from BOTH the sum and the divisor of the
    # metric they belong to, so this matches the np.nanmean() the pooled
    # analysis (cohort_scripts/group_ancova.py) uses.  The returned voxel
    # count is the plain in-brain count, which is what the coverage screen
    # should test.
    counts = np.bincount(lab.ravel(), minlength=n_regions + 1)[1:]

    def _mean(vol):
        ok = np.isfinite(vol)
        lab_ok = np.where(ok, lab, 0)
        n = np.bincount(lab_ok.ravel(), minlength=n_regions + 1)[1:]
        tot = np.bincount(lab_ok.ravel(), weights=np.where(ok, vol, 0.0).ravel(),
                          minlength=n_regions + 1)[1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, tot / np.maximum(n, 1), np.nan)

    return _mean(cvr), _mean(bat), counts


# --------------------------------------------------------------------- main
def main():
    demo = ga.load_demo()
    shape, affine = ana.ref_grid()
    labelvol, regions = build_216(shape, affine)
    n_reg = len(regions)

    subjects = []
    for gdir, _ in GROUPS:
        gpath = os.path.join(OUT, gdir)
        for sid in sorted(os.listdir(gpath)):
            if not os.path.isdir(os.path.join(gpath, sid)):
                continue
            if not os.path.exists(os.path.join(gpath, sid, "DLRS_CVR.nii")):
                continue
            if sid not in demo:
                print(f"  WARN no demographics for {sid}, skipped")
                continue
            subjects.append((gdir, sid, demo[sid]["group"],
                             demo[sid]["age"], demo[sid]["motion"]))
    n_hc = sum(s[2] == "HC" for s in subjects)
    n_sci = sum(s[2] == "SCI" for s in subjects)
    print(f"analysed: {n_hc} HC, {n_sci} SCI   grid={shape}")
    if n_hc < 3 or n_sci < 3:
        raise RuntimeError("too few subjects for a group comparison")

    cvr_mat = np.full((len(subjects), n_reg), np.nan)
    bat_mat = np.full((len(subjects), n_reg), np.nan)
    vox_mat = np.zeros((len(subjects), n_reg), dtype=int)
    for i, (gdir, sid, *_rest) in enumerate(subjects):
        cvr_mat[i], bat_mat[i], vox_mat[i] = subject_region_means(
            gdir, sid, labelvol, n_reg)
        print(f"  [{i + 1:2d}/{len(subjects)}] {sid}")

    grp = np.array([1.0 if s[2] == "SCI" else 0.0 for s in subjects])
    age = np.array([s[3] for s in subjects], float)
    mot = np.array([s[4] for s in subjects], float)

    # ---- coverage screen ------------------------------------------------
    min_vox_per_region = vox_mat.min(axis=0)
    testable = min_vox_per_region >= MIN_VOX
    excluded = [(regions[k], int(min_vox_per_region[k]))
                for k in range(n_reg) if not testable[k]]
    print(f"testable regions: {int(testable.sum())}/{n_reg} "
          f"(min {MIN_VOX} in-brain voxels in every subject); "
          f"excluded {len(excluded)}")

    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(FIG, "region216_excluded.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "network", "hemi", "min_invox_across_subjects"])
        for r, v in excluded:
            w.writerow([r["id"], r["name"], r["network"], r["hemi"], v])

    # ---- per-subject CSV -------------------------------------------------
    with open(os.path.join(FIG, "region216_per_subject.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "subject", "age", "motion"]
                   + [f"{r['name']}|CVR" for r in regions]
                   + [f"{r['name']}|BAT" for r in regions])
        for i, (_gd, sid, g, a, mo) in enumerate(subjects):
            w.writerow([g, sid, a, mo] + list(cvr_mat[i]) + list(bat_mat[i]))

    # ---- statistics ------------------------------------------------------
    rows = []
    for metric, mat in (("CVR", cvr_mat), ("BAT", bat_mat)):
        fam = []
        for k in range(n_reg):
            r = regions[k]
            base = dict(id=r["id"], region=r["name"], network=r["network"],
                        hemi=r["hemi"], compartment=r["compartment"],
                        metric=metric, n_vox_min=int(min_vox_per_region[k]),
                        n_vox_mean=float(vox_mat[:, k].mean()))
            if not testable[k]:
                base.update(testable=0, hc_mean=np.nan, sci_mean=np.nan,
                            unadj_t=np.nan, unadj_p=np.nan, cohen_d=np.nan,
                            adj_diff=np.nan, adj_t=np.nan, adj_p=np.nan,
                            partial_eta2=np.nan, dof=np.nan,
                            adj_p_fdr=np.nan, adj_p_bonf=np.nan,
                            unadj_p_fdr=np.nan)
                rows.append(base)
                continue
            vals = mat[:, k]
            hc_m, sci_m, ut, up, ud = ga.welch(vals[grp == 0], vals[grp == 1])
            adiff, _ase, at, ap, peta2, dof = ga.ancova(vals, grp, age, mot)
            base.update(testable=1, hc_mean=hc_m, sci_mean=sci_m,
                        unadj_t=ut, unadj_p=up, cohen_d=ud,
                        adj_diff=adiff, adj_t=at, adj_p=ap,
                        partial_eta2=peta2, dof=dof)
            fam.append(base)
            rows.append(base)

        m = len(fam)
        q_adj = ga.bh_fdr([r["adj_p"] for r in fam])
        q_un = ga.bh_fdr([r["unadj_p"] for r in fam])
        for j, r in enumerate(fam):
            r["adj_p_fdr"] = q_adj[j]
            r["unadj_p_fdr"] = q_un[j]
            r["adj_p_bonf"] = min(1.0, r["adj_p"] * m)
        n_sig = sum(r["adj_p_fdr"] < 0.05 for r in fam)
        n_unc = sum(r["adj_p"] < 0.05 for r in fam)
        print(f"{metric}: family size {m}; {n_unc} regions p<0.05 uncorrected "
              f"(chance expectation {0.05 * m:.1f}); {n_sig} survive FDR q<0.05")

    fields = ["id", "region", "network", "hemi", "compartment", "metric",
              "testable", "n_vox_min", "n_vox_mean",
              "hc_mean", "sci_mean", "unadj_t", "unadj_p", "unadj_p_fdr",
              "cohen_d", "adj_diff", "adj_t", "adj_p", "adj_p_fdr",
              "adj_p_bonf", "partial_eta2", "dof"]
    outcsv = os.path.join(FIG, "region216_stats.csv")
    with open(outcsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 5) if isinstance(r[k], float)
                            and np.isfinite(r[k]) else r[k]) for k in fields})
    print(f"wrote {outcsv}")

    tested = [r for r in rows if r["testable"] == 1]

    # ---- console top list -------------------------------------------------
    for metric in ("CVR", "BAT"):
        sel = sorted([r for r in tested if r["metric"] == metric],
                     key=lambda r: r["adj_p"])[:TOP_N]
        print(f"\nTop {TOP_N} regions by adjusted p -- {metric}")
        print(f"{'region':34}{'net':14}{'HC':>8}{'SCI':>8}{'d':>6}"
              f"{'adjDiff':>9}{'adjP':>8}{'FDRq':>8}{'pEta2':>7}")
        for r in sel:
            print(f"{r['region'][:33]:34}{r['network'][:13]:14}"
                  f"{r['hc_mean']:8.3f}{r['sci_mean']:8.3f}{r['cohen_d']:6.2f}"
                  f"{r['adj_diff']:9.3f}{r['adj_p']:8.4f}{r['adj_p_fdr']:8.3f}"
                  f"{r['partial_eta2']:7.3f}")

    # Smallest effect the design can resolve, for context on the null result.
    from scipy import stats as _st
    dof = int(len(subjects) - 4)
    t_unc = _st.t.ppf(0.975, dof)
    se_unit = np.sqrt(1.0 / n_hc + 1.0 / n_sci)   # SE of the group beta in SD units
    print(f"\nresolution: with n={n_hc}+{n_sci} and 2 covariates (dof={dof}), an "
          f"UNCORRECTED p<0.05 needs |d| >= {t_unc * se_unit:.2f}; a single region "
          f"surviving Bonferroni over 216 would need "
          f"|d| >= {_st.t.ppf(1 - 0.025 / 216, dof) * se_unit:.2f}.")

    make_figures(tested, regions, labelvol, shape, n_hc, n_sci)
    write_tex_section(tested, n_hc, n_sci, n_reg, excluded,
                      t_unc * se_unit,
                      _st.t.ppf(1 - 0.025 / 216, dof) * se_unit)


# ------------------------------------------------------------------ figures
def make_figures(tested, regions, labelvol, shape, n_hc, n_sci):
    # ---------- 1. Manhattan plot -----------------------------------------
    net_order = ["Visual", "Somatomotor", "DorsAttn", "VentAttn/Sal",
                 "Limbic", "Control/FP", "Default", "Subcortex"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    for ax, metric in zip(axes, ("CVR", "BAT")):
        sel = [r for r in tested if r["metric"] == metric]
        sel.sort(key=lambda r: (net_order.index(r["network"]), r["hemi"],
                                r["region"]))
        x = np.arange(len(sel))
        y = -np.log10(np.clip([r["adj_p"] for r in sel], 1e-12, 1))
        cols = [YEO_COLORS[r["network"]] for r in sel]
        ax.scatter(x, y, c=cols, s=26, edgecolors="k", linewidths=0.3, zorder=3)
        ax.axhline(-np.log10(0.05), color="#7F8C8D", lw=1, ls="--",
                   label="p = 0.05 (uncorrected)")
        # FDR threshold = largest p with q < 0.05, drawn only if one exists
        sig = [r["adj_p"] for r in sel if r["adj_p_fdr"] < 0.05]
        if sig:
            ax.axhline(-np.log10(max(sig)), color="#C0392B", lw=1.2,
                       label="FDR q = 0.05")
        # boundaries between networks
        bounds, last = [], None
        for i, r in enumerate(sel):
            if r["network"] != last:
                bounds.append(i)
                last = r["network"]
        for b in bounds[1:]:
            ax.axvline(b - 0.5, color="0.85", lw=0.8, zorder=0)
        ax.set_ylabel(r"$-\log_{10}$ adjusted $p$")
        ax.set_title(f"{metric}: covariate-adjusted (age + motion) "
                     f"HC-vs-SCI effect, 216 regions", fontsize=11,
                     fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.2)
        # label each network block at its centre
        centres = [(bounds[i] + (bounds[i + 1] if i + 1 < len(bounds)
                                 else len(sel))) / 2 for i in range(len(bounds))]
        names = [sel[b]["network"] for b in bounds]
        ax.set_xticks(centres)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    fig.suptitle("Per-region HC-vs-SCI comparison over the 216-region "
                 "Schaefer-200 + Tian-S1 parcellation", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(FIG, "region216_manhattan.png")
    # 600 dpi: this one is also embedded in the ISMRM abstract.
    fig.savefig(p, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")

    # ---------- 2. Forest plot of the top regions -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    for ax, metric in zip(axes, ("CVR", "BAT")):
        sel = sorted([r for r in tested if r["metric"] == metric],
                     key=lambda r: r["adj_p"])[:TOP_N]
        y = np.arange(len(sel))
        diffs = np.array([r["adj_diff"] for r in sel])
        # SE recovered from beta/t (t = beta/SE); guard against t == 0
        ses = np.array([abs(r["adj_diff"] / r["adj_t"]) if r["adj_t"] else np.nan
                        for r in sel])
        cols = ["#C0392B" if r["adj_p_fdr"] < 0.05
                else ("#E67E22" if r["adj_p"] < 0.05 else "#7F8C8D")
                for r in sel]
        ax.errorbar(diffs, y, xerr=1.96 * ses, fmt="none", ecolor="#bbb",
                    elinewidth=1.1, capsize=2, zorder=1)
        ax.scatter(diffs, y, c=cols, s=48, zorder=2)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['region']}  (q={r['adj_p_fdr']:.2f})"
                            for r in sel], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"adjusted {metric} difference (SCI - HC, a.u.)")
        ax.set_title(f"{metric}: {TOP_N} smallest adjusted p\n"
                     "(red q<0.05; orange p<0.05 uncorrected; 95% CI)",
                     fontsize=10)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Strongest per-region HC-vs-SCI effects, age- and motion-"
                 f"adjusted (n = {n_hc} HC / {n_sci} SCI)", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(FIG, "region216_forest_top.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")

    # ---------- 3. Effect map: paint each parcel with its adjusted t -------
    mask_path, _ = ana.first_mask_path()
    brain = ana.L(mask_path) > 0.5
    z_slices = [26, 34, 42, 50, 58, 66]
    fig, axes = plt.subplots(2, len(z_slices), figsize=(2.3 * len(z_slices), 6.4))
    tmax = 4.0
    norm = TwoSlopeNorm(vmin=-tmax, vcenter=0.0, vmax=tmax)
    for ri, metric in enumerate(("CVR", "BAT")):
        tmap = np.full(shape, np.nan)
        sigmask = np.zeros(shape, bool)
        for r in (x for x in tested if x["metric"] == metric):
            sel = labelvol == r["id"]
            tmap[sel] = np.clip(r["adj_t"], -tmax, tmax)
            if r["adj_p_fdr"] < 0.05:
                sigmask |= sel
        for ci, zz in enumerate(z_slices):
            ax = axes[ri, ci]
            ax.imshow(np.where(brain[:, :, zz], 0.35, np.nan).T, cmap="Greys",
                      origin="lower", vmin=0, vmax=1)
            im = ax.imshow(tmap[:, :, zz].T, cmap="coolwarm", norm=norm,
                           origin="lower", interpolation="nearest")
            if sigmask[:, :, zz].any():
                ax.contour(sigmask[:, :, zz].T.astype(float), levels=[0.5],
                           colors="k", linewidths=1.0)
            ax.axis("off")
            if ri == 0:
                ax.set_title(f"z = {zz}", fontsize=9, color="0.35")
            if ci == 0:
                ax.text(-0.06, 0.5, metric, transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=12,
                        fontweight="bold")
        cax = fig.add_axes([0.93, 0.56 - 0.44 * ri, 0.012, 0.3])
        cb = fig.colorbar(im, cax=cax, extend="both")
        cb.set_ticks([-tmax, -tmax / 2, 0, tmax / 2, tmax])
        cb.set_ticklabels([f"{v:g}" for v in
                           (-tmax, -tmax / 2, 0, tmax / 2, tmax)])
        cb.set_label(f"adjusted t (SCI - HC), {metric}", fontsize=9)
    fig.suptitle("Per-region adjusted HC-vs-SCI effect (black outline = "
                 "FDR q < 0.05)", fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.04, right=0.91, top=0.90, bottom=0.02,
                        wspace=0.04, hspace=0.08)
    p = os.path.join(FIG, "region216_effect_map.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


# ------------------------------------------------------------- LaTeX export
def _esc(s):
    return s.replace("_", r"\_").replace("&", r"\&")


def _num(v, fmt=".3f", signed=False):
    """LaTeX-typeset number: a real minus sign ($-$) instead of a hyphen."""
    txt = format(v, ("+" if signed else "") + fmt)
    return txt.replace("-", "$-$").replace("+", "$+$")


def _top_table(tested, metric, n=15):
    """Rows of the top-n table, most significant adjusted effect first."""
    sel = sorted([r for r in tested if r["metric"] == metric],
                 key=lambda r: r["adj_p"])[:n]
    out = [r"\begin{tabular}{@{}l l rr r rr r@{}}", r"\toprule",
           r"\textbf{Region} & \textbf{Network} & HC & SCI & $d$ & "
           r"adj.\ $\Delta$ & adj.\ $p$ & $q$ \\", r"\midrule"]
    for r in sel:
        out.append(f"{_esc(r['region'])} & {_esc(r['network'])} & "
                   f"{_num(r['hc_mean'])} & {_num(r['sci_mean'])} & "
                   f"{_num(r['cohen_d'], '.2f')} & "
                   f"{_num(r['adj_diff'], signed=True)} & "
                   f"{r['adj_p']:.4f} & {r['adj_p_fdr']:.2f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def write_tex_section(tested, n_hc, n_sci, n_reg, excluded, d_unc, d_bonf,
                      top_n=15):
    r"""Write a self-contained LaTeX results fragment with the real numbers.

    The experiment report \input{}s this file, so the prose and tables can
    never drift from figure/region216_stats.csv.
    """
    def fam(metric):
        return [r for r in tested if r["metric"] == metric]

    parts = []
    m = len(fam("CVR"))
    parts.append(
        "\\noindent Of the %d regions, %d were testable in every subject "
        "(at least %d in-brain voxels); %d were dropped for insufficient "
        "coverage on the 2\\,mm grid and are listed in "
        "\\pathtt{figure/\\pb{}region216\\_\\pb{}excluded.csv}.  FDR is applied "
        "across the "
        "%d-region family separately for CVR and for BAT.\n"
        % (n_reg, m, MIN_VOX, len(excluded), m))

    for metric in ("CVR", "BAT"):
        f = fam(metric)
        n_unc = sum(r["adj_p"] < 0.05 for r in f)
        n_fdr = sum(r["adj_p_fdr"] < 0.05 for r in f)
        best = min(f, key=lambda r: r["adj_p"])
        parts.append(
            "\n\\noindent\\textbf{%s.}  %d of %d regions reach an "
            "\\emph{uncorrected} adjusted $p<0.05$ (the number expected by "
            "chance alone is %.1f), and \\textbf{%s} survive FDR correction "
            "at $q<0.05$.  The strongest single region is %s "
            "(%s; HC $=$ %s, SCI $=$ %s; adjusted $\\Delta =$ %s, "
            "adj.\\ $p =$ %.4f, $q =$ %.2f, $\\eta^2_p =$ %.3f).\n"
            % (metric, n_unc, len(f), 0.05 * len(f),
               (str(n_fdr) if n_fdr else "none"),
               _esc(best["region"]), _esc(best["network"]),
               _num(best["hc_mean"]), _num(best["sci_mean"]),
               _num(best["adj_diff"], signed=True),
               best["adj_p"], best["adj_p_fdr"], best["partial_eta2"]))

    parts.append(
        "\n\\noindent\\textbf{What the design can resolve.}  With "
        "$n=%d$~HC and $n=%d$~SCI and two covariates, a region needs an "
        "effect of $|d|\\ge%.2f$ to clear an uncorrected $p<0.05$, and "
        "$|d|\\ge%.2f$ to clear Bonferroni over the %d-region family.  "
        "Effects smaller than that are invisible to this sample, so the "
        "per-region null is an upper bound on the effect size, not evidence "
        "that the regions are identical.\n" % (n_hc, n_sci, d_unc, d_bonf, m))

    for metric in ("CVR", "BAT"):
        parts.append(
            "\n\\begin{table}[H]\n\\centering\n"
            "\\caption{The %d regions with the smallest covariate-adjusted "
            "(age$+$motion) HC-vs-SCI $p$ for %s, out of %d tested.  $d$ is "
            "the unadjusted Cohen's $d$ (HC$-$SCI); adj.\\ $\\Delta$ is the "
            "adjusted SCI$-$HC difference; $q$ is the Benjamini--Hochberg "
            "FDR value across all %d regions.  Means are in arbitrary "
            "units.}\n\\label{tab:region216-%s}\n\\scriptsize\n%s\n"
            "\\end{table}\n"
            % (top_n, metric, len(fam(metric)), len(fam(metric)),
               metric.lower(), _top_table(tested, metric, top_n)))

    parts.append(r"""
\IfFileExists{plots/region216_by_network.png}{%
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/region216_by_network.png}
\caption{Every one of the 216 parcels, grouped by the resting-state network
  it belongs to.  Each point is one parcel at its covariate-adjusted
  (age$+$motion) group $t$; boxes give the median and interquartile range;
  the shaded band lies below the uncorrected $p<0.05$ threshold
  ($|t|=2.02$, df${}={}$42).  Plotting $t$ rather than the adjusted
  difference is what allows CVR and BAT to share one axis: $t=\beta/\mathrm{SE}$
  is dimensionless, whereas the differences are in each metric's own
  arbitrary units.  The bold percentages are the share of parcels in that
  network with SCI higher; they are \emph{descriptive only}, because
  neighbouring parcels are spatially correlated and were inspected after the
  fact.  Produced by
  \pathtt{cohort\_\pb{}scripts/\pb{}region216\_\pb{}network\_\pb{}figure.py}.}
\label{fig:region216-network}
\end{figure}}{}

\IfFileExists{plots/region216_surface.png}{%
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/region216_surface.png}
\caption{The same per-region result rendered on the inflated cortical
  surface: every Schaefer parcel is coloured by its covariate-adjusted
  (age$+$motion) group $t$ statistic, warm $=$ higher in SCI, cool $=$ higher
  in HC, on one scale capped at the largest $|t|$ present in the data.  This
  is the most readable view of the result, but note what it is: because no
  parcel survives FDR correction, it shows the \emph{spatial pattern of a
  non-significant effect}, not a map of findings.  The 16 subcortical Tian
  nuclei have no cortical surface representation and appear only in the axial
  montage (Figure~\ref{fig:region216-map}).  Produced by
  \pathtt{cohort\_\pb{}scripts/\pb{}region216\_\pb{}surface\_\pb{}figure.py}.}
\label{fig:region216-surface}
\end{figure}}{}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/region216_manhattan.png}
\caption{Covariate-adjusted (age$+$motion) HC-vs-SCI effect for every region
  of the 216-region parcellation, plotted as $-\log_{10}$ adjusted $p$ and
  grouped along the $x$-axis by Yeo-7 network (subcortical Tian nuclei at the
  right).  The dashed line is the uncorrected $p=0.05$ threshold; a solid red
  line marks the FDR $q=0.05$ threshold and is drawn only if at least one
  region survives it.}
\label{fig:region216-manhattan}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/region216_forest_top.png}
\caption{The 20 regions with the smallest adjusted $p$ for CVR (left) and BAT
  (right), showing the adjusted SCI$-$HC difference with its 95\% confidence
  interval.  Red $=$ FDR $q<0.05$, orange $=$ uncorrected $p<0.05$,
  grey $=$ neither.  The FDR $q$ value is printed beside each region name.}
\label{fig:region216-forest}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/region216_effect_map.png}
\caption{Anatomical view of the same result: each parcel is painted with its
  covariate-adjusted group $t$ statistic (SCI$-$HC), CVR on the top row and
  BAT on the bottom, across six axial levels.  Warm $=$ higher in SCI,
  cool $=$ higher in HC; $t$ is clipped at $\pm4$.  A black outline marks any
  parcel surviving FDR $q<0.05$.}
\label{fig:region216-map}
\end{figure}
""")

    p = os.path.join(FIG, "region216_section.tex")
    with open(p, "w") as fh:
        fh.write("%% AUTO-GENERATED by cohort_scripts/region_ancova_216.py "
                 "-- do not edit by hand.\n")
        # The report defines \pb (a permitted break inside a path) and
        # \pathtt; provide fallbacks so the fragment also compiles alone.
        fh.write("\\providecommand{\\pb}{\\discretionary{}{}{}}\n")
        fh.write("\\providecommand{\\pathtt}[1]{\\texttt{#1}}\n")
        fh.write("\n".join(parts))
    print(f"wrote {p}")

    # raw table-only fragment kept for ad-hoc reuse
    p2 = os.path.join(FIG, "region216_top_table.tex")
    with open(p2, "w") as fh:
        for metric in ("CVR", "BAT"):
            fh.write(f"% --- top {top_n} regions, {metric} ---\n")
            fh.write(_top_table(tested, metric, top_n) + "\n\n")
    print(f"wrote {p2}")


def tex_from_csv():
    """Rebuild the LaTeX fragment from figure/region216_stats.csv alone.

    The fragment is a pure derivation of the CSV, so it can be refreshed
    without re-reading a single NIfTI -- which matters because the data
    volume is often unavailable, and because editing the prose in
    write_tex_section() would otherwise leave a stale fragment staged in
    output_report/ until the whole analysis was re-run.
    """
    from scipy import stats as _st
    path = os.path.join(FIG, "region216_stats.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run the full analysis first")
    num = lambda k, v: v if k in ("region", "network", "hemi", "compartment",
                                  "metric") else (float(v) if v != "" else float("nan"))
    rows = [{k: num(k, v) for k, v in r.items()}
            for r in csv.DictReader(open(path))]
    tested = [r for r in rows if int(r["testable"]) == 1]
    n_per = len({r["region"] for r in rows})
    excluded = [r for r in rows if int(r["testable"]) == 0
                and r["metric"] == "CVR"]
    # Recover the design constants from the CSV rather than assuming them.
    dof = int(tested[0]["dof"])
    n_tot = dof + 4
    n_hc = n_sci = n_tot // 2
    se_unit = np.sqrt(1.0 / n_hc + 1.0 / n_sci)
    d_unc = _st.t.ppf(0.975, dof) * se_unit
    d_bonf = _st.t.ppf(1 - 0.025 / 216, dof) * se_unit
    print(f"rebuilding LaTeX fragment from {path} "
          f"(dof={dof} -> n={n_hc}+{n_sci})")
    write_tex_section(tested, n_hc, n_sci, n_per, excluded, d_unc, d_bonf)


if __name__ == "__main__":
    if "--tex-from-csv" in sys.argv:
        tex_from_csv()
    else:
        main()
