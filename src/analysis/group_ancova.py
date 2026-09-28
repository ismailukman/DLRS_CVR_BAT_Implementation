#!/usr/bin/env python
"""Covariate-adjusted (ANCOVA) HC-vs-SCI group comparison of DLRS CVR/BAT.

For every metric x region we fit an ordinary-least-squares linear model

        value ~ intercept + group + age + motion

where group is SCI=1 / HC=0, age is years, and motion is the preprocessing
censor fraction (%).  We report the covariate-ADJUSTED group effect
(coefficient = SCI-HC, t, p, partial eta^2) alongside the UNADJUSTED Welch
t-test and Cohen's d, and Benjamini-Hochberg FDR within each region family.

Sex is not a covariate: both groups are identically matched (18M/5F).

Regions:
  - whole-brain (brain mask)
  - Yeo-7 cortical networks (Schaefer-200 -> 7)         [analyze_network_atlases]
  - Tian S1 subcortex, 7 bilateral structures          [analyze_network_atlases]
  - MCA territory total (Neuromorphometrics labels)     [model_result_group]

Outputs: figure/ancova_group_stats.csv, figure/ancova_per_subject.csv,
         figure/ancova_forest.png
Run:  /opt/anaconda3/envs/cvr_bat/bin/python cohort_scripts/group_ancova.py
"""
import os, sys, csv
import numpy as np
import nibabel as nib
from scipy import stats
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# This script lives in src/analysis, so the project root is two levels up.
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJ)
import analyze_network_atlases as ana          # atlas builders + constants
import model_result_group as mrg               # MCA label sets

DATA, OUT, FIG = ana.DATA, ana.OUT, ana.FIG
COHORT_XLSX = "/Volumes/WDPassport2/A-fMRIData/sci_project/sci_data/DLRS_HC_SCI_study_cohort.xlsx"
NEURO = os.path.join(PROJ, "preprocessing", "atlas",
                     "labels_Neuromorphometrics", "wlabels_Neuromorphometrics_unique.nii")

# ----------------------------------------------------------- demographics
def load_demo():
    wb = openpyxl.load_workbook(COHORT_XLSX, data_only=True)
    ws = wb["Study_max2run"]
    rows = list(ws.iter_rows(values_only=True))
    hdr_i = next(i for i, r in enumerate(rows)
                 if r and "Group" in r and "Subject ID" in r)
    hdr = rows[hdr_i]
    ix = {name: j for j, name in enumerate(hdr) if name}
    demo = {}
    for r in rows[hdr_i + 1:]:
        g = r[ix["Group"]]; sid = r[ix["Subject ID"]]
        age = r[ix["Age (yr)"]]; cf = r[ix["Censor frac (%)"]]
        if g in ("HC", "SCI") and sid is not None and isinstance(age, (int, float)):
            demo[str(sid)] = {"group": g, "age": float(age),
                              "motion": float(cf) if isinstance(cf, (int, float)) else np.nan}
    return demo

# ----------------------------------------------------------- region masks
def build_region_masks():
    shape, affine = ana.ref_grid()
    netvol, net_names = ana.build_cortex_networkvol(shape, affine)   # 1..7 Yeo
    subvol, sub_names = ana.build_subcortex_vol(shape, affine)       # 1..7 Tian
    atlas = np.rint(ana.L(NEURO)).astype(int)
    if atlas.shape != shape:                      # align if grids differ
        atlas = ana.resample_labels(NEURO, shape, affine)
    mca = np.isin(atlas, mrg.MCA_LABELS_ALL)
    regions = [("whole-brain", "whole-brain", None)]
    for k, nm in enumerate(net_names, 1):
        regions.append(("cortex", nm, netvol == k))
    for k, nm in enumerate(sub_names, 1):
        regions.append(("subcortex", nm, subvol == k))
    regions.append(("mca", "MCA-total", mca))
    return regions, shape

# ----------------------------------------------------------- per-subject means
def brain_mask(gdir, sid):
    mp = os.path.join(DATA, gdir, sid, "mask", "brainMask_RS.nii")
    return ana.L(mp) > 0.5 if os.path.exists(mp) else None

def subject_region_means(gdir, sid, regions):
    cvr = ana.L(os.path.join(OUT, gdir, sid, "DLRS_CVR.nii"))
    bat = ana.L(os.path.join(OUT, gdir, sid, "DLRS_BAT.nii"))
    bm = brain_mask(gdir, sid)
    if bm is None:
        bm = np.abs(cvr) > 0.01
    out = {}
    for comp, name, rmask in regions:
        m = bm if rmask is None else (rmask & bm)
        out[(name, "CVR")] = float(np.nanmean(cvr[m])) if m.sum() else np.nan
        out[(name, "BAT")] = float(np.nanmean(bat[m])) if m.sum() else np.nan
    return out

# ----------------------------------------------------------- statistics
def ancova(y, grp, age, mot):
    """OLS: y ~ 1 + grp + age + mot.  Return adjusted group beta, t, p, partial eta^2."""
    keep = ~(np.isnan(y) | np.isnan(age) | np.isnan(mot))
    y, grp, age, mot = y[keep], grp[keep], age[keep], mot[keep]
    n = len(y)
    X = np.column_stack([np.ones(n), grp, age, mot])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - X.shape[1]
    mse = float(resid @ resid) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * mse)
    t = beta[1] / se[1]
    p = 2 * stats.t.sf(abs(t), dof)
    peta2 = t**2 / (t**2 + dof)
    return beta[1], se[1], t, p, peta2, dof

def welch(hc, sci):
    hc, sci = np.asarray(hc), np.asarray(sci)
    hc, sci = hc[~np.isnan(hc)], sci[~np.isnan(sci)]
    t, p = stats.ttest_ind(hc, sci, equal_var=False)
    n1, n2 = len(hc), len(sci)
    sp = np.sqrt(((n1-1)*hc.var(ddof=1) + (n2-1)*sci.var(ddof=1)) / (n1+n2-2))
    d = (hc.mean() - sci.mean()) / sp if sp > 0 else 0.0     # HC - SCI
    return hc.mean(), sci.mean(), t, p, d

def bh_fdr(pvals):
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for i in range(n-1, -1, -1):
        idx = order[i]; prev = min(prev, p[idx]*n/(i+1)); q[idx] = prev
    return q

# ----------------------------------------------------------- main
def main():
    demo = load_demo()
    regions, shape = build_region_masks()
    print(f"grid={shape}  regions={len(regions)}  subjects with demo={len(demo)}")

    subjects = []   # (gdir, sid, group, age, motion)
    for gdir, glabel in [("HC_subj", "HC"), ("SCI_subj", "SCI")]:
        for sid in sorted(os.listdir(os.path.join(OUT, gdir))):
            if not os.path.isdir(os.path.join(OUT, gdir, sid)):
                continue
            if sid not in demo:
                print(f"  WARN no demo for {sid}, skipped"); continue
            subjects.append((gdir, sid, demo[sid]["group"], demo[sid]["age"], demo[sid]["motion"]))
    print(f"analysed: {sum(s[2]=='HC' for s in subjects)} HC, {sum(s[2]=='SCI' for s in subjects)} SCI")

    # per-subject region means
    permeans = {}
    for gdir, sid, *_ in subjects:
        permeans[sid] = subject_region_means(gdir, sid, regions)

    grp = np.array([1.0 if s[2] == "SCI" else 0.0 for s in subjects])
    age = np.array([s[3] for s in subjects])
    mot = np.array([s[4] for s in subjects])

    # per-subject CSV
    with open(os.path.join(FIG, "ancova_per_subject.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        cols = [f"{n}|{m}" for _, n, _ in regions for m in ("CVR", "BAT")]
        w.writerow(["group", "subject", "age", "motion"] + cols)
        for gdir, sid, g, a, mo in subjects:
            w.writerow([g, sid, a, mo] + [permeans[sid][(n, m)] for _, n, _ in regions for m in ("CVR", "BAT")])

    # stats per metric x region, FDR within family = (compartment, metric)
    rows = []
    for metric in ("CVR", "BAT"):
        for comp, name, _ in regions:
            vals = np.array([permeans[s[1]][(name, metric)] for s in subjects])
            hc_v = vals[grp == 0]; sci_v = vals[grp == 1]
            hc_m, sci_m, ut, up, ud = welch(hc_v, sci_v)
            adiff, ase, at, ap, peta2, dof = ancova(vals, grp, age, mot)
            rows.append(dict(compartment=comp, region=name, metric=metric,
                             hc_mean=hc_m, sci_mean=sci_m,
                             unadj_t=ut, unadj_p=up, cohen_d=ud,
                             adj_diff=adiff, adj_t=at, adj_p=ap, partial_eta2=peta2, dof=dof))
    # FDR within each (compartment, metric) family that has >1 region
    from collections import defaultdict
    fam = defaultdict(list)
    for i, r in enumerate(rows):
        fam[(r["compartment"], r["metric"])].append(i)
    for key, idxs in fam.items():
        if len(idxs) > 1:
            q_un = bh_fdr([rows[i]["unadj_p"] for i in idxs])
            q_adj = bh_fdr([rows[i]["adj_p"] for i in idxs])
            for j, i in enumerate(idxs):
                rows[i]["unadj_p_fdr"] = q_un[j]; rows[i]["adj_p_fdr"] = q_adj[j]
        else:
            rows[idxs[0]]["unadj_p_fdr"] = rows[idxs[0]]["unadj_p"]
            rows[idxs[0]]["adj_p_fdr"] = rows[idxs[0]]["adj_p"]

    # write CSV
    fields = ["compartment", "region", "metric", "hc_mean", "sci_mean",
              "unadj_t", "unadj_p", "unadj_p_fdr", "cohen_d",
              "adj_diff", "adj_t", "adj_p", "adj_p_fdr", "partial_eta2", "dof"]
    outcsv = os.path.join(FIG, "ancova_group_stats.csv")
    with open(outcsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {outcsv}")

    # console summary
    print(f"\n{'region':22}{'metric':5}{'HC':>8}{'SCI':>8}{'d':>6}{'unadjP':>8}{'adjP':>8}{'adjFDR':>8}{'pEta2':>7}")
    for r in sorted(rows, key=lambda r: r["adj_p"]):
        print(f"{r['region']:22}{r['metric']:5}{r['hc_mean']:8.3f}{r['sci_mean']:8.3f}"
              f"{r['cohen_d']:6.2f}{r['unadj_p']:8.3f}{r['adj_p']:8.3f}{r['adj_p_fdr']:8.3f}{r['partial_eta2']:7.3f}")

    # forest plot: adjusted group effect (SCI-HC) per region, per metric
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    for ax, metric in zip(axes, ("CVR", "BAT")):
        rr = [r for r in rows if r["metric"] == metric and r["region"] != "whole-brain"]
        rr = sorted(rr, key=lambda r: (r["compartment"], r["region"]))
        y = np.arange(len(rr))
        diffs = [r["adj_diff"] for r in rr]
        ses = [r["adj_diff"]/r["adj_t"] if r["adj_t"] != 0 else 0 for r in rr]
        cols = ["#C0392B" if r["adj_p"] < 0.05 else ("#E67E22" if r["adj_p"] < 0.1 else "#7F8C8D") for r in rr]
        ax.errorbar(diffs, y, xerr=[1.96*abs(s) for s in ses], fmt="o", ecolor="#bbb", elinewidth=1, capsize=2, mfc="w", mec="w", zorder=1)
        ax.scatter(diffs, y, c=cols, s=45, zorder=2)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_yticks(y); ax.set_yticklabels([f"{r['compartment'][:4]}:{r['region']}" for r in rr], fontsize=8)
        ax.set_title(f"Adjusted group effect (SCI-HC), {metric}\n(red p<.05, orange p<.10; 95% CI)", fontsize=10)
        ax.set_xlabel(f"adjusted {metric} difference")
        ax.invert_yaxis()
    fig.suptitle("Covariate-adjusted (age+motion) HC-vs-SCI regional effects, n=23/group",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fp = os.path.join(FIG, "ancova_forest.png")
    fig.savefig(fp, dpi=600, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {fp}")

if __name__ == "__main__":
    main()
