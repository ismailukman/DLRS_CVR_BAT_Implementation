#!/usr/bin/env python
r"""Cortical-surface rendering of the per-region HC-vs-SCI effect.

A readable replacement for the Manhattan plot: instead of 216 dots on an
axis, every Schaefer parcel is painted on an inflated cortical surface with
its covariate-adjusted (age + motion) group t statistic, in the presentation
style of Boerger et al. (2023, Front Hum Neurosci) Figure 1.

IMPORTANT, and stated on the figure itself: no parcel survives FDR
correction, so this shows the SPATIAL PATTERN of a non-significant effect.
It is an exploratory map, not a findings map.  The colour scale is capped at
the largest |t| actually present so the map cannot imply more than the data
contain, and the "no parcel survives FDR" banner is drawn unconditionally
whenever that is true.

The 16 Tian subcortical nuclei have no cortical surface representation and
are therefore absent here; they appear in the axial montage
(region216_effect_map.png) and in region216_stats.csv.

Reads   figure/region216_stats.csv   (written by region_ancova_216.py)
Writes  figure/region216_surface.png

Needs no access to the external data volume: the statistics come from the
CSV and the parcellation from preprocessing/atlas/, both on the internal
disk.  fsaverage is downloaded once by nilearn and cached in ~/nilearn_data.

Run: /opt/anaconda3/envs/cvr_bat/bin/python cohort_scripts/region216_surface_figure.py
"""
import os
import sys
import csv
import numpy as np
import nibabel as nib
import nibabel.processing as niproc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

HERE = os.path.dirname(os.path.abspath(__file__))
# This script lives in src/analysis, so the project root is two levels up.
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJ)

FIG = os.path.join(PROJ, "figure")
ATLAS = os.path.join(PROJ, "preprocessing", "atlas")
SCHAEFER = os.path.join(ATLAS, "schaefer_2018", "schaefer_2018",
                        "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz")
SCHAEFER_TXT = os.path.join(ATLAS, "schaefer_2018", "schaefer_2018",
                            "Schaefer2018_200Parcels_7Networks_order.txt")

# The study grid, verified identical to standard FSL-MNI152 2 mm, so this
# script does not need a subject mask (and therefore not the data volume).
GRID_SHAPE = (91, 109, 91)
GRID_AFFINE = np.array([[-2., 0., 0., 90.],
                        [0., 2., 0., -126.],
                        [0., 0., 2., -72.],
                        [0., 0., 0., 1.]])


def load_stats():
    """Per-parcel adjusted t, keyed by Schaefer parcel name, per metric."""
    rows = [r for r in csv.DictReader(open(os.path.join(FIG, "region216_stats.csv")))
            if r["testable"] == "1" and r["compartment"] == "cortex"]
    if not rows:
        raise RuntimeError("no testable cortical rows in region216_stats.csv")
    out = {}
    for m in ("CVR", "BAT"):
        sel = [r for r in rows if r["metric"] == m]
        out[m] = {r["region"]: float(r["adj_t"]) for r in sel}
        out[m + "_q"] = {r["region"]: float(r["adj_p_fdr"]) for r in sel}
    return out


def load_dof():
    """Residual dof of the ANCOVA, read from the CSV rather than assumed."""
    for r in csv.DictReader(open(os.path.join(FIG, "region216_stats.csv"))):
        if r["testable"] == "1":
            return float(r["dof"])
    raise RuntimeError("no testable row in region216_stats.csv")


def parcel_id_by_name():
    ids = {}
    with open(SCHAEFER_TXT) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 2:
                ids[p[1].replace("7Networks_", "")] = int(p[0])
    return ids


def t_volume(tmap, ids):
    """Paint each Schaefer parcel with its adjusted t on the study grid."""
    src = nib.load(SCHAEFER)
    res = niproc.resample_from_to(src, (GRID_SHAPE, GRID_AFFINE), order=0)
    parc = np.rint(res.get_fdata()).astype(int)
    vol = np.full(parc.shape, np.nan, dtype=float)
    painted = 0
    for name, t in tmap.items():
        pid = ids.get(name)
        if pid is None:
            continue
        sel = parc == pid
        if sel.any():
            vol[sel] = t
            painted += 1
    print(f"  painted {painted}/{len(tmap)} parcels")
    return nib.Nifti1Image(np.nan_to_num(vol, nan=0.0), GRID_AFFINE), vol


def main():
    from nilearn import datasets, surface

    stats = load_stats()
    ids = parcel_id_by_name()

    n_sig = {m: sum(q < 0.05 for q in stats[m + "_q"].values()) for m in ("CVR", "BAT")}
    dof = int(load_dof())
    print(f"cortical parcels surviving FDR: CVR {n_sig['CVR']}, BAT {n_sig['BAT']}")

    print("fetching fsaverage surfaces (cached in ~/nilearn_data after the first run)")
    fsa = datasets.fetch_surf_fsaverage("fsaverage")

    # One symmetric colour scale for BOTH metrics.  This is legitimate only
    # because the mapped quantity is the t statistic (beta/SE), which is
    # dimensionless: CVR and BAT have different native ranges (displayed at
    # +/-0.8 and +/-1.2 a.u.) and their adjusted DIFFERENCES could NOT share a
    # colour bar, but their t values can, and both metrics here have the same
    # dof so equal t means equal p.  The scale is capped at the largest |t|
    # actually observed so the map cannot suggest a stronger effect than exists.
    tmax = max(max(abs(v) for v in stats[m].values()) for m in ("CVR", "BAT"))
    tmax = float(np.ceil(tmax * 10) / 10)
    print(f"colour scale: +/-{tmax:g} (largest |adjusted t| in the data)")
    norm = TwoSlopeNorm(vmin=-tmax, vcenter=0.0, vmax=tmax)
    cmap = "coolwarm"

    views = [("left", "lateral"), ("left", "medial"),
             ("right", "lateral"), ("right", "medial")]
    col_title = ["Left lateral", "Left medial", "Right lateral", "Right medial"]

    fig = plt.figure(figsize=(16, 6.6))
    gs = fig.add_gridspec(2, 4, wspace=-0.02, hspace=-0.14,
                          left=0.03, right=0.88, top=0.93, bottom=-0.02)

    for ri, metric in enumerate(("CVR", "BAT")):
        img, _ = t_volume(stats[metric], ids)
        tex = {}
        for hemi in ("left", "right"):
            mesh = fsa[f"pial_{hemi}"]
            # nearest-neighbour: this is a parcellated map, so no interpolation
            # across parcel boundaries.
            tex[hemi] = surface.vol_to_surf(img, mesh, interpolation="nearest")

        for ci, (hemi, view) in enumerate(views):
            ax = fig.add_subplot(gs[ri, ci], projection="3d")
            from nilearn import plotting
            plotting.plot_surf_stat_map(
                fsa[f"infl_{hemi}"], tex[hemi], hemi=hemi, view=view,
                colorbar=False, cmap=cmap, vmax=tmax, threshold=None,
                bg_map=fsa[f"sulc_{hemi}"], bg_on_data=True, darkness=0.5,
                axes=ax, figure=fig)
            ax.set_title(col_title[ci] if ri == 0 else "", fontsize=11,
                         fontweight="bold", pad=0)
            if ci == 0:
                ax.text2D(-0.02, 0.5, metric, transform=ax.transAxes,
                          rotation=90, va="center", ha="center",
                          fontsize=15, fontweight="bold")

    cax = fig.add_axes([0.90, 0.30, 0.013, 0.40])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks([-tmax, -tmax / 2, 0, tmax / 2, tmax])
    cb.set_ticklabels([f"{v:g}" for v in (-tmax, -tmax / 2, 0, tmax / 2, tmax)])
    cb.set_label("adjusted $t$  (SCI $-$ HC)", fontsize=11)
    cb.ax.tick_params(labelsize=9)
    # Spell out why one scale serves both rows, because "CVR and BAT have
    # different ranges" is the first objection a reader raises.
    fig.text(0.955, 0.50, "$t=\\beta/\\mathrm{SE}$ is dimensionless,\n"
             "so CVR and BAT share one\nscale (both df $=$ %d)" % dof,
             fontsize=8.5, color="0.30", rotation=90, va="center", ha="center")
    fig.text(0.90, 0.715, "SCI higher", fontsize=9, color="#8B1A1A", ha="left")
    fig.text(0.90, 0.275, "HC higher", fontsize=9, color="#1A3D8B", ha="left")

    fig.suptitle("Per-region HC-vs-SCI effect on the cortical surface "
                 "(Schaefer-200 parcels, age- and motion-adjusted)",
                 fontsize=14, fontweight="bold", y=0.99)

    if n_sig["CVR"] == 0 and n_sig["BAT"] == 0:
        banner = ("No parcel survives FDR correction (q < 0.05) for either metric "
                  f" -  largest |t| = {tmax:g}. "
                  "This is the spatial pattern of a non-significant effect, not a findings map.")
        colour = "#8B1A1A"
    else:
        banner = (f"Parcels surviving FDR q < 0.05: CVR {n_sig['CVR']}, "
                  f"BAT {n_sig['BAT']}.")
        colour = "#1A5276"
    fig.text(0.5, 0.055, banner, ha="center", fontsize=11, color=colour,
             fontweight="bold", wrap=True)
    fig.text(0.5, 0.018, "Subcortical Tian S1 nuclei have no cortical surface "
             "representation and are not shown; see the axial montage.",
             ha="center", fontsize=8.5, color="0.35")

    out = os.path.join(FIG, "region216_surface.png")
    # 300 dpi on a 16-inch canvas is ~4800 px; embedded at 6.4 inches in the
    # abstract that is ~750 dpi effective, comfortably above the 600 dpi
    # target, without the 9 MB docx that dpi=600 produced.
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
