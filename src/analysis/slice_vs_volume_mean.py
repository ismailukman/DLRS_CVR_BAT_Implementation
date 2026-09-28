#!/usr/bin/env python
r"""Single mid-axial slice vs. through-brain mean, side by side.

Answers a question the existing figures leave ambiguous: the "Group Mean
CVR / BAT" maps are the mean ACROSS SUBJECTS, then displayed as ONE axial
cross-section at the middle slice (k = 45 of 91, MNI z = +18 mm).  They are
NOT averaged through the volume.

This figure puts the two representations next to each other so the
difference is visible:

  * "mid-axial slice (z = +18)"  - exactly what Figures 3, 4 and the
    per-subject appendix show: one cross-section of the group-mean volume.
  * "through-brain mean"         - the same group-mean volume averaged along
    the z axis over in-brain voxels only, i.e. every axial level pooled into
    one image.

Neither is more correct; they answer different questions.  The single slice
preserves the spatial detail of one level, which is what a reader needs to
judge grey/white contrast and ventricular structure.  The through-brain mean
summarises the whole brain but blurs together levels with very different
anatomy, so structures that exist at only a few levels wash out.

Both panels use the report's fixed display ranges, CVR [-0.8, 0.8] and
BAT [-1.2, 1.2] (arbitrary units), with end-point ticks and clip arrows.

Four rows in ONE figure: for each cohort the across-subject mean, then a
single NAMED subject.  The named subject is the medoid -- closest to the
group centroid in (mean CVR, mean BAT) -- which is the same subject the
representative-maps figure shows, so the two figures cannot disagree about
who is "representative".  The subject ID is printed on the figure rather
than left implicit.

Writes figure/slice_vs_volume_mean.png
Needs the data volume mounted.

Run: /opt/anaconda3/envs/cvr_bat/bin/python cohort_scripts/slice_vs_volume_mean.py
"""
import os
import sys
import glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# This script lives in src/analysis, so the project root is two levels up.
PROJ = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJ)

import visualize_group_comparison as V     # display constants + helpers

DATA = os.path.join(PROJ, "data")
OUT = os.path.join(DATA, "output")
FIG = os.path.join(PROJ, "figure")
GROUPS = [("HC_subj", "HC"), ("SCI_subj", "SCI")]

GRID_AFFINE = np.array([[-2., 0., 0., 90.],
                        [0., 2., 0., -126.],
                        [0., 0., 2., -72.],
                        [0., 0., 0., 1.]])


def medoid_subject(gdir):
    """The subject Figure 3 shows: closest to the group centroid in
    (mean CVR, mean BAT).  Same criterion as visualize_group_comparison, so
    the two figures cannot disagree about who is 'representative'."""
    subs = sorted(s for s in os.listdir(os.path.join(OUT, gdir))
                  if os.path.exists(os.path.join(OUT, gdir, s, "DLRS_CVR.nii")))
    cm, bm = [], []
    for sid in subs:
        cvr = V.load_nifti(os.path.join(OUT, gdir, sid, "DLRS_CVR.nii"))
        bat = V.load_nifti(os.path.join(OUT, gdir, sid, "DLRS_BAT.nii"))
        m = V.get_brain_mask(DATA, gdir, sid)
        m = (m > 0.5) if m is not None else (np.abs(cvr) > 0.01)
        cm.append(float(np.nanmean(cvr[m])))
        bm.append(float(np.nanmean(bat[m])))
    return V.representative_subject(subs, cm, bm)


def subject_volume(gdir, sid):
    """One subject's CVR/BAT volumes and their own brain mask."""
    cvr = V.load_nifti(os.path.join(OUT, gdir, sid, "DLRS_CVR.nii"))
    bat = V.load_nifti(os.path.join(OUT, gdir, sid, "DLRS_BAT.nii"))
    m = V.get_brain_mask(DATA, gdir, sid)
    mask = (m > 0.5) if m is not None else (np.abs(cvr) > 0.01)
    return cvr, bat, mask


def group_mean_volume(gdir):
    """Mean DLRS volume across the subjects of one group, plus a group mask."""
    subs = sorted(s for s in os.listdir(os.path.join(OUT, gdir))
                  if os.path.exists(os.path.join(OUT, gdir, s, "DLRS_CVR.nii")))
    cvr, bat, masks = [], [], []
    for sid in subs:
        cvr.append(V.load_nifti(os.path.join(OUT, gdir, sid, "DLRS_CVR.nii")))
        bat.append(V.load_nifti(os.path.join(OUT, gdir, sid, "DLRS_BAT.nii")))
        mp = os.path.join(DATA, gdir, sid, "mask", "brainMask_RS.nii")
        if os.path.exists(mp):
            masks.append(V.load_nifti(mp) > 0.5)
    # A group mask (voxels in every subject's mask), rather than the first
    # subject's mask, which is what the existing group-mean figure uses.
    mask = np.logical_and.reduce(masks) if masks else np.abs(cvr[0]) > 0.01
    return np.mean(cvr, 0), np.mean(bat, 0), mask, len(subs)


def errts_tsd(gdir, sid):
    """Temporal SD of this subject's OWN errts, on the study grid.

    The anatomical reference is derived from the data, not from a template:
    errts.<sid>.r01.fanaticor+tlrc.nii is the fast-ANATICOR residual series
    that the pipeline works from.  Its voxel-wise temporal standard deviation
    shows tissue, vascular and ventricular structure.

    The errts lives on a 3 mm grid (64x76x64), not the 2 mm DLRS grid, so the
    tSD volume is resampled to the study grid before use.  Run 1 alone is
    enough for an anatomical reference and halves the I/O, which matters
    because this volume is on an external disk that drops out.
    """
    import nibabel.processing as niproc
    pat = os.path.join(DATA, gdir, sid, f"errts.{sid}.r01.fanaticor+tlrc.nii")
    if not os.path.exists(pat):
        cand = sorted(glob.glob(os.path.join(DATA, gdir, sid, "errts.*.nii")))
        if not cand:
            return None
        pat = cand[0]
    img = nib.load(pat)
    tsd = np.asarray(img.dataobj).std(axis=-1)
    tsd_img = nib.Nifti1Image(tsd.astype(np.float32), img.affine)
    res = niproc.resample_from_to(tsd_img, ((91, 109, 91), GRID_AFFINE), order=1)
    return res.get_fdata()


def group_errts_tsd(gdir, subs):
    """Across-subject mean of the errts tSD, for the group rows."""
    acc, n = None, 0
    for sid in subs:
        t = errts_tsd(gdir, sid)
        if t is None:
            continue
        acc = t if acc is None else acc + t
        n += 1
        print(f"    errts tSD {n}/{len(subs)} ({sid})", flush=True)
    return None if acc is None else acc / n


def through_brain_mean(vol, mask):
    """Average along z over in-brain voxels only, so empty slices do not dilute."""
    v = np.where(mask, vol, np.nan)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(v, axis=2)
    return out


def main():
    if not os.path.exists(OUT):
        raise SystemExit("data/output unreachable - mount WDPassport2 first")

    z = 91 // 2
    z_mm = 2 * z - 72
    print(f"mid-axial slice: voxel k={z} of 91  ->  MNI z = {z_mm:+d} mm")

    # One figure, four rows: for each cohort the group mean and a single
    # named subject, so group-level and individual-level behaviour can be
    # compared directly instead of across two separate files.
    rows = []
    for gdir, glabel in GROUPS:
        subs = sorted(x for x in os.listdir(os.path.join(OUT, gdir))
                      if os.path.exists(os.path.join(OUT, gdir, x, "DLRS_CVR.nii")))
        cvr_g, bat_g, mask_g, n = group_mean_volume(gdir)
        print(f"  {glabel}: group n={n} - building errts tSD reference")
        anat_g = group_errts_tsd(gdir, subs)
        rows.append((glabel, f"{glabel}\ngroup mean (n={n})", cvr_g, bat_g,
                     mask_g, anat_g))
        sid = medoid_subject(gdir)
        cvr_s, bat_s, mask_s = subject_volume(gdir, sid)
        rows.append((glabel, f"{glabel}\nsubject {sid}", cvr_s, bat_s, mask_s,
                     errts_tsd(gdir, sid)))
        print(f"  {glabel}: representative subject {sid}")

    col_titles = ["Anatomical reference\nerrts temporal SD (z = %+d)" % z_mm,
                  f"CVR\nmid-axial slice (z = {z_mm:+d})", "CVR\nthrough-brain mean",
                  f"BAT\nmid-axial slice (z = {z_mm:+d})", "BAT\nthrough-brain mean"]

    fig, axes = plt.subplots(len(rows), 5, figsize=(17.5, 3.25 * len(rows)))

    for ri, (glabel, row_label, cvr_m, bat_m, mask, anat) in enumerate(rows):
        slice_mask = V.clean_mask_slice(mask[:, :, z])
        proj_mask = mask.any(axis=2)
        if anat is not None:
            a_sl = np.where(slice_mask, anat[:, :, z], np.nan)
            inside = a_sl[np.isfinite(a_sl)]
            lo, hi = (np.percentile(inside, [2, 98]) if inside.size else (0, 1))
        else:
            a_sl, lo, hi = np.where(slice_mask, 0.4, np.nan), 0, 1
        panels = [
            (a_sl, "gray", lo, hi, None, None),
            (np.where(slice_mask, cvr_m[:, :, z], np.nan), V.CVR_CMAP,
             V.CVR_VMIN, V.CVR_VMAX, V.CVR_TICKS, "CVR (a.u.)"),
            (np.where(proj_mask, through_brain_mean(cvr_m, mask), np.nan),
             V.CVR_CMAP, V.CVR_VMIN, V.CVR_VMAX, V.CVR_TICKS, "CVR (a.u.)"),
            (np.where(slice_mask, bat_m[:, :, z], np.nan), V.BAT_CMAP,
             V.BAT_VMIN, V.BAT_VMAX, V.BAT_TICKS, "BAT (a.u.)"),
            (np.where(proj_mask, through_brain_mean(bat_m, mask), np.nan),
             V.BAT_CMAP, V.BAT_VMIN, V.BAT_VMAX, V.BAT_TICKS, "BAT (a.u.)"),
        ]
        for ci, (img, cmap, vmin, vmax, ticks, lab) in enumerate(panels):
            ax = axes[ri, ci]
            im = ax.imshow(img.T, cmap=cmap, origin="lower", vmin=vmin,
                           vmax=vmax, interpolation="bilinear")
            ax.axis("off")
            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=11, fontweight="bold",
                             pad=10)
            if ci == 0:
                ax.text(-0.07, 0.5, row_label, transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=11,
                        fontweight="bold",
                        color=V.HC_COLOR if glabel == "HC" else V.SCI_COLOR)
            # column 0 is the grayscale anatomy: no colour bar
            if ri == len(rows) - 1 and ticks is not None:
                V.style_cbar(fig.colorbar(im, ax=axes[:, ci], fraction=0.030,
                                          pad=0.02, extend="both",
                                          location="bottom"),
                             ticks, lab, fontsize=9)

    fig.suptitle("One mid-axial cross-section versus the whole volume averaged "
                 "along z, at group and single-subject level",
                 fontsize=14, fontweight="bold", y=0.945)
    fig.text(0.5, 0.055,
             "Each pair of rows is one cohort: its across-subject mean, then one "
             "named subject (the medoid, i.e. the subject closest to the group "
             "centroid in mean CVR and mean BAT \u2014 the same subject the "
             "representative-maps figure shows). Every row is the SAME volume "
             f"reduced two ways. Figures 3, 4 and the appendix all show the "
             f"mid-axial slice at z = {z_mm:+d} mm.",
             ha="center", fontsize=9.5, color="0.25", wrap=True)

    out = os.path.join(FIG, "slice_vs_volume_mean.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
