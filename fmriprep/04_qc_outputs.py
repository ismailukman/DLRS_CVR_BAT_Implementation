#!/usr/bin/env python
"""QC of fMRIPrep outputs, per run: atlas grid, TR and volumes, brain-mask
overlap with the atlas, cerebellum coverage, low-frequency power, tSNR and
motion. Writes qc_summary.csv.

    python fmriprep/04_qc_outputs.py [label ...]
"""
import os
import re
import sys
import csv
import glob
import numpy as np
import nibabel as nib

OUT = "/Volumes/WDPassport2/A-fMRIData/sci_project/01_DATA/04_DERIVATIVES/01_PREPROC/pipeline-fmriprep/v001"
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
ATLAS = os.path.join(PROJ, "preprocessing", "atlas", "labels_Neuromorphometrics",
                     "wlabels_Neuromorphometrics_unique.nii")
CEREB = [10, 11, 12, 13, 31, 32, 33]


def to_atlas_layout(img, ref):
    """Reindex img onto ref's voxel grid. Requires identical voxel centres."""
    ia = np.array(np.meshgrid(*[np.arange(s) for s in ref.shape[:3]], indexing="ij")).reshape(3, -1).T
    v = nib.affines.apply_affine(np.linalg.inv(img.affine) @ ref.affine, ia)
    vr = np.round(v).astype(int)
    if not np.allclose(v, vr, atol=1e-3):
        raise ValueError("grids do not share voxel centres")
    data = np.asanyarray(img.dataobj)
    out = data[vr[:, 0], vr[:, 1], vr[:, 2], ...]
    return out.reshape(ref.shape[:3] + data.shape[3:])


def lowfreq_fraction(ts, tr):
    ts = ts - ts.mean()
    f = np.fft.rfftfreq(len(ts), tr)
    p = np.abs(np.fft.rfft(ts)) ** 2
    band = (f > 0.008) & (f <= 0.25)
    return p[(f > 0.008) & (f < 0.1)].sum() / p[band].sum()


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def qc_run(bold_path, atlas, lab):
    img = nib.load(bold_path)
    row = {"run": os.path.basename(bold_path).split("_space")[0]}
    try:
        d = to_atlas_layout(img, atlas).astype(np.float32)
        row["grid"] = "ok (lossless reindex)"
    except ValueError as e:
        row["grid"] = f"FAIL {e}"
        return row
    tr = float(img.header.get_zooms()[3])
    row["TR"] = round(tr, 3)
    row["vols"] = d.shape[3]
    mpath = bold_path.replace("desc-preproc_bold", "desc-brain_mask")
    bm = to_atlas_layout(nib.load(mpath), atlas) > 0.5 if os.path.exists(mpath) else d.mean(3) > 0
    row["mask_dice_vs_atlas"] = round(float(dice(bm, lab > 0)), 3)
    cer = np.isin(lab, CEREB)
    row["cerebellum_in_mask_%"] = round(100 * float((cer & bm).sum() / cer.sum()), 1)
    gs = d[bm].mean(axis=0)
    row["lowfreq_power_%"] = round(100 * float(lowfreq_fraction(gs, tr)), 1)
    cs = d[cer & bm].mean(axis=0)
    row["cereb_lowfreq_%"] = round(100 * float(lowfreq_fraction(cs, tr)), 1)
    m = d.mean(axis=3)
    row["tSNR_median"] = round(float(np.median((m / (d.std(axis=3) + 1e-6))[bm])), 1)
    cpath = re.sub(r"_space-.*", "_desc-confounds_timeseries.tsv", bold_path)
    if os.path.exists(cpath):
        rows = list(csv.reader(open(cpath), delimiter="\t"))
        hdr, vals = rows[0], rows[1:]
        fd = np.array([float(r[hdr.index("framewise_displacement")]) for r in vals[1:]
                       if r[hdr.index("framewise_displacement")] not in ("n/a", "")])
        row["meanFD_mm"] = round(float(fd.mean()), 3)
        row["maxFD_mm"] = round(float(fd.max()), 2)
        row["FD>0.5_%"] = round(100 * float((fd > 0.5).mean()), 1)
        row["nonsteady_vols"] = sum(1 for h in hdr if h.startswith("non_steady_state_outlier"))
        row["motion_cols"] = sum(1 for h in hdr if h in ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"))
    return row


def main():
    atlas = nib.load(ATLAS)
    lab = np.rint(np.asanyarray(atlas.dataobj)).astype(int)
    subs = sys.argv[1:] or sorted(re.sub(r"sub-|\.html", "", os.path.basename(h))
                                  for h in glob.glob(os.path.join(OUT, "sub-*.html")))
    allrows = []
    for sub in subs:
        html = os.path.join(OUT, f"sub-{sub}.html")
        rep = open(html, errors="replace").read() if os.path.exists(html) else ""
        err = "no report" if not rep else ("errors reported" if re.search(r"crashfile|Traceback", rep) else "clean")
        print(f"\n=== sub-{sub}   report: {err}")
        bolds = sorted(glob.glob(os.path.join(OUT, f"sub-{sub}", "ses-*", "func",
                                              "*space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz")))
        for b in bolds:
            row = qc_run(b, atlas, lab)
            row["subject"] = sub
            row["report"] = err
            allrows.append(row)
            for k, v in row.items():
                if k not in ("subject", "report"):
                    print(f"    {k:22s} {v}")
    out = os.path.join(HERE, "qc_summary.csv")
    if allrows and sys.argv[1:] and os.path.exists(out):
        # a subset run replaces only its own participants' rows
        keep = [r for r in csv.DictReader(open(out)) if r["subject"] not in subs]
        allrows = keep + allrows
    if allrows:
        keys = []
        for r in allrows:
            keys += [k for k in r if k not in keys]
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(allrows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
