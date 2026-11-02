#!/usr/bin/env python
"""Write the MATLAB input for each participant from the fMRIPrep output:

    func_run_combined.nii   selected runs on the atlas grid, concatenated
    rp_fmriprep.txt         six motion parameters (mm, rad)
    censor_fmriprep.txt     1 = keep, 0 = censored
    parameter_RS.txt        settings read by MATLAB
    run_info.json           runs used and censored volumes
    censor_details.tsv      per volume: motion, outlier fraction, censored or not
    motion_censoring.png    motion and outlier fraction per run, censored volumes shaded

Runs: visit 1 runs 1 and 2; Kirby21 one run per visit; run 1 only for sci28,
sci31 and sci37; run 2 only for sci12 and sci36. Each run is linearly
detrended per voxel before concatenation.

Censoring, as afni_proc (-regress_censor_motion 0.4 -regress_censor_outliers 0.1):
a volume is censored when the norm of the motion change from the previous
volume (mm and degrees) exceeds 0.4 (the volume before it is censored too), or
when more than 10% of brain voxels are outliers (3dToutcount on the raw run).

    python fmriprep/06_prepare_dlrs_input.py [label ...]
    python fmriprep/06_prepare_dlrs_input.py --censor-only [label ...]   (censor list and run_info only)
"""
import os
import sys
import shutil
import subprocess
import csv
import glob
import json
import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
FMRIPREP = "/Volumes/WDPassport2/A-fMRIData/sci_project/01_DATA/04_DERIVATIVES/01_PREPROC/pipeline-fmriprep/v001"
OUT = "/Volumes/WDPassport2/A-fMRIData/sci_project/01_DATA/04_DERIVATIVES/02_DLRS_INPUT/fmriprep"
ATLAS = os.path.join(PROJ, "preprocessing", "atlas", "labels_Neuromorphometrics",
                     "wlabels_Neuromorphometrics_unique.nii")
SPACE = "space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
CENSOR_LIMIT = 0.4
OUTLIER_LIMIT = 0.1
BIDS = "/Volumes/WDPassport2/A-fMRIData/sci_project/01_DATA/01_RAW/bids"
TOUTCOUNT = shutil.which("3dToutcount") or "3dToutcount"   # AFNI
MOTION = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]

# (session, run) exceptions to the default run selection
RUNS = {
    "sci12": [("1", "2")],
    "sci28": [("1", "1")],   # workbook: only run 1
    "sci31": [("1", "1")],
    "sci36": [("1", "2")],   # workbook: only run 2
    "sci37": [("1", "1")],   # workbook: only run 1
}

sys.path.insert(0, HERE)
from importlib import import_module
to_atlas_layout = import_module("04_qc_outputs").to_atlas_layout


def bold_path(lab, ses, run):
    return os.path.join(FMRIPREP, f"sub-{lab}", f"ses-{ses}", "func",
                        f"sub-{lab}_ses-{ses}_task-rest_run-{run}_{SPACE}")


def select_runs(lab):
    if lab in RUNS:
        return RUNS[lab]
    found = sorted(glob.glob(os.path.join(FMRIPREP, f"sub-{lab}", "ses-*", "func", f"*_run-*_{SPACE}")))
    keyed = []
    for f in found:
        b = os.path.basename(f)
        keyed.append((b.split("_ses-")[1].split("_")[0], b.split("_run-")[1].split("_")[0]))
    v1 = sorted([k for k in keyed if k[0] == "1"], key=lambda k: int(k[1]))
    if len(v1) >= 2:
        return v1[:2]
    other = sorted([k for k in keyed if k[0] != "1"], key=lambda k: (int(k[0]), int(k[1])))
    return (v1 + other)[:2]


def detrend_linear(y):
    """Remove a least-squares line along axis 0; returns residuals."""
    n = y.shape[0]
    X = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
    beta = np.linalg.lstsq(X, y.reshape(n, -1), rcond=None)[0]
    return (y.reshape(n, -1) - X @ beta).reshape(y.shape)


def censor_mask(motion):
    """Censor where the norm of the motion change (mm, degrees) exceeds the limit, and the volume before."""
    m = motion.copy()
    m[:, 3:] = np.degrees(m[:, 3:])
    d = np.vstack([np.zeros((1, 6)), np.diff(m, axis=0)])
    enorm = np.sqrt((d ** 2).sum(axis=1))
    bad = enorm > CENSOR_LIMIT
    bad[:-1] |= bad[1:]
    return ~bad, enorm


def outlier_keep(lab, ses, run, n, tr):
    """afni_proc outlier count on the raw run: 3dToutcount -automask -fraction -polort P -legendre."""
    raw = os.path.join(BIDS, f"sub-{lab}", f"ses-{ses}", "func", f"sub-{lab}_ses-{ses}_task-rest_run-{run}_bold.nii.gz")
    polort = 1 + int(n * tr / 150)
    out = subprocess.run([TOUTCOUNT, "-automask", "-fraction", "-polort", str(polort), "-legendre", raw],
                         capture_output=True, text=True, check=True).stdout
    frac = np.array([float(x) for x in out.split()])
    if frac.size != n:
        raise ValueError(f"{lab} ses-{ses} run-{run}: 3dToutcount gave {frac.size} values for {n} volumes")
    return frac <= OUTLIER_LIMIT, frac


def write_censor_details(odir, lab, detail):
    with open(os.path.join(odir, "censor_details.tsv"), "w") as fh:
        fh.write("visit\trun\tvolume\tmotion_mm\toutlier_fraction\tcensored_motion\tcensored_outlier\tcensored\n")
        for ses, run, en, fr, km, ko in detail:
            for v in range(len(en)):
                fh.write(f"{ses}\t{run}\t{v + 1}\t{en[v]:.3f}\t{fr[v]:.4f}\t{int(not km[v])}\t{int(not ko[v])}\t{int(not (km[v] and ko[v]))}\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(detail), figsize=(6.5 * len(detail), 5), squeeze=False, sharex="col")
    for c, (ses, run, en, fr, km, ko) in enumerate(detail):
        t = np.arange(1, len(en) + 1)
        bad = ~(km & ko)
        for r, (y, lim, lab_y) in enumerate(((en, CENSOR_LIMIT, "Motion between volumes (mm)"),
                                             (fr, OUTLIER_LIMIT, "Outlier fraction"))):
            ax = axes[r, c]
            for v in t[bad]:
                ax.axvspan(v - 0.5, v + 0.5, color="#F4CCCC", lw=0)
            ax.plot(t, y, color="#1F3864", lw=1)
            ax.axhline(lim, color="#C00000", ls="--", lw=1)
            ax.set_ylabel(lab_y)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0, c].set_title(f"{lab}  visit {ses} run {run}: {int(bad.sum())} of {len(en)} volumes censored")
        axes[1, c].set_xlabel("Volume")
    fig.text(0.01, 0.005, "Red dashed line: threshold (0.4 mm motion, 0.1 outlier fraction). Shaded: censored volumes.",
             fontsize=8, color="#595959")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(odir, "motion_censoring.png"), dpi=110)
    plt.close(fig)


def prepare(lab, atlas, censor_only=False):
    runs = select_runs(lab)
    if not runs:
        return {"label": lab, "status": "no fMRIPrep output"}
    group = "HC_subj" if lab.startswith("hc") else "SCI_subj"
    odir = os.path.join(OUT, group, lab)
    os.makedirs(odir, exist_ok=True)

    data, motion, keep, info, detail = [], [], [], [], []
    for ses, run in runs:
        p = bold_path(lab, ses, run)
        tr = float(json.load(open(p.replace(".nii.gz", ".json")))["RepetitionTime"])
        if abs(tr - 2.0) > 1e-3:
            raise ValueError(f"{lab} ses-{ses} run-{run}: TR {tr} s, expected 2 s")
        if not censor_only:
            d = to_atlas_layout(nib.load(p), atlas).astype(np.float32)   # x, y, z, t
            d = np.moveaxis(d, -1, 0)                                    # t, x, y, z
            mean = d.mean(axis=0)
            data.append((detrend_linear(d).astype(np.float32), mean))
        conf = os.path.join(FMRIPREP, f"sub-{lab}", f"ses-{ses}", "func",
                            f"sub-{lab}_ses-{ses}_task-rest_run-{run}_desc-confounds_timeseries.tsv")
        rows = list(csv.DictReader(open(conf), delimiter="\t"))
        mot = np.array([[float(r[k]) for k in MOTION] for r in rows])
        fd = np.array([float(r["framewise_displacement"]) if r["framewise_displacement"] != "n/a" else 0.0 for r in rows])
        km, enorm = censor_mask(mot)
        ko, frac = outlier_keep(lab, ses, run, len(rows), tr)
        k = km & ko
        detail.append((ses, run, enorm, frac, km, ko))
        motion.append(detrend_linear(mot))
        keep.append(k)
        info.append({"session": ses, "run": run, "volumes": len(rows),
                     "censored": int((~k).sum()), "censored_motion": int((~km).sum()),
                     "censored_outliers": int((~ko).sum()), "mean_FD_mm": round(float(fd[1:].mean()), 3),
                     "max_enorm": round(float(enorm.max()), 3), "max_outlier_frac": round(float(frac.max()), 3)})

    keep = np.concatenate(keep)
    np.savetxt(os.path.join(odir, "censor_fmriprep.txt"), keep.astype(int), fmt="%d")
    summary = {"label": lab, "group": group[:-5], "runs": info, "volumes": int(keep.size),
               "censored": int((~keep).sum()), "censor_frac_pct": round(100 * float((~keep).mean()), 1),
               "censored_motion": sum(r["censored_motion"] for r in info),
               "censored_outliers": sum(r["censored_outliers"] for r in info),
               "censor_rule": f"motion > {CENSOR_LIMIT} (mm, deg) plus previous volume; outlier fraction > {OUTLIER_LIMIT}",
               "status": "ok"}
    json.dump(summary, open(os.path.join(odir, "run_info.json"), "w"), indent=2)
    write_censor_details(odir, lab, detail)
    if censor_only:
        return summary

    grand_mean = np.mean([m for _, m in data], axis=0)
    combined = np.concatenate([d + grand_mean for d, _ in data], axis=0)
    combined = np.moveaxis(combined, 0, -1)
    hdr = atlas.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr.set_data_shape(combined.shape)
    hdr.set_xyzt_units("mm", "sec")
    hdr["pixdim"][4] = 2.0
    nib.save(nib.Nifti1Image(combined, atlas.affine, hdr), os.path.join(odir, "func_run_combined.nii"))

    np.savetxt(os.path.join(odir, "rp_fmriprep.txt"), np.vstack(motion), fmt="%.8f")
    with open(os.path.join(odir, "parameter_RS.txt"), "w") as fh:
        fh.write('"UserInput" : {\n\t "SmoothFWHMmm" : 8\n\t "TRs" : 2.0\n\t "fmriprepInput" : 1\n}\n')
    return summary


def main():
    atlas = nib.load(ATLAS)
    args = sys.argv[1:]
    censor_only = "--censor-only" in args
    args = [a for a in args if a != "--censor-only"]
    labels = args or sorted(os.path.basename(p)[4:] for p in glob.glob(os.path.join(FMRIPREP, "sub-*"))
                                    if os.path.isdir(p))
    out_csv = os.path.join(HERE, "dlrs_input_manifest.csv")
    old = {}
    if os.path.exists(out_csv):
        old = {r["label"]: r for r in csv.DictReader(open(out_csv))}
    for lab in labels:
        try:
            s = prepare(lab, atlas, censor_only)
        except Exception as e:
            s = {"label": lab, "status": f"ERROR: {e}"}
        runs = ";".join(f"ses-{r['session']}/run-{r['run']}" for r in s.get("runs", []))
        fd = ";".join(str(r["mean_FD_mm"]) for r in s.get("runs", []))
        old[lab] = {"label": lab, "group": s.get("group", ""), "runs": runs, "volumes": s.get("volumes", ""),
                    "censored": s.get("censored", ""), "censored_motion": s.get("censored_motion", ""),
                    "censored_outliers": s.get("censored_outliers", ""), "censor_frac_pct": s.get("censor_frac_pct", ""),
                    "mean_FD_mm_per_run": fd, "status": s["status"]}
        print(f"{lab:6s} {runs:28s} vols={s.get('volumes', '-')!s:4s} censored={s.get('censor_frac_pct', '-')}%  {s['status']}")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "group", "runs", "volumes", "censored", "censored_motion",
                                           "censored_outliers", "censor_frac_pct", "mean_FD_mm_per_run", "status"],
                           extrasaction="ignore")
        w.writeheader()
        for lab in sorted(old):
            w.writerow(old[lab])


if __name__ == "__main__":
    main()
