"""Run-to-run repeatability of DLRS CVR and BAT in 216 parcels.

Healthy controls whose two resting-state runs were processed separately
(<label>_run1 and <label>_run2 folders with DLRS_CVR.nii and DLRS_BAT.nii).
Parcels: Schaefer 200 + Tian S1 (16). Brain = voxels with a non-zero model
output. Parcels with fewer than 10 voxels in any run are left out.

ICC(2,1): two-way random effects, absolute agreement, single run.

    DLRS_RETEST_DIR=<data_retest>/output/HC_subj python src/analysis/reliability.py

Output: results/reliability_icc.csv
"""
import os
import numpy as np
import pandas as pd

from common import PROJ, RESULTS, load, parcels_216

RETEST = os.environ.get("DLRS_RETEST_DIR", os.path.join(PROJ, "data_retest", "output", "HC_subj"))
MIN_VOX = 10


def parcel_means(path, labels, n):
    vol = load(path)
    lab = np.where((np.abs(vol) > 1e-8) & np.isfinite(vol), labels, 0).ravel()
    count = np.bincount(lab, minlength=n + 1)[1:]
    total = np.bincount(lab, weights=np.nan_to_num(vol).ravel(), minlength=n + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan), count


def icc_2_1(x, y):
    Y = np.column_stack([x, y])
    n, k = Y.shape
    grand = Y.mean()
    ms_rows = k * ((Y.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((Y.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + grand
    ms_err = (resid ** 2).sum() / ((n - 1) * (k - 1))
    return (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n)


def main():
    labels, names = parcels_216()
    n = len(names)
    subjects = sorted({d.rsplit("_run", 1)[0] for d in os.listdir(RETEST) if "_run" in d})
    subjects = [s for s in subjects if all(os.path.isdir(os.path.join(RETEST, f"{s}_run{r}")) for r in (1, 2))]
    rows = []
    for metric in ("CVR", "BAT"):
        runs = {r: [] for r in (1, 2)}
        vox = []
        for s in subjects:
            for r in (1, 2):
                m, c = parcel_means(os.path.join(RETEST, f"{s}_run{r}", f"DLRS_{metric}.nii"), labels, n)
                runs[r].append(m)
                vox.append(c)
        a, b = np.array(runs[1]), np.array(runs[2])
        ok = np.min(vox, axis=0) >= MIN_VOX
        for i in np.where(ok)[0]:
            rows.append(dict(parcel=names[i], metric=metric, icc_2_1=icc_2_1(a[:, i], b[:, i]),
                             n_subjects=len(subjects)))
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "reliability_icc.csv")
    df.round(4).to_csv(out, index=False)
    for metric in ("CVR", "BAT"):
        v = df[df.metric == metric].icc_2_1
        print(f"{metric}: {len(v)} parcels, {len(subjects)} HC, median ICC(2,1) {v.median():.2f}")
    print("->", out)


if __name__ == "__main__":
    main()
