"""Spatial agreement between DLRS and GLM maps, per participant.

Pearson r over brain-mask voxels between DLRS_CVR and the GLM CVR map, and
between DLRS_BAT and the GLM delay map (both written by the MATLAB step).

    python src/analysis/dlrs_glm_agreement.py

Output: results/dlrs_glm_agreement.csv
"""
import os
import numpy as np
import pandas as pd

from common import DATA, GROUPS, RESULTS, brain_mask, dlrs_map, load, participants

GLM = os.path.join("CVR_voxelshift_etco2_cerebellum", "drwarfunc_run_combined_s8_")


def main():
    rows = []
    for group, label in participants():
        m = brain_mask(group, label)
        row = {"group": group, "subject": label}
        for metric, glm in (("CVR", "CVR"), ("BAT", "co2delay")):
            a = dlrs_map(group, label, metric)[m]
            b = load(os.path.join(DATA, GROUPS[group], label, GLM + glm + ".img"))[m]
            ok = np.isfinite(a) & np.isfinite(b)
            row[f"{metric.lower()}_r"] = np.corrcoef(a[ok], b[ok])[0, 1]
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "dlrs_glm_agreement.csv")
    df.round(4).to_csv(out, index=False)
    print(f"{len(df)} participants; mean r CVR {df.cvr_r.mean():.3f}, BAT {df.bat_r.mean():.3f} -> {out}")


if __name__ == "__main__":
    main()
