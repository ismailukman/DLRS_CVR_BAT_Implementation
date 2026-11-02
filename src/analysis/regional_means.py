"""Regional mean DLRS CVR and BAT for every participant.

Regions: whole brain, 7 cortical networks (Schaefer 200, Yeo-7) and 7 bilateral
subcortical structures (Tian S1), each within the participant's brain mask.
Motion is the percentage of censored volumes (run_info.json).

    python src/analysis/regional_means.py

Output: results/regional_means.csv
"""
import os
import json
import numpy as np
import pandas as pd

from common import DATA, GROUPS, METRICS, RESULTS, brain_mask, dlrs_map, participants, region_masks


def main():
    masks = region_masks()
    rows = []
    for group, label in participants():
        brain = brain_mask(group, label)
        info = json.load(open(os.path.join(DATA, GROUPS[group], label, "run_info.json")))
        row = {"group": group, "subject": label, "motion": info["censor_frac_pct"]}
        for metric in METRICS:
            vol = dlrs_map(group, label, metric)
            for region, m in masks.items():
                m = brain if m is None else (m & brain)
                row[f"{region}|{metric}"] = float(np.nanmean(vol[m])) if m.any() else np.nan
        rows.append(row)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "regional_means.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.group.value_counts().to_dict(), "->", out)


if __name__ == "__main__":
    main()
