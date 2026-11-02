#!/usr/bin/env python
"""List every raw scan and classify it from its own header (PAR or JSON).
Nothing is converted. Writes scan_survey.csv and scan_survey.txt (per-subject
summary and cases to check).

    python fmriprep/01_survey_scans.py
"""
import os
import re
import csv
import json
import glob
from collections import defaultdict

import nibabel as nib

INTAKE = "/Volumes/WDPassport2/A-fMRIData/sci_project/01_DATA/00_INTAKE/incoming"
HERE = os.path.dirname(os.path.abspath(__file__))

PAR_FIELDS = {
    "protocol": r"Protocol name\s*:\s*(.*)",
    "exam_date": r"Examination date/time\s*:\s*(.*)",
    "technique": r"Technique\s*:\s*(.*)",
    "scan_mode": r"Scan mode\s*:\s*(.*)",
    "n_slices": r"Max\. number of slices/locations\s*:\s*(\d+)",
    "n_dynamics": r"Max\. number of dynamics\s*:\s*(\d+)",
    "tr_ms": r"Repetition time \[ms\]\s*:\s*([\d.]+)",
    "resolution": r"Scan resolution\s+\(x, y\)\s*:\s*(.*)",
}


def read_par(path):
    out = {}
    with open(path, "r", errors="replace") as fh:
        head = fh.read(20000)
    for key, pat in PAR_FIELDS.items():
        m = re.search(pat, head)
        out[key] = m.group(1).strip() if m else ""
    return out


def read_json(path):
    d = json.load(open(path))
    return {
        "protocol": d.get("ProtocolName") or d.get("SeriesDescription", ""),
        "exam_date": d.get("AcquisitionDateTime", d.get("AcquisitionTime", "")),
        "technique": d.get("PulseSequenceName", d.get("ScanningSequence", "")),
        "scan_mode": d.get("MRAcquisitionType", ""),
        "n_slices": "",
        "n_dynamics": "",
        "tr_ms": str(round(float(d["RepetitionTime"]) * 1000, 3)) if "RepetitionTime" in d else "",
        "resolution": "",
    }


def classify(h, n_vols):
    """Return anat_T1w, func_bold, or other, with the reason."""
    p = (h.get("protocol", "") + " " + h.get("technique", "")).upper()
    dyn = int(h["n_dynamics"]) if str(h.get("n_dynamics", "")).isdigit() else n_vols
    if ("MPRAGE" in p or "T1TFE" in p or "T1W" in p) and (dyn in (None, 1)):
        return "anat_T1w", "MPRAGE/T1TFE, single volume"
    if dyn and dyn >= 50 and ("FEEPI" in p or "FMRI" in p or "REST" in p or "BOLD" in p or "EPI" in p):
        return "func_bold", f"EPI time series, {dyn} volumes"
    if dyn and dyn >= 50:
        return "func_bold?", f"{dyn} volumes but protocol does not say EPI/rest: check"
    return "other", "not T1 and not a time series"


def nifti_vols(path):
    try:
        shp = nib.load(path).shape
        return (shp[3] if len(shp) > 3 else 1), "x".join(map(str, shp))
    except Exception as e:
        return None, f"unreadable: {e}"[:60]


def main():
    rows = []
    skipped = []
    for group in ("control", "subject"):
        for sdir in sorted(glob.glob(os.path.join(INTAKE, group, "*"))):
            if not os.path.isdir(sdir):
                continue
            subj = os.path.basename(sdir)
            files = sorted(os.listdir(sdir))
            for f in files:
                full = os.path.join(sdir, f)
                if f.startswith("._") or f == ".DS_Store":
                    skipped.append(full)
                    continue
                stem, ext = None, None
                if f.lower().endswith(".par"):
                    stem, ext = f[:-4], "par"
                elif f.lower().endswith(".json"):
                    stem, ext = f[:-5], "json"
                else:
                    continue
                h = read_par(full) if ext == "par" else read_json(full)
                img = None
                for cand in (stem + ".rec", stem + ".REC", stem + ".nii", stem + ".nii.gz"):
                    if os.path.exists(os.path.join(sdir, cand)):
                        img = cand
                        break
                n_vols, shape = (None, "")
                if img and img.endswith((".nii", ".nii.gz")):
                    n_vols, shape = nifti_vols(os.path.join(sdir, img))
                kind, why = classify(h, n_vols)
                rows.append({
                    "group": group, "subject": subj, "header": f, "image": img or "MISSING",
                    "kind": kind, "reason": why, "protocol": h["protocol"],
                    "technique": h["technique"], "scan_mode": h["scan_mode"],
                    "n_dynamics": h["n_dynamics"] or (n_vols or ""), "n_slices": h["n_slices"],
                    "tr_ms": h["tr_ms"], "exam_date": h["exam_date"], "nifti_shape": shape,
                })

    out_csv = os.path.join(HERE, "scan_survey.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["group"], r["subject"])][r["kind"]].append(r)
    lines = []
    lines.append(f"scans surveyed: {len(rows)}   AppleDouble/.DS_Store skipped: {len(skipped)}")
    flags = []
    for (g, s), kinds in sorted(by.items()):
        na, nf = len(kinds.get("anat_T1w", [])), len(kinds.get("func_bold", []))
        trs = sorted({r["tr_ms"] for r in kinds.get("func_bold", [])})
        dyn = sorted({str(r["n_dynamics"]) for r in kinds.get("func_bold", [])})
        dates = sorted({r["exam_date"][:10] for r in rows if r["subject"] == s and r["exam_date"]})
        lines.append(f"{g:8s} {s:14s} T1w={na}  BOLD={nf}  TR={','.join(trs) or '-'}  "
                     f"vols={','.join(dyn) or '-'}  dates={','.join(dates) or '-'}")
        if na == 0:
            flags.append(f"{s}: no T1w found (fMRIPrep cannot run without one)")
        if nf == 0:
            flags.append(f"{s}: no BOLD run found")
        if len(dates) > 1:
            flags.append(f"{s}: scans from {len(dates)} different dates -> separate sessions?")
        if kinds.get("func_bold?"):
            flags.append(f"{s}: {len(kinds['func_bold?'])} long series with unclear protocol")
        for r in kinds.get("anat_T1w", []) + kinds.get("func_bold", []):
            if r["image"] == "MISSING":
                flags.append(f"{s}: header {r['header']} has no matching image")
    lines.append("")
    lines.append(f"TO CHECK ({len(flags)}):")
    lines += [f"  - {x}" for x in flags] or ["  none"]
    txt = "\n".join(lines)
    open(os.path.join(HERE, "scan_survey.txt"), "w").write(txt + "\n")
    print(txt)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
