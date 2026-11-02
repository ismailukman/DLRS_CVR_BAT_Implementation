#!/usr/bin/env python
"""Convert the raw intake to BIDS (resting BOLD and MPRAGE only).

- dcm2niix v1.0.20230411 for PAR/REC; Kirby21 NIfTI files are copied.
- TR is the spacing of the dynamic begin times in each PAR file.
- Slice times: sequential ascending (seq+z).
- Truncated runs are not converted.
- Identifying JSON fields are removed.
- Every file written is logged to bids_conversion_log.csv.
- Per-participant exceptions are read from local_config.json
  (see local_config.example.json).

    python fmriprep/02_convert_to_bids.py --subjects <intake_folder> ... [--dry-run]
"""
import os
import re
import csv
import glob
import json
import gzip
import shutil
import argparse
import subprocess
import tempfile

import numpy as np
import nibabel as nib

INTAKE = "/Volumes/WDPassport2/A-fMRIData/sci_project/01_DATA/00_INTAKE/incoming"
INTAKE2 = "/Volumes/WDPassport2/A-fMRIData/sci_parcellation_project/01_DATA/00_INTAKE/incoming"
BIDS = "/Volumes/WDPassport2/A-fMRIData/sci_project/01_DATA/01_RAW/bids"
DCM2NIIX = os.environ.get("DCM2NIIX", "dcm2niix")   # v1.0.20230411
HERE = os.path.dirname(os.path.abspath(__file__))
WORKBOOK = os.path.join(os.path.dirname(HERE), "DLRS_HC_SCI_subjects.xlsx")

CONFIG = os.path.join(HERE, "local_config.json")   # per-participant decisions (not in git)
DECISIONS = json.load(open(CONFIG))["decisions"] if os.path.exists(CONFIG) else {}
DROP_FIELDS = ("ImageComments", "PatientName", "PatientID", "PatientBirthDate",
               "AcquisitionDateTime", "InstitutionName", "InstitutionAddress")


def par_field(par, name):
    m = re.search(re.escape(name) + r"\s*:\s*(.*)", open(par, errors="replace").read(20000))
    return m.group(1).strip() if m else ""


def par_tr(par):
    """Spacing of the dynamic begin times (column 32); header TR for single volumes."""
    first = {}
    for line in open(par, errors="replace"):
        r = line.split()
        if len(r) > 31 and not line.startswith(("#", ".")):
            first.setdefault(int(r[2]), float(r[31]))
    if len(first) < 2:
        return round(float(par_field(par, "Repetition time [ms]").split()[0]) / 1000, 6)
    dt = np.diff([first[k] for k in sorted(first)])
    if dt.max() - dt.min() > 0.05:
        raise ValueError(f"{par}: uneven volume spacing {dt.min()}-{dt.max()} s")
    return round(float(np.median(dt)), 6)


def is_t1(par):
    # protocol name, not technique: the cor/sag reformats and the scout are also T1TFE
    return "MPRAGE" in par_field(par, "Protocol name").upper() and \
        par_field(par, "Max. number of dynamics") == "1"


def is_bold(par):
    dyn = par_field(par, "Max. number of dynamics")
    return dyn.isdigit() and int(dyn) >= 50 and "TASK" not in par_field(par, "Protocol name").upper()


def clean_json(path, extra):
    j = json.load(open(path))
    for k in DROP_FIELDS:
        j.pop(k, None)
    j.update(extra)
    return j


def slice_timing(nii, tr):
    n = nib.load(nii).shape[2]
    return {"SliceTiming": [round(k * tr / n, 6) for k in range(n)], "SliceEncodingDirection": "k"}


def set_nifti_tr(path, tr):
    img = nib.load(path)
    hdr = img.header.copy()
    zooms = list(hdr.get_zooms())
    zooms[3] = tr
    hdr.set_zooms(zooms)
    hdr.set_xyzt_units("mm", "sec")
    nib.save(nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine, hdr), path)


def find_group(folder, wb):
    if "source" in DECISIONS.get(folder, {}):          # session outside the intake (e.g. SCIFES)
        return "subject3"
    for g in ("control", "subject"):
        if os.path.isdir(os.path.join(INTAKE, g, folder)):
            return g
    if os.path.isdir(os.path.join(INTAKE2, folder)):
        key = folder.split("_")[0].lower()
        for sheet, g in (("HC_42", "control2"), ("SCI_38", "subject2")):
            if any(r and r[2] is not None and str(r[2]).strip().lower() == key
                   for r in wb[sheet].iter_rows(min_row=2, values_only=True)):
                return g
    return None


def source_dir(group, folder):
    if group == "subject3":
        return DECISIONS[folder]["source"]
    return os.path.join(INTAKE2, folder) if group.endswith("2") else os.path.join(INTAKE, group, folder)


def subject_label(group, folder, wb):
    if "bids_label" in DECISIONS.get(folder, {}):
        return DECISIONS[folder]["bids_label"]
    key = folder.split("_")[0].lower() if group.endswith("2") else re.sub(r"^kirby", "", folder.lower())
    sheet = "HC_42" if group.startswith("control") else "SCI_38"
    for r in wb[sheet].iter_rows(min_row=2, values_only=True):
        if r and r[2] is not None and str(r[2]).strip().lower() == key:
            return str(r[1]).strip()
    return None


def convert_subject(group, folder, wb, log, dry):
    src = source_dir(group, folder)
    label = subject_label(group, folder, wb)
    dec = DECISIONS.get(folder, {})
    if not label:
        print(f"  {folder}: no workbook label, skipped")
        return
    if dec.get("skip"):
        log.append([folder, "", "", "skipped (config)"])
        return
    pars = sorted(p for p in glob.glob(os.path.join(src, "*.par")) if not os.path.basename(p).startswith("._"))
    kirby = folder.lower().startswith("kirby")
    if kirby:
        visits = sorted({re.match(r"KKI2009-(\d+)", os.path.basename(p)).group(1) for p in pars})
        ses_of = lambda p: str(visits.index(re.match(r"KKI2009-(\d+)", os.path.basename(p)).group(1)) + 1)
    else:
        ses_of = lambda p: dec.get("session", "1")

    kept, counts = [], {}
    for par in pars:
        if os.path.basename(par) in dec.get("skip_files", []):
            log.append([folder, os.path.basename(par), "", "skipped (config)"])
            continue
        kind = "anat" if is_t1(par) else "func" if is_bold(par) else None
        if kind is None:
            continue
        ses = ses_of(par)
        counts[(ses, kind)] = counts.get((ses, kind), 0) + 1
        if kind == "func" and counts[(ses, kind)] > dec.get("max_bold_runs", 99):
            log.append([folder, os.path.basename(par), "", "skipped: beyond max_bold_runs"])
            continue
        kept.append((par, kind, ses, counts[(ses, kind)]))
    n_anat = {ses: sum(1 for k in kept if k[1] == "anat" and k[2] == ses) for _, _, ses, _ in kept}

    for par, kind, ses, run in kept:
        tr = par_tr(par)
        base = f"sub-{label}_ses-{ses}"
        if kind == "anat":
            name = f"{base}_run-{run}_T1w" if n_anat[ses] > 1 else f"{base}_T1w"
        else:
            name = f"{base}_task-rest_run-{run}_bold"
        outdir = os.path.join(BIDS, f"sub-{label}", f"ses-{ses}", kind)
        out_nii = os.path.join(outdir, name + ".nii.gz")
        log.append([folder, os.path.basename(par), os.path.relpath(out_nii, BIDS), f"TR={tr}s"])
        if dry:
            continue
        os.makedirs(outdir, exist_ok=True)
        extra = {"Manufacturer": "Philips", "ProtocolName": par_field(par, "Protocol name")}
        if kind == "func":
            extra.update({"RepetitionTime": tr, "TaskName": "rest"})

        if kirby:
            with open(par[:-4] + ".nii", "rb") as fi, gzip.open(out_nii, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            j = dict(extra)
        else:
            with tempfile.TemporaryDirectory() as td:
                r = subprocess.run([DCM2NIIX, "-b", "y", "-z", "y", "-f", "x", "-o", td, par],
                                   capture_output=True, text=True)
                niis = glob.glob(os.path.join(td, "*.nii.gz"))
                if r.returncode != 0 or len(niis) != 1:
                    why = (r.stderr.strip().splitlines() or ["no output"])[0][:120]
                    log[-1][2:] = ["", f"SKIPPED: dcm2niix failed: {why}"]
                    continue
                shutil.move(niis[0], out_nii)
                j = clean_json(glob.glob(os.path.join(td, "*.json"))[0], extra)

        if kind == "func":
            want = int(par_field(par, "Max. number of dynamics"))
            img = nib.load(out_nii)
            got = img.shape[3] if img.ndim == 4 else 1
            if got != want:
                os.remove(out_nii)
                log[-1][2:] = ["", f"SKIPPED: truncated, {got} of {want} volumes"]
                continue
            if abs(float(img.header["pixdim"][4]) - tr) > 1e-3:
                set_nifti_tr(out_nii, tr)
            j.update(slice_timing(out_nii, tr))
        json.dump(j, open(out_nii.replace(".nii.gz", ".json"), "w"), indent=2)

    if any(k in dec for k in ("dicom_func", "dicom_anat", "nifti_bold", "dicom_files_func", "dicom_files_anat")):
        convert_extra(src, label, dec, log, dry)


def convert_extra(src, label, dec, log, dry):
    """Runs without a PAR file: DICOM series or bare NIfTI listed in the config."""
    base = f"sub-{label}_ses-{dec.get('session', '1')}"
    anat = dec.get("dicom_anat", []) or dec.get("dicom_files_anat", [])
    how_a = "dicomfile" if "dicom_files_anat" in dec else "dicom"
    jobs = [("anat", f"{base}_T1w" if len(anat) == 1 else f"{base}_run-{i}_T1w", how_a, s)
            for i, s in enumerate(anat, 1)]
    jobs += [("func", f"{base}_task-rest_run-{i}_bold", "dicom", s) for i, s in enumerate(dec.get("dicom_func", []), 1)]
    jobs += [("func", f"{base}_task-rest_run-{i}_bold", "dicomfile", f)
             for i, f in enumerate(dec.get("dicom_files_func", []), 1)]
    jobs += [("func", f"{base}_task-rest_run-{i}_bold", "nifti", f) for i, f in enumerate(dec.get("nifti_bold", []), 1)]
    for kind, name, how, what in jobs:
        out_nii = os.path.join(BIDS, f"sub-{label}", base.split("_")[1], kind, name + ".nii.gz")
        log.append([os.path.basename(src), f"{how}:{what}", os.path.relpath(out_nii, BIDS), ""])
        if dry:
            continue
        os.makedirs(os.path.dirname(out_nii), exist_ok=True)
        if how in ("dicom", "dicomfile"):
            with tempfile.TemporaryDirectory() as td:
                if how == "dicom":
                    subprocess.run([DCM2NIIX, "-b", "y", "-z", "y", "-f", "%s", "-o", td,
                                    os.path.join(src, dec["dicom_dir"])], capture_output=True, text=True)
                    nii = os.path.join(td, f"{what}.nii.gz")
                else:                                       # one (enhanced) DICOM file per series
                    subprocess.run([DCM2NIIX, "-s", "y", "-b", "y", "-z", "y", "-f", "x", "-o", td,
                                    os.path.join(src, what)], capture_output=True, text=True)
                    nii = os.path.join(td, "x.nii.gz")
                    what = os.path.join(td, "x")
                if not os.path.exists(nii):
                    log[-1][2:] = ["", f"SKIPPED: DICOM series {what} not found"]
                    continue
                shutil.move(nii, out_nii)
                j = clean_json(os.path.join(td, f"{what}.json"), {})
                if kind == "func" and abs(float(nib.load(out_nii).header["pixdim"][4]) - j["RepetitionTime"]) > 1e-3:
                    set_nifti_tr(out_nii, j["RepetitionTime"])
            if kind == "func":
                j["TaskName"] = "rest"
                j.update(slice_timing(out_nii, j["RepetitionTime"]))
                log[-1][3] = f"TR={j['RepetitionTime']}s from DICOM"
        else:
            with open(os.path.join(src, what), "rb") as fi, gzip.open(out_nii, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            tr = dec["nifti_tr"]
            set_nifti_tr(out_nii, tr)
            j = {"Manufacturer": "Philips", "RepetitionTime": tr, "TaskName": "rest",
                 "ProtocolName": "WIP RestingState SENSE"}
            j.update(slice_timing(out_nii, tr))
            log[-1][3] = f"TR={tr}s from NIfTI header; run identified from physio log times"
        json.dump(j, open(out_nii.replace(".nii.gz", ".json"), "w"), indent=2)


def write_dataset_files(labels):
    os.makedirs(BIDS, exist_ok=True)
    json.dump({"Name": "SCI resting-state CVR study", "BIDSVersion": "1.9.0", "DatasetType": "raw",
               "GeneratedBy": [{"Name": "02_convert_to_bids.py", "Description": "dcm2niix v1.0.20230411"}]},
              open(os.path.join(BIDS, "dataset_description.json"), "w"), indent=2)
    ptsv = os.path.join(BIDS, "participants.tsv")
    existing = set()
    if os.path.exists(ptsv):
        existing = {l.split("\t")[0][4:] for l in open(ptsv).read().splitlines()[1:] if l.strip()}
    with open(ptsv, "w") as fh:
        fh.write("participant_id\tgroup\n")
        for lab in sorted(existing | set(labels)):
            fh.write(f"sub-{lab}\t{'HC' if lab.startswith('hc') else 'SCI'}\n")
    readme = os.path.join(BIDS, "README")
    if not os.path.exists(readme):
        open(readme, "w").write("Resting-state BOLD and T1w, SCI CVR study.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True, help="intake folder names")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import openpyxl
    wb = openpyxl.load_workbook(WORKBOOK, data_only=True)
    log, labels = [], []
    for folder in a.subjects:
        group = find_group(folder, wb)
        if group is None:
            print(f"{folder}: not in either intake, skipped")
            continue
        lab = subject_label(group, folder, wb)
        print(f"{folder} -> sub-{lab}")
        convert_subject(group, folder, wb, log, a.dry_run)
        if lab:
            labels.append(lab)
    if not a.dry_run:
        write_dataset_files(labels)
        out = os.path.join(HERE, "bids_conversion_log.csv")
        new = not os.path.exists(out)
        with open(out, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["intake_folder", "source_header", "bids_file", "note"])
            w.writerows(log)
    for row in log:
        print(f"  {row[1]:34s} -> {row[2] or '-':60s} {row[3]}")


if __name__ == "__main__":
    main()
