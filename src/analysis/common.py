"""Paths, atlases and region masks shared by the analysis scripts.

Data layout (DLRS_DATA_DIR, default data/):

    HC_subj/<label>/mask/brainMask_RS.nii       from the MATLAB step
    HC_subj/<label>/run_info.json               from fmriprep/06_prepare_dlrs_input.py
    output/HC_subj/<label>/DLRS_CVR.nii         from src/DLRS_CVR_BAT_inference.py
    output/HC_subj/<label>/DLRS_BAT.nii
    (same for SCI_subj)

Tables are written to results/ (not tracked) and figures to figure/.
"""
import os
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("DLRS_DATA_DIR", os.path.join(PROJ, "data"))
RESULTS = os.environ.get("DLRS_RESULTS_DIR", os.path.join(PROJ, "results"))
FIGURES = os.path.join(PROJ, "figure")
PARTICIPANTS = os.environ.get("DLRS_PARTICIPANTS", os.path.join(PROJ, "participants.tsv"))

ATLAS = os.path.join(PROJ, "preprocessing", "atlas")
NEUROMORPH = os.path.join(ATLAS, "labels_Neuromorphometrics", "wlabels_Neuromorphometrics_unique.nii")
SCHAEFER = os.path.join(ATLAS, "schaefer_2018",
                        "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz")
SCHAEFER_TXT = os.path.join(ATLAS, "schaefer_2018", "Schaefer2018_200Parcels_7Networks_order.txt")
TIAN = os.path.join(ATLAS, "tian_s1", "Tian_Subcortex_S1_3T_2009cAsym.nii.gz")
TIAN_TXT = os.path.join(ATLAS, "tian_s1", "Tian_Subcortex_S1_3T_label.txt")

GROUPS = {"HC": "HC_subj", "SCI": "SCI_subj"}
METRICS = ("CVR", "BAT")
CORTEX = ["Visual", "Somatomotor", "DorsAttn", "VentAttn/Sal", "Limbic", "Control/FP", "Default"]
SUBCORTEX = ["Thalamus", "Caudate", "Putamen", "Pallidum", "Accumbens", "Hippocampus", "Amygdala"]
YEO7 = {"Vis": "Visual", "SomMot": "Somatomotor", "DorsAttn": "DorsAttn", "SalVentAttn": "VentAttn/Sal",
        "Limbic": "Limbic", "Cont": "Control/FP", "Default": "Default"}
TIAN_GROUP = {"aTHA": "Thalamus", "pTHA": "Thalamus", "CAU": "Caudate", "PUT": "Putamen", "GP": "Pallidum",
              "NAc": "Accumbens", "HIP": "Hippocampus", "AMY": "Amygdala"}
NICE = {"DorsAttn": "Dorsal attention", "VentAttn/Sal": "Ventral attention", "Control/FP": "Frontoparietal control",
        "Default": "Default mode", "whole-brain": "Whole brain"}


def nice(region):
    return NICE.get(region, region)


def load(path):
    return np.squeeze(np.asanyarray(nib.load(path).dataobj)).astype(np.float64)


def grid():
    """Shape and affine of the analysis grid (MNI152 2 mm, 91 x 109 x 91)."""
    img = nib.load(NEUROMORPH)
    return img.shape[:3], img.affine


def resample_labels(path):
    shape, affine = grid()
    out = resample_from_to(nib.load(path), (shape, affine), order=0)
    return np.rint(np.asanyarray(out.dataobj)).astype(int)


def schaefer_names():
    """Parcel id -> name (e.g. 'LH_Vis_1') from the Schaefer label file."""
    names = {}
    for line in open(SCHAEFER_TXT):
        parts = line.split()
        if len(parts) >= 2:
            names[int(parts[0])] = parts[1].replace("7Networks_", "")
    return names


def tian_names():
    return [l.strip() for l in open(TIAN_TXT) if l.strip()]


def region_masks():
    """Whole brain (None = the participant's brain mask), 7 Yeo networks, 7 bilateral Tian S1 structures."""
    parc = resample_labels(SCHAEFER)
    sub = resample_labels(TIAN)
    masks = {"whole-brain": None}
    names = schaefer_names()
    for net in CORTEX:
        ids = [i for i, n in names.items() if YEO7[n.split("_")[1]] == net]
        masks[net] = np.isin(parc, ids)
    tian = tian_names()
    for struct in SUBCORTEX:
        ids = [i for i, n in enumerate(tian, 1) if TIAN_GROUP[n.split("-")[0]] == struct]
        masks[struct] = np.isin(sub, ids)
    return masks


def parcels_216():
    """Label volume with 200 Schaefer parcels (1-200) and 16 Tian S1 nuclei (201-216).
    Tian takes precedence where the two overlap. Returns (labelvol, names)."""
    parc = resample_labels(SCHAEFER)
    sub = resample_labels(TIAN)
    names = [schaefer_names()[i] for i in range(1, 201)] + tian_names()
    lab = np.where(parc > 0, parc, 0)
    lab[sub > 0] = 200 + sub[sub > 0]
    return lab, names


def participants(data=DATA):
    """(group, label) for every participant with DLRS output."""
    out = []
    for g, gd in GROUPS.items():
        d = os.path.join(data, "output", gd)
        out += [(g, s) for s in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, s, "DLRS_CVR.nii"))]
    return out


def dlrs_map(group, label, metric, data=DATA):
    return load(os.path.join(data, "output", GROUPS[group], label, f"DLRS_{metric}.nii"))


def brain_mask(group, label, data=DATA):
    return load(os.path.join(data, GROUPS[group], label, "mask", "brainMask_RS.nii")) > 0.5
