# DLRS CVR and BAT in spinal cord injury

![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)
[![SPM25](https://img.shields.io/badge/MATLAB-SPM25-0076A8)](https://github.com/spm/spm)
[![fMRIPrep](https://img.shields.io/badge/fMRIPrep-25.2.6-4C9A2A)](https://github.com/nipreps/fmriprep)
![BIDS](https://img.shields.io/badge/data-BIDS-F2A900)
![Modality](https://img.shields.io/badge/MRI-resting--state%20fMRI-6A5ACD)

Cerebrovascular reactivity (CVR) and bolus arrival time (BAT) maps from
resting-state fMRI without a gas challenge, using the pretrained deep-learning
resting-state (DLRS) model of Hou et al. (2023), and a comparison of spinal cord
injury (SCI) and healthy control (HC) participants.

## Layout

```
fmriprep/          BIDS conversion, fMRIPrep, QC, censoring, MATLAB input
preprocessing/     MATLAB: 136 input channels (133 ROI correlation maps + 3 GLM priors)
  atlas/           empty atlas folders (files not tracked)
src/               U-Net model, training, inference
  analysis/        regional means, group comparison, repeatability, figures
figure/            main figures
```

## Preprocessing

[fMRIPrep](https://github.com/nipreps/fmriprep) 25.2.6 (Esteban et al., 2019),
run unmodified from its Docker image with `--slice-time-ref 0` and
`--output-spaces MNI152NLin6Asym:res-2`. After fMRIPrep: linear detrending per
run, censoring of volumes with motion > 0.4 mm or > 10% outlier voxels, 8 mm
smoothing, 0-0.1164 Hz filtering and motion covariates (6 parameters and their
squares). Participants with > 25% censored volumes are excluded.

## Setup

```bash
conda create -n cvr_bat python=3.9
conda activate cvr_bat
pip install -r requirements.txt
```

Also needed: Docker with `nipreps/fmriprep:25.2.6` and a FreeSurfer license,
dcm2niix v1.0.20230411, AFNI (only `3dToutcount`, for outlier censoring),
MATLAB with [SPM25](https://github.com/spm/spm)
(25.01.02) in `preprocessing/spm12/` (the code expects this folder name), and
the pretrained weights in `model/`.

Atlases go in `preprocessing/atlas/`:
- `labels_Neuromorphometrics/`: `wlabels_Neuromorphometrics_unique.nii` (from the DLRS code of Hou et al.)
- `schaefer_2018/`: 200 parcels, 7 networks, [CBIG](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI)
- `tian_s1/`: 3T scale 1, [Tian subcortex atlas](https://github.com/yetianmed/subcortex/tree/master/Group-Parcellation/3T/Subcortex-Only)

Data folders are set at the top of each `fmriprep/` script.

## Running

```bash
python fmriprep/02_convert_to_bids.py --subjects <intake_folder> ...
bash fmriprep/03_run_fmriprep.sh sci01 hc01
python fmriprep/06_prepare_dlrs_input.py
matlab -batch "cd preprocessing; rs_running_fmriprep"
DLRS_DATA_DIR=<data> python src/DLRS_CVR_BAT_inference.py

export DLRS_DATA_DIR=<data>
python src/analysis/regional_means.py
python src/analysis/group_comparison.py      # needs participants.tsv (subject, age, sex)
python src/analysis/make_figures.py
```

## References

Esteban O, Markiewicz CJ, Blair RW, Moodie CA, Isik AI, Erramuzpe A, et al.
fMRIPrep: a robust preprocessing pipeline for functional MRI. Nat Methods.
2019;16(1):111-6. doi:10.1038/s41592-018-0235-4

Tierney TM, Alexander NA, Ashburner J, Avila NL, Balbastre Y, Barnes G, et al.
SPM 25: open source neuroimaging analysis software. J Open Source Softw.
2025;10(110):8103. doi:10.21105/joss.08103

Hou X, Guo P, Wang P, Liu P, Lin DDM, Fan H, et al. Deep-learning-enabled brain
hemodynamic mapping using resting-state fMRI. NPJ Digit Med. 2023;6(1):116.
doi:10.1038/s41746-023-00859-y
