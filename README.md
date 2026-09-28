# DLRS_CVR_BAT

Cerebrovascular reactivity (CVR) and bolus arrival time (BAT) mapping from
resting-state fMRI, without a gas challenge. Implementation and group analysis for a
healthy control (HC) versus spinal cord injury (SCI) study.

Model architecture from Hou et al., *npj Digital Medicine* 6:116 (2023).

## Pipeline

```
errts (AFNI fast-ANATICOR residuals)
  -> MATLAB: 136-channel input
     133 Neuromorphometrics ROI correlation maps + CVR beta0, beta1, BAT priors
  -> Python: pretrained dual-decoder UNet, ~10M parameters
  -> DLRS_CVR.nii, DLRS_BAT.nii
```

Outputs are in arbitrary units from a `5*tanh()` layer, not %BOLD/mmHg or seconds.
Display ranges used throughout: CVR `[-0.8, 0.8]`, BAT `[-1.2, 1.2]`.

## Layout

```
preprocessing/     MATLAB feature generation (SPM12 installed separately)
  utils/           per-step routines
  atlas/           Neuromorphometrics, Schaefer-2018, Tian S1
src/               model, training, inference
  analysis/        group statistics and figure generation
figure/            generated figures and result tables
model/             pretrained weights (not tracked)
data/              subject data (not tracked)
```

## Setup

```bash
conda create -n cvr_bat python=3.9
conda activate cvr_bat
pip install -r requirements.txt
python check_setup.py
```

MATLAB requires SPM12 in `preprocessing/spm12/` (download from
https://www.fil.ion.ucl.ac.uk/spm/software/spm12/). Pretrained weights go in
`model/`.

## Running

Single subject:

```bash
matlab -nodisplay -r "rs_running; exit"
python src/DLRS_CVR_BAT_inference.py
```

Group processing expects `data/HC_subj/` and `data/SCI_subj/`:

```bash
matlab -nodisplay -r "rs_running_groups; exit"
python src/DLRS_CVR_BAT_inference.py
python src/analysis/visualize_results.py
```

## Analysis

Scripts in `src/analysis/` read the DLRS maps and write to `figure/`. Run them
from the project root, for example `python src/analysis/group_ancova.py`.

| Script | Output |
|---|---|
| `visualize_group_comparison.py` | representative and group-mean maps, per-subject montage, box plots |
| `model_result_group.py` | DLRS versus GLM metrics, Bland-Altman, MCA territory, hemispheric |
| `analyze_network_atlases.py` | Yeo-7 and Tian S1 network comparison, atlas QC |
| `group_ancova.py` | age and motion adjusted group effects, 16 pooled regions |
| `region_ancova_216.py` | per-region effects across all 216 parcels, FDR corrected |
| `mca_finding_figure.py` | MCA territory result and its age dependence |
| `region216_network_figure.py` | 216 parcels grouped by resting-state network |
| `region216_surface_figure.py` | per-region effect on the cortical surface (needs nilearn) |
| `slice_vs_volume_mean.py` | mid-axial slice versus through-brain mean |
| `visualize_results.py` | per-subject CVR and BAT maps for a quick check |

## Notes

Group maps are the mean across subjects shown as a single axial slice at voxel
k = 45 of 91 (MNI z = +18 mm), not averaged through the volume.

Statistics use an age and motion adjusted linear model. Motion is the preprocessing
censor fraction, which differs substantially between groups and is carried as a
covariate throughout.
