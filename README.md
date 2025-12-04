# DLRS_CVR_BAT: Deep Learning for CVR and BAT Mapping

![alt image](https://img.shields.io/badge/python-3.9-green)

This repository contains the code and pre-trained models for deep-learning-enabled brain hemodynamic mapping using resting-state fMRI.

**Paper:** [Deep-learning-enabled brain hemodynamic mapping using resting-state fMRI](https://www.nature.com/articles/s41746-023-00859-y)
**Citation:** Hou, X., Guo, P., Wang, P. et al. Deep-learning-enabled brain hemodynamic mapping using resting-state fMRI. npj Digit. Med. 6, 116 (2023).

---

## Quick Start

### Prerequisites
- Python 3.9
- MATLAB with SPM12 and SUIT toolbox
- Preprocessed resting-state fMRI data (BOLD + T1/MPRAGE)

### Installation & Setup

```bash
# 1. Install Python dependencies
./install_dependencies.sh

# 2. Verify setup
python check_setup.py

# 3. Run MATLAB preprocessing
cd preprocessing
matlab -nodisplay -r 'run rs_running.m; exit'

# 4. Run Python inference
cd ..
python ./src/DLRS_CVR_BAT_inference.py

# 5. Visualize results
python visualize_results.py
```

### Output
- `data/output/<subject>/DLRS_CVR.nii` - Cerebrovascular Reactivity map
- `data/output/<subject>/DLRS_BAT.nii` - Bolus Arrival Time map
- `data/CVR_BAT_Results.png` - Visualization comparing all subjects
- See [OUTPUT_EXPLANATION.md](OUTPUT_EXPLANATION.md) for detailed interpretation

---

## Overview

This pipeline maps **cerebrovascular reactivity (CVR)** and **bolus arrival time (BAT)** from resting-state fMRI data using deep learning. Unlike traditional methods requiring hypercapnic challenges, this approach uses natural CO2 fluctuations extracted from resting-state data.

### Method
1. **MATLAB Preprocessing**: Extracts cerebellum signal as surrogate CO2, calculates correlation maps with 134 brain ROIs, and generates 136-channel feature maps
2. **Python Deep Learning**: UNet model predicts CVR and BAT from the feature maps

### Network Architecture
![Network Architecture](/figure/Figure1.png)

**Model:** UNet with dual output
- **Input:** 136 channels (134 ROI correlations + CVR coefficients + BAT)
- **Output:** 2 channels (CVR map + BAT map)
- **Architecture:** 5-level encoder-decoder with batch normalization

---

## Data Requirements

### Input Files Per Subject

Each subject folder needs:

```
data/subject_name/
├── parameter_RS.txt          # Configuration file
├── slice_order_RS.txt        # Slice acquisition order (36 slices)
├── PreprocessedData/
│   ├── func_run1_df.nii     # BOLD timeseries (4D NIfTI)
│   └── struct_rf.nii        # T1/MPRAGE anatomical scan
```

### Configuration File Format

**parameter_RS.txt:**
```
"UserInput" : {
     "SmoothFWHMmm" : 8
     "TRs" : 2.0
     "mprageFileName" : "PreprocessedData/struct_rf.nii"
     "boldFileName_RS" : "PreprocessedData/func_run1_df.nii"
     "sliceOrderFile_RS" : "slice_order_RS.txt"
     "refSlice" : 1
}
```

**slice_order_RS.txt:** (one number per line, 1-36 for 36 slices)
```
1
2
3
...
36
```

---

## Installation

### 1. Create Python Environment

```bash
conda create -n DLRS_CVR_BAT_env python=3.9
conda activate DLRS_CVR_BAT_env
```

### 2. Install Python Dependencies

```bash
./install_dependencies.sh
```

Or manually:
```bash
pip install torch==2.0.1
pip install "numpy<2.0,>=1.23.0"
pip install nibabel==5.1.0
pip install matplotlib==3.7.1
```

**Important:** NumPy must be < 2.0 for compatibility with NiBabel.

### 3. Install MATLAB Dependencies

- Download and install SPM12 into `./preprocessing/spm12/`
- Download SUIT toolbox (v3.5) into `./preprocessing/spm12/toolbox/suit-3.5/`
  - SUIT: https://github.com/jdiedrichsen/suit/releases/tag/3.5

### 4. Download Pre-trained Model

Contact Hanzhang Lu (hanzhang.lu@jhu.edu) to request the pre-trained model weights.
Place `pretrained_DLRS_CVR_BAT_model.pt` in the `./model/` directory.

---

## Usage

### Step 1: Prepare Data

Ensure your data follows the structure above. Each subject needs:
- BOLD functional data (4D NIfTI)
- Structural T1/MPRAGE (3D NIfTI)
- Configuration files (parameter_RS.txt, slice_order_RS.txt)

**Note:** The MATLAB code auto-detects NIfTI (.nii) or ANALYZE (.img) formats.

### Step 2: Run MATLAB Preprocessing

This step creates the 136-channel feature maps required by the deep learning model.

```bash
cd preprocessing
matlab -nodisplay -r 'run rs_running.m; exit'
```

**What it does:**
1. Realigns and normalizes BOLD data to MNI space
2. Extracts cerebellum signal as surrogate CO2
3. Calculates correlation maps with 134 brain ROIs (Neuromorphometrics atlas)
4. Performs GLM regression to compute CVR coefficients
5. Calculates voxel-wise BAT using delay analysis
6. Combines all features into 136-channel 2D slices
7. Outputs to `data/<subject>/DLRS_input/DLRS_input_layer_001.nii` through `091.nii`

**Processing time:** 30-60 minutes per subject

### Step 3: Run Python Inference

```bash
python ./src/DLRS_CVR_BAT_inference.py
```

**What it does:**
1. Loads all DLRS_input slices (136-channel 2D images)
2. Runs through pre-trained UNet model
3. Generates CVR and BAT predictions for each slice
4. Reconstructs 3D volumes
5. Saves outputs to `data/output/<subject>/`

**Processing time:** 5-10 minutes for all subjects

### Step 4: View Results

Results are saved in `data/output/`:
```
data/output/
├── subject1/
│   ├── DLRS_CVR.nii    # CVR map (% BOLD/mmHg CO2)
│   └── DLRS_BAT.nii    # BAT map (seconds)
└── subject2/
    ├── DLRS_CVR.nii
    └── DLRS_BAT.nii
```

Visualize with:
- **Python Script (Recommended):** `python visualize_results.py` - Creates 2×3 grid comparing subjects
- **FSLeyes:** `fsleyes data/output/sub-03a/DLRS_CVR.nii`
- **AFNI:** `afni data/output/sub-03a/`
- **SPM:** Load in MATLAB SPM viewer

---

## Interpreting Results

### CVR Map (DLRS_CVR.nii)
- **Units:** % BOLD signal change per mmHg CO2
- **Interpretation:** Higher values = greater vascular responsiveness
- **Typical range:** 0.1 - 0.5 %/mmHg
- **Clinical use:** Identifies regions with impaired vascular reactivity

### BAT Map (DLRS_BAT.nii)
- **Units:** Seconds
- **Interpretation:** Hemodynamic delay relative to reference
- **Typical range:** -5 to +5 seconds
- **Clinical use:** Detects perfusion delays indicating stenosis or collateral flow


## File Structure

```
DLRS_CVR_BAT/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── install_dependencies.sh            # Installation script
├── check_setup.py                     # Setup verification
├── data/                              # Subject data
│   ├── <subject>/
│   │   ├── parameter_RS.txt          # Config file
│   │   ├── slice_order_RS.txt        # Slice order
│   │   ├── PreprocessedData/         # Input data
│   │   │   ├── func_run1_df.nii     # BOLD
│   │   │   └── struct_rf.nii        # T1
│   │   └── DLRS_input/               # Created by preprocessing
│   │       └── DLRS_input_layer_*.nii
│   └── output/                        # Final results
│       └── <subject>/
│           ├── DLRS_CVR.nii
│           └── DLRS_BAT.nii
├── src/                               # Python source code
│   ├── DLRS_CVR_BAT_inference.py     # Main inference script
│   ├── DLRS_CVR_BAT_model.py         # UNet model definition
│   └── DLRS_CVR_BAT_train.py         # Training script (reference)
├── preprocessing/                     # MATLAB preprocessing
│   ├── rs_running.m                  # Main entry point
│   ├── utils/                        # Preprocessing functions
│   │   ├── RS_preprocessing.m        # Image processing
│   │   ├── RS_BAT.m                  # BAT calculation
│   │   └── RS_CVR_corrMap.m          # CVR correlation maps
│   └── spm12/                        # SPM12 installation
│       └── toolbox/
│           └── suit-3.5/             # SUIT toolbox
├── model/
│   └── pretrained_DLRS_CVR_BAT_model.pt  # Pre-trained weights
└── figure/
    └── Figure1.png                    # Network architecture
```

---

## Technical Details

### Feature Engineering (MATLAB Preprocessing)

The 136 input channels consist of:

1. **Channels 1-134:** Correlation maps
   - Each channel = correlation between voxel timeseries and average signal from one of 134 brain ROIs
   - ROIs from Neuromorphometrics atlas
   - Captures functional connectivity patterns

2. **Channel 135:** Mean BOLD signal (CVR beta0)
   - Average BOLD intensity after filtering and detrending

3. **Channel 136:** CVR coefficient (CVR beta1)
   - Slope from GLM regression: BOLD ~ surrogate_CO2
   - Represents voxel-wise CVR estimate

4. **BAT information:** Incorporated from voxel-shift analysis
   - Calculated by shifting surrogate CO2 timeseries -9 to +9 seconds
   - Finding optimal delay that maximizes correlation

### Deep Learning Model

**Architecture:** U-Net with dual output heads
- **Encoder:** 5 levels with downsampling (conv + batch norm + ReLU)
- **Decoder:** 5 levels with upsampling (upconv mode)
- **Output:** 2 channels via separate conv layers (CVR + BAT)
- **Parameters:** ~10M trainable parameters

**Training:** Model trained on paired hypercapnic and resting-state data
- Supervised learning using CO2-calibrated CVR as ground truth
- Loss function: MSE on both CVR and BAT predictions

## Important Notes

### Cannot Skip MATLAB Preprocessing
The Python model requires the 136-channel feature maps created by MATLAB preprocessing. You cannot:
- Feed raw BOLD data directly to the Python model
- Skip the preprocessing step
- Use only AFNI/FSL preprocessed data without MATLAB processing

The MATLAB stage performs essential feature engineering that the model was trained on.

### Data Format Compatibility
The patched MATLAB code auto-detects and handles both:
- NIfTI format (.nii files)
- ANALYZE format (.img/.hdr files)

No manual conversion needed.

### Slice Timing
Ensure `slice_order_RS.txt` matches your BOLD data:
- Count slices in your BOLD data
- Create a file with numbers 1 through N (one per line)
- Sequential order assumes slices acquired bottom-to-top

---

## Verification & Testing

### Check Setup
```bash
python check_setup.py
```

This verifies:
- Directory structure is correct
- Model file exists
- Python dependencies installed correctly
- MATLAB dependencies available
- Subject data properly configured

### Test Run
Test on one subject first:
```bash
# Run preprocessing for one subject only
cd preprocessing
matlab -nodisplay -r "RS_preprocessing('../data', 'subject1', pwd); RS_BAT('../data', 'subject1', pwd); RS_CVR_corrMap('../data', 'subject1', pwd); exit"

# Check output
ls -la data/subject1/DLRS_input/

# Run inference
cd ..
python ./src/DLRS_CVR_BAT_inference.py
```

---

## Common Workflows

### Processing Multiple Subjects
The pipeline automatically processes all subject folders in `data/`:
```bash
cd preprocessing
matlab -nodisplay -r 'run rs_running.m; exit'
```

### Processing Single Subject
```matlab
cd preprocessing
RS_preprocessing('../data', 'subject_name', pwd);
RS_BAT('../data', 'subject_name', pwd);
RS_CVR_corrMap('../data', 'subject_name', pwd);
```

### Re-running Failed Subject
If preprocessing failed partway through:
1. Delete incomplete `DLRS_input` folder
2. Delete intermediate files in `PreprocessedData_RS` folder
3. Re-run preprocessing for that subject

---

## Updates & Changelog

### v1.0.1 (Current)
- Fixed .DS_Store processing error in rs_running.m
- Added auto-detection for NIfTI (.nii) and ANALYZE (.img) formats
- Updated for NumPy 2.0 compatibility
- Added comprehensive documentation and verification scripts

### v1.0.0 (Original Release)
- Initial release with paper publication
- MATLAB preprocessing pipeline
- Python inference with pre-trained model

---

**Last Updated:** December 2024
