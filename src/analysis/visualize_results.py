#!/usr/bin/env python3
"""
Visualize DLRS CVR and BAT Results
Creates grid: subjects x 3 maps (Anatomy, CVR, BAT)
"""

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from matplotlib import cm

# Canonical DLRS display settings, identical to visualize_group_comparison.py
# and model_result_group.py.  The DLRS outputs are RELATIVE (arbitrary) units
# from a 5*tanh() layer, NOT %BOLD/mmHg or seconds, and they are signed, so a
# symmetric range and a symmetric tick set are required.
CVR_VMIN, CVR_VMAX = -0.8, 0.8
BAT_VMIN, BAT_VMAX = -1.2, 1.2
CVR_TICKS = [-0.8, -0.4, 0.0, 0.4, 0.8]
BAT_TICKS = [-1.2, -0.6, 0.0, 0.6, 1.2]
CVR_CMAP = 'hot'
BAT_CMAP = 'jet'


def style_cbar(cb, ticks, label=None, fontsize=10):
    """Force a colour bar to print its exact end points."""
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{t:g}' for t in ticks])
    cb.ax.tick_params(labelsize=max(fontsize - 2, 6))
    if label is not None:
        cb.set_label(label, rotation=270, labelpad=15, fontsize=fontsize)
    return cb


def load_nifti(filepath):
    """Load NIfTI file and return data array"""
    img = nib.load(filepath)
    data = img.get_fdata()
    return data


def get_middle_slice(data, axis=2):
    """Extract middle slice along specified axis"""
    middle_idx = data.shape[axis] // 2
    if axis == 0:
        return data[middle_idx, :, :]
    elif axis == 1:
        return data[:, middle_idx, :]
    else:  # axis == 2
        return data[:, :, middle_idx]


def create_brain_mask(data, threshold=0.01):
    """Create binary mask from data"""
    return np.abs(data) > threshold


def clean_mask_slice(mask_2d):
    """Remove salt-and-pepper noise from a 2D mask slice.

    Uses a flood-fill from the centroid to keep only the main connected
    brain region, discarding isolated speckle voxels.
    """
    mask = mask_2d.astype(bool).copy()
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return mask
    cy, cx = int(ys.mean()), int(xs.mean())
    if not mask[cy, cx]:
        dists = (ys - cy) ** 2 + (xs - cx) ** 2
        nearest = np.argmin(dists)
        cy, cx = ys[nearest], xs[nearest]

    queue = [(cy, cx)]
    visited[cy, cx] = True
    while queue:
        y, x = queue.pop(0)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    return visited


def main():
    # Paths - resolve relative to script location
    # This script lives in src/analysis, so the project root is two levels up.
    script_dir = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(script_dir, 'data')
    output_dir = os.path.join(data_dir, 'output')

    # Auto-detect subjects
    if not os.path.exists(output_dir):
        print(f"Error: Output directory not found: {output_dir}")
        print("Please run the inference first: python src/DLRS_CVR_BAT_inference.py")
        return

    subjects = sorted([
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('sub-')
    ])

    if len(subjects) == 0:
        print(f"Error: No subject output folders found in {output_dir}")
        return

    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")
    n_subjects = len(subjects)

    # Create figure
    fig, axes = plt.subplots(n_subjects, 3, figsize=(15, n_subjects * 5))
    if n_subjects == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('DLRS CVR and BAT Results', fontsize=16, fontweight='bold')

    for row, subject in enumerate(subjects):
        subject_dir = os.path.join(output_dir, subject)
        subject_data_dir = os.path.join(data_dir, subject)

        # Load data
        cvr_file = os.path.join(subject_dir, 'DLRS_CVR.nii')
        bat_file = os.path.join(subject_dir, 'DLRS_BAT.nii')

        cvr_data = load_nifti(cvr_file)
        bat_data = load_nifti(bat_file)

        # Extract middle axial slices (Z-axis)
        cvr_slice = get_middle_slice(cvr_data, axis=2)
        bat_slice = get_middle_slice(bat_data, axis=2)

        # Create brain mask for display
        # The AFNI pipeline produces model output across the full volume,
        # so we use the preprocessing brain mask to get the brain shape.
        mask_file = os.path.join(subject_data_dir, 'mask', 'brainMask_RS.nii')
        if os.path.exists(mask_file):
            mask_data = load_nifti(mask_file)
            brain_mask = get_middle_slice(mask_data, axis=2) > 0.5
            # Clean salt-and-pepper from resampling artifacts
            brain_mask = clean_mask_slice(brain_mask)
        else:
            # Fallback if no preprocessing mask exists
            brain_mask = create_brain_mask(cvr_slice)

        # Apply mask to data (set background to NaN for transparency)
        cvr_masked = np.where(brain_mask, cvr_slice, np.nan)
        bat_masked = np.where(brain_mask, bat_slice, np.nan)

        # Plot 1: Anatomical outline (from brain mask)
        ax1 = axes[row, 0]
        ax1.imshow(brain_mask, cmap='gray', origin='lower')
        ax1.set_title(f'{subject}\nBrain Mask', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Plot 2: CVR map
        ax2 = axes[row, 1]
        im2 = ax2.imshow(cvr_masked.T, cmap=CVR_CMAP, origin='lower',
                         vmin=CVR_VMIN, vmax=CVR_VMAX, interpolation='bilinear')
        ax2.set_title(f'{subject}\nCVR (a.u.)', fontsize=12, fontweight='bold')
        ax2.axis('off')
        style_cbar(plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04,
                                extend='both'), CVR_TICKS, 'CVR (a.u.)')

        # Plot 3: BAT map
        ax3 = axes[row, 2]
        im3 = ax3.imshow(bat_masked.T, cmap=BAT_CMAP, origin='lower',
                         vmin=BAT_VMIN, vmax=BAT_VMAX, interpolation='bilinear')
        ax3.set_title(f'{subject}\nBAT (a.u.)', fontsize=12, fontweight='bold')
        ax3.axis('off')
        style_cbar(plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04,
                                extend='both'), BAT_TICKS, 'BAT (a.u.)')

        # Add row label
        axes[row, 0].text(-0.15, 0.5, f'{subject}',
                          transform=axes[row, 0].transAxes,
                          fontsize=14, fontweight='bold', rotation=90,
                          verticalalignment='center')

    # Add column labels at bottom
    col_labels = ['Brain Anatomy', 'Cerebrovascular Reactivity (CVR)', 'Bolus Arrival Time (BAT)']
    for col, label in enumerate(col_labels):
        axes[n_subjects - 1, col].text(0.5, -0.15, label,
                          transform=axes[n_subjects - 1, col].transAxes,
                          fontsize=12, fontweight='bold', ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save figure
    output_file = os.path.join(data_dir, 'CVR_BAT_Results.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}", flush=True)
    plt.close()

    # Print summary statistics
    print("\n" + "="*60, flush=True)
    print("SUMMARY STATISTICS", flush=True)
    print("="*60, flush=True)

    for subject in subjects:
        subject_dir = os.path.join(output_dir, subject)

        cvr_file = os.path.join(subject_dir, 'DLRS_CVR.nii')
        bat_file = os.path.join(subject_dir, 'DLRS_BAT.nii')

        cvr_data = load_nifti(cvr_file)
        bat_data = load_nifti(bat_file)

        # Mask out zeros
        cvr_brain = cvr_data[np.abs(cvr_data) > 0.01]
        bat_brain = bat_data[np.abs(bat_data) > 0.01]

        print(f"\n{subject}:", flush=True)
        if len(cvr_brain) == 0:
            print(f"  CVR Map: No non-zero voxels (subject not processed?)", flush=True)
        else:
            print(f"  CVR Map:", flush=True)
            print(f"    Mean +/- SD: {cvr_brain.mean():.3f} +/- {cvr_brain.std():.3f} a.u.", flush=True)
            print(f"    Range: [{cvr_brain.min():.3f}, {cvr_brain.max():.3f}]", flush=True)
            print(f"    Median: {np.median(cvr_brain):.3f}", flush=True)

        if len(bat_brain) == 0:
            print(f"  BAT Map: No non-zero voxels (subject not processed?)", flush=True)
        else:
            print(f"  BAT Map:", flush=True)
            print(f"    Mean +/- SD: {bat_brain.mean():.2f} +/- {bat_brain.std():.2f} a.u.", flush=True)
            print(f"    Range: [{bat_brain.min():.2f}, {bat_brain.max():.2f}]", flush=True)
            print(f"    Median: {np.median(bat_brain):.2f}", flush=True)

    print("\n" + "="*60, flush=True)
    print("INTERPRETATION GUIDE (Resting-State Analysis)", flush=True)
    print("="*60, flush=True)
    print("""
NOTE: DLRS outputs are in RELATIVE units, not absolute %BOLD/mmHg [1].

CVR (Cerebrovascular Reactivity):
  - Positive values: BOLD signal correlates with reference
  - Negative values: Inverse correlation (vascular steal or noise)
  - Model output is bounded to [-5, +5] by the 5*tanh() layer; the
    within-brain bulk falls inside the -0.8 to +0.8 display range used here

  Clinical Interpretation:
    Normal function:   Positive CVR (healthy vascular response)
    Reduced function:  Near-zero CVR (impaired reactivity)
    Steal phenomenon:  Negative CVR (paradoxical response) [2]

BAT (Relative Delay / Bolus Arrival Time):
  - This is a RELATIVE delay vs cerebellum reference signal
  - Negative: Voxel BOLD leads the reference (faster response)
  - Positive: Voxel BOLD lags the reference (slower response)

  NOTE: the model's BAT output is NOT in seconds.  It is a relative,
  arbitrary-unit delay; the display range used here is -1.2 to +1.2 a.u.
  Converting to seconds would require calibration against a timed
  stimulus, which this resting-state pipeline does not have.  The
  qualitative reading below (from [3]) therefore applies to the SIGN and
  the relative ordering of values, not to absolute thresholds:
    Near zero          timing close to the reference signal
    More negative      earlier arrival than the reference
    More positive      later arrival than the reference

REFERENCES:
[1] Hou et al. (2023) npj Digital Medicine. doi:10.1038/s41746-023-00859-y
[2] Pillai & Mikulis (2015) AJNR. doi:10.3174/ajnr.A4066
[3] Siegel et al. (2016) Neuroimage. doi:10.1016/j.neuroimage.2015.12.011
    """)


if __name__ == "__main__":
    main()
