#!/usr/bin/env python3
"""
Visualize DLRS CVR and BAT Results
Creates 2x3 grid: 2 subjects × 3 maps (Anatomy, CVR, BAT)
"""

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from matplotlib import cm


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


def main():
    # Paths
    data_dir = './data'
    output_dir = os.path.join(data_dir, 'output')
    subjects = ['sub-03a', 'sub-07a']

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('DLRS CVR and BAT Results (SCI Subjects)', fontsize=16, fontweight='bold')

    for row, subject in enumerate(subjects):
        subject_dir = os.path.join(output_dir, subject)

        # Load data
        cvr_file = os.path.join(subject_dir, 'DLRS_CVR.nii')
        bat_file = os.path.join(subject_dir, 'DLRS_BAT.nii')

        cvr_data = load_nifti(cvr_file)
        bat_data = load_nifti(bat_file)

        # Extract middle axial slices (Z-axis)
        cvr_slice = get_middle_slice(cvr_data, axis=2)
        bat_slice = get_middle_slice(bat_data, axis=2)

        # Create brain mask from CVR data
        brain_mask = create_brain_mask(cvr_slice)

        # Apply mask to data (set background to NaN for transparency)
        cvr_masked = np.where(brain_mask, cvr_slice, np.nan)
        bat_masked = np.where(brain_mask, bat_slice, np.nan)

        # Plot 1: Anatomical outline (from CVR mask)
        ax1 = axes[row, 0]
        ax1.imshow(brain_mask, cmap='gray', origin='lower')
        ax1.set_title(f'{subject}\nBrain Mask', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Plot 2: CVR map
        ax2 = axes[row, 1]
        im2 = ax2.imshow(cvr_masked.T, cmap='jet', origin='lower',
                         vmin=0, vmax=0.4, interpolation='bilinear')
        ax2.set_title(f'{subject}\nCVR (%BOLD/mmHg)', fontsize=12, fontweight='bold')
        ax2.axis('off')
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('CVR', rotation=270, labelpad=15)

        # Plot 3: BAT map
        ax3 = axes[row, 2]
        im3 = ax3.imshow(bat_masked.T, cmap='coolwarm', origin='lower',
                         vmin=0, vmax=10, interpolation='bilinear')
        ax3.set_title(f'{subject}\nBAT (seconds)', fontsize=12, fontweight='bold')
        ax3.axis('off')
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label('BAT (s)', rotation=270, labelpad=15)

        # Add row label
        axes[row, 0].text(-0.15, 0.5, f'{subject}',
                          transform=axes[row, 0].transAxes,
                          fontsize=14, fontweight='bold', rotation=90,
                          verticalalignment='center')

    # Add column labels at bottom
    col_labels = ['Brain Anatomy', 'Cerebrovascular Reactivity (CVR)', 'Bolus Arrival Time (BAT)']
    for col, label in enumerate(col_labels):
        axes[1, col].text(0.5, -0.15, label, transform=axes[1, col].transAxes,
                          fontsize=12, fontweight='bold', ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save figure
    output_file = os.path.join(data_dir, 'CVR_BAT_Results.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_file}", flush=True)
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
        print(f"  CVR Map:", flush=True)
        print(f"    Mean ± SD: {cvr_brain.mean():.3f} ± {cvr_brain.std():.3f} %BOLD/mmHg", flush=True)
        print(f"    Range: [{cvr_brain.min():.3f}, {cvr_brain.max():.3f}]", flush=True)
        print(f"    Median: {np.median(cvr_brain):.3f}", flush=True)

        print(f"  BAT Map:", flush=True)
        print(f"    Mean ± SD: {bat_brain.mean():.2f} ± {bat_brain.std():.2f} seconds", flush=True)
        print(f"    Range: [{bat_brain.min():.2f}, {bat_brain.max():.2f}]", flush=True)
        print(f"    Median: {np.median(bat_brain):.2f}", flush=True)

    print("\n" + "="*60, flush=True)
    print("INTERPRETATION GUIDE", flush=True)
    print("="*60, flush=True)
    print("""
CVR (Cerebrovascular Reactivity):
  - Normal: 0.15 - 0.30 %BOLD/mmHg
  - Reduced: < 0.15 (impaired vascular function)
  - High: > 0.30 (hyperperfusion)

BAT (Bolus Arrival Time):
  - Normal: 2 - 6 seconds
  - Delayed: > 6 seconds (stenosis, collateral flow)
  - Early: < 2 seconds (AVM, hyperperfusion)
    """)


if __name__ == "__main__":
    main()
