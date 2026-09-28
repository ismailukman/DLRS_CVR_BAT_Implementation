#!/usr/bin/env python3
"""
model_result_group.py - Quantitative evaluation of DLRS CVR/BAT predictions
against GLM-derived ground-truth maps for HC and SCI groups.

Analyses performed:
  1. Voxelwise metrics: Pearson correlation, SSIM, PSNR, RMSE
  2. Hemispheric comparison (left vs right) with two-way ANOVA (Group x Hemisphere)
  3. Bland-Altman plots (DLRS vs GLM)
  4. Regional MCA-territory CVR and BAT analysis

Outputs (saved to figure/):
  - model_eval_metrics.csv         Per-subject metric table
  - model_eval_barplots.png        Grouped bar charts of metrics
  - bland_altman_cvr.png           Bland-Altman CVR
  - bland_altman_bat.png           Bland-Altman BAT
  - hemispheric_comparison.png     L vs R hemisphere box plots
  - mca_regional_analysis.png      MCA-territory CVR/BAT
  - model_eval_full_report.csv     All numeric results in one file

Usage:
    conda activate cvr_bat
    python model_result_group.py
"""

import os
import csv
import glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CVR_VMIN, CVR_VMAX = -0.8, 0.8
BAT_VMIN, BAT_VMAX = -1.2, 1.2
# Explicit colour-bar ticks so the exact end points are always printed
# (matplotlib's auto-locator drops -1.2/+1.2 in favour of -1.0/+1.0).
CVR_TICKS = [-0.8, -0.4, 0.0, 0.4, 0.8]
BAT_TICKS = [-1.2, -0.6, 0.0, 0.6, 1.2]
CVR_CMAP = 'hot'
BAT_CMAP = 'jet'


def style_cbar(cb, ticks, label=None, fontsize=10):
    """Force a colour bar to print its exact end points (see CVR_TICKS)."""
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{t:g}' for t in ticks])
    cb.ax.tick_params(labelsize=max(fontsize - 2, 6))
    if label is not None:
        cb.set_label(label, fontsize=fontsize)
    return cb

# Display range for the *normalised* (robust z-score) maps used in the
# DLRS-vs-GLM visual comparison and Bland-Altman plots.  Because the GLM
# reference and the DLRS output live on completely different scales
# (raw GLM CVR beta ~[-2e4, 2.6e4]; GLM BAT delay [0, 9] vs. DLRS tanh
# output [-5, 5]), both maps are converted to within-brain robust
# z-scores before display so they can share a single intensity scale.
# +/-3 z captures the central structure while clipping GLM outlier voxels.
Z_VMIN, Z_VMAX = -3.0, 3.0
Z_TICKS = [-3.0, -1.5, 0.0, 1.5, 3.0]

# For the Bland-Altman agreement analysis, voxels whose robust z-score
# exceeds this magnitude in either map are treated as non-physiological
# fit artifacts (they arise almost exclusively from ill-conditioned GLM
# beta estimates) and are excluded so the bias and limits of agreement
# reflect the bulk of brain tissue rather than a handful of extreme voxels.
BA_ZCLIP = 8.0

# Neuromorphometrics labels approximating MCA perfusion territory.
# These cover lateral frontal, parietal, temporal cortices, insula,
# and basal ganglia - regions primarily supplied by the MCA.
#
# Left-hemisphere (even or L-specific) / Right-hemisphere (odd or R-specific)
# Mapping based on standard Neuromorphometrics nomenclature:
#   Frontal: superior/middle/inferior frontal gyri, precentral gyrus
#   Parietal: postcentral, supramarginal, angular gyri, superior parietal
#   Temporal: superior/middle/inferior temporal gyri, fusiform
#   Insula, putamen, caudate, pallidum
MCA_LABELS_LEFT = [
    3, 5, 23, 25, 27, 29,          # frontal (L)
    35, 42, 44, 46,                  # parietal (L)
    48, 50, 52, 54, 56,             # temporal (L)
    60, 62,                          # insula, basal ganglia (L)
    80, 84, 86, 88,                  # additional cortical (L)
    100, 102, 104, 106, 108,         # associative cortex (L)
    110, 112, 116, 118, 120,         # lateral cortex (L)
]

MCA_LABELS_RIGHT = [
    4, 6, 24, 26, 28, 30,           # frontal (R)
    34, 43, 45, 47,                  # parietal (R)
    51, 53, 55, 57, 59,             # temporal (R)
    61, 63,                          # insula, basal ganglia (R)
    81, 85, 87, 89,                  # additional cortical (R)
    101, 103, 105, 107, 109,         # associative cortex (R)
    111, 113, 117, 119, 121,         # lateral cortex (R)
]

MCA_LABELS_ALL = sorted(set(MCA_LABELS_LEFT + MCA_LABELS_RIGHT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_nifti(fpath):
    """Load NIfTI (.nii or .hdr/.img pair) and return data array."""
    return nib.load(fpath).get_fdata()


def find_subjects(output_dir, group_name):
    group_out = os.path.join(output_dir, group_name)
    if not os.path.isdir(group_out):
        return []
    return sorted([
        s for s in os.listdir(group_out)
        if os.path.isdir(os.path.join(group_out, s))
        and os.path.exists(os.path.join(group_out, s, 'DLRS_CVR.nii'))
    ])


def get_brain_mask(data_dir, group_name, subj_id):
    mf = os.path.join(data_dir, group_name, subj_id, 'mask', 'brainMask_RS.nii')
    if os.path.exists(mf):
        return load_nifti(mf) > 0.5
    return None


def bold_tsd_slice(data_dir, group_name, subj_id, z=None, n_frames=120):
    """BOLD temporal-standard-deviation slice as a structural reference,
    matching the anatomical column used in the representative Figure 3.
    The preprocessed resting-state BOLD frames are already on the 2 mm MNI
    grid of the DLRS maps, so no resampling is needed."""
    fs = sorted(glob.glob(os.path.join(
        data_dir, group_name, subj_id, 'PreprocessedData_RS',
        'drwarfunc_run_combined-*-001.img')))
    if not fs:
        return None
    sel = np.linspace(0, len(fs) - 1, min(n_frames, len(fs))).astype(int)
    stack = np.stack([np.squeeze(load_nifti(fs[i])) for i in sel], axis=-1)
    tsd = stack.std(axis=-1)
    if z is None:
        z = tsd.shape[2] // 2
    return tsd[:, :, z]


def load_gt(data_dir, group_name, subj_id):
    """Load GLM ground-truth CVR and BAT (co2delay) maps."""
    cvr_dir = os.path.join(data_dir, group_name, subj_id,
                           'CVR_voxelshift_etco2_cerebellum')
    gt_cvr_path = os.path.join(cvr_dir, 'drwarfunc_run_combined_s8_CVR.img')
    gt_bat_path = os.path.join(cvr_dir, 'drwarfunc_run_combined_s8_co2delay.img')

    # Try .img first, then .nii
    if not os.path.exists(gt_cvr_path):
        gt_cvr_path = gt_cvr_path.replace('.img', '.nii')
    if not os.path.exists(gt_bat_path):
        gt_bat_path = gt_bat_path.replace('.img', '.nii')

    if not os.path.exists(gt_cvr_path) or not os.path.exists(gt_bat_path):
        return None, None
    return np.squeeze(load_nifti(gt_cvr_path)), np.squeeze(load_nifti(gt_bat_path))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def pearson_corr(a, b):
    """Spatial Pearson cross-correlation between two masked vectors."""
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 10:
        return np.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def compute_ssim(a, b, data_range=None):
    """Structural Similarity Index Measure (SSIM) for 3-D volumes.
    Simplified Wang et al. (2004) implementation without requiring
    scikit-image, using sliding-window means and variances."""
    valid = np.isfinite(a) & np.isfinite(b)
    a_v, b_v = a[valid].astype(np.float64), b[valid].astype(np.float64)
    if len(a_v) < 10:
        return np.nan
    if data_range is None:
        data_range = max(a_v.max() - a_v.min(), b_v.max() - b_v.min())
        if data_range == 0:
            return 1.0
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_a, mu_b = a_v.mean(), b_v.mean()
    sig_a, sig_b = a_v.std(), b_v.std()
    sig_ab = np.mean((a_v - mu_a) * (b_v - mu_b))
    ssim_val = ((2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)) / \
               ((mu_a**2 + mu_b**2 + C1) * (sig_a**2 + sig_b**2 + C2))
    return float(ssim_val)


def compute_psnr(pred, gt):
    """Peak signal-to-noise ratio.  PSNR = 10.log10(MAX^2 / MSE)."""
    valid = np.isfinite(pred) & np.isfinite(gt)
    if valid.sum() < 10:
        return np.nan
    mse = np.mean((pred[valid] - gt[valid]) ** 2)
    if mse < 1e-15:
        return np.inf
    max_val = np.max(np.abs(pred[valid]))
    if max_val < 1e-15:
        return np.nan
    return float(10.0 * np.log10(max_val ** 2 / mse))


def compute_rmse(a, b):
    """Root mean square error."""
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 10:
        return np.nan
    return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))


def zscore_in_mask(vol, mask, robust=True):
    """Convert a map to within-brain z-scores so that images on very
    different native scales (e.g. raw GLM beta vs. tanh-scaled DLRS output)
    can be displayed and compared on a single common scale.

    A *robust* z-score (median / median-absolute-deviation) is used by
    default because the GLM CVR reference contains a small number of
    extreme outlier voxels (|beta| up to ~2.6e4 from poorly conditioned
    fits) that would otherwise dominate an ordinary mean/SD normalisation
    and collapse all real structure toward zero.  Spatial Pearson
    correlation is invariant to this affine rescaling, so the accuracy
    metrics in Table~\\ref{tab:metrics} are unaffected."""
    v = vol[mask]
    v = v[np.isfinite(v)]
    if v.size < 10:
        return np.full_like(vol, np.nan, dtype=float)
    if robust:
        center = np.median(v)
        mad = np.median(np.abs(v - center))
        scale = 1.4826 * mad if mad > 0 else v.std()
    else:
        center = v.mean()
        scale = v.std()
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return (vol - center) / scale


# ---------------------------------------------------------------------------
# Bland-Altman
# ---------------------------------------------------------------------------
def bland_altman_plot(ax, dlrs_vals, gt_vals, title='', color='#4DBEEE',
                      ylabel_units=''):
    """Draw a Bland-Altman plot on the given axes."""
    mean_val = (dlrs_vals + gt_vals) / 2.0
    diff_val = dlrs_vals - gt_vals
    md = np.nanmean(diff_val)
    sd = np.nanstd(diff_val)

    ax.scatter(mean_val, diff_val, alpha=0.15, s=4, color=color, rasterized=True)
    ax.axhline(md, color='black', linewidth=1.2, label=f'Mean diff = {md:.4f}')
    ax.axhline(md + 1.96 * sd, color='red', linestyle='--', linewidth=0.9,
               label=f'+1.96 SD = {md + 1.96*sd:.4f}')
    ax.axhline(md - 1.96 * sd, color='red', linestyle='--', linewidth=0.9,
               label=f'-1.96 SD = {md - 1.96*sd:.4f}')
    ax.set_xlabel(f'Mean of DLRS & GLM {ylabel_units}', fontsize=10)
    ax.set_ylabel(f'Difference (DLRS - GLM) {ylabel_units}', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # This script lives in src/analysis, so the project root is two levels up.
    script_dir = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(script_dir, 'data')
    output_dir = os.path.join(data_dir, 'output')
    figure_dir = os.path.join(script_dir, 'figure')
    # Neuromorphometrics atlas (moved under preprocessing/atlas/; keep a
    # fallback to the previous location for backward compatibility).
    atlas_path = os.path.join(script_dir, 'preprocessing', 'atlas',
                              'labels_Neuromorphometrics',
                              'wlabels_Neuromorphometrics_unique.nii')
    if not os.path.exists(atlas_path):
        atlas_path = os.path.join(script_dir, 'preprocessing',
                                  'labels_Neuromorphometrics',
                                  'wlabels_Neuromorphometrics_unique.nii')
    os.makedirs(figure_dir, exist_ok=True)

    # Load atlas (91x109x91)
    atlas_data = load_nifti(atlas_path).astype(int)
    midline_x = atlas_data.shape[0] // 2   # x=45 splits L/R

    # Discover subjects
    groups = {
        'HC':  ('HC_subj',  find_subjects(output_dir, 'HC_subj')),
        'SCI': ('SCI_subj', find_subjects(output_dir, 'SCI_subj')),
    }

    print('=' * 70)
    print('DLRS CVR/BAT - Model Evaluation & Group Comparison')
    print('=' * 70)
    for glabel, (gdir, subjs) in groups.items():
        print(f"  {glabel}: {len(subjs)} subjects - {', '.join(subjs)}")

    # ======================================================================
    # 1. Per-subject voxelwise metrics  (Pearson, SSIM, PSNR, RMSE)
    # ======================================================================
    print(f"\n{'-' * 70}")
    print('1. Voxelwise Image Quality Metrics (DLRS vs GLM ground truth)')
    print(f"{'-' * 70}")

    metric_rows = []   # for CSV
    hdr = (f"{'Group':<6} {'Subject':<18} "
           f"{'CVR_r':>7} {'CVR_SSIM':>9} {'CVR_PSNR':>9} {'CVR_RMSE':>9} "
           f"{'BAT_r':>7} {'BAT_SSIM':>9} {'BAT_PSNR':>9} {'BAT_RMSE':>9}")
    print(hdr)

    for glabel, (gdir, subjs) in groups.items():
        for sid in subjs:
            dlrs_cvr = load_nifti(os.path.join(output_dir, gdir, sid, 'DLRS_CVR.nii'))
            dlrs_bat = load_nifti(os.path.join(output_dir, gdir, sid, 'DLRS_BAT.nii'))
            gt_cvr, gt_bat = load_gt(data_dir, gdir, sid)

            mask = get_brain_mask(data_dir, gdir, sid)
            if mask is None:
                mask = np.abs(dlrs_cvr) > 0.01

            if gt_cvr is None:
                print(f"  {glabel:<6} {sid:<18} - ground-truth not found, skipping")
                continue

            # Mask both
            d_cvr = np.where(mask, dlrs_cvr, np.nan).ravel()
            g_cvr = np.where(mask, gt_cvr, np.nan).ravel()
            d_bat = np.where(mask, dlrs_bat, np.nan).ravel()
            g_bat = np.where(mask, gt_bat, np.nan).ravel()

            row = {
                'group': glabel, 'subject': sid,
                'cvr_pearson': pearson_corr(d_cvr, g_cvr),
                'cvr_ssim':    compute_ssim(d_cvr, g_cvr),
                'cvr_psnr':    compute_psnr(d_cvr, g_cvr),
                'cvr_rmse':    compute_rmse(d_cvr, g_cvr),
                'bat_pearson': pearson_corr(d_bat, g_bat),
                'bat_ssim':    compute_ssim(d_bat, g_bat),
                'bat_psnr':    compute_psnr(d_bat, g_bat),
                'bat_rmse':    compute_rmse(d_bat, g_bat),
            }
            metric_rows.append(row)
            print(f"  {glabel:<6} {sid:<18} "
                  f"{row['cvr_pearson']:>7.4f} {row['cvr_ssim']:>9.4f} "
                  f"{row['cvr_psnr']:>9.2f} {row['cvr_rmse']:>9.4f} "
                  f"{row['bat_pearson']:>7.4f} {row['bat_ssim']:>9.4f} "
                  f"{row['bat_psnr']:>9.2f} {row['bat_rmse']:>9.4f}")

    # Group means
    for glabel in ['HC', 'SCI']:
        g = [r for r in metric_rows if r['group'] == glabel]
        if not g:
            continue
        print(f"  {glabel:<6} {'MEAN':<18} ", end='')
        for key in ['cvr_pearson', 'cvr_ssim', 'cvr_psnr', 'cvr_rmse',
                     'bat_pearson', 'bat_ssim', 'bat_psnr', 'bat_rmse']:
            vals = [r[key] for r in g if np.isfinite(r[key])]
            m = np.mean(vals) if vals else np.nan
            if 'psnr' in key:
                print(f"{m:>9.2f} ", end='')
            else:
                print(f"{m:>9.4f} " if 'rmse' in key or 'ssim' in key else f"{m:>7.4f} ", end='')
        print()

    # Save CSV
    csv_path = os.path.join(figure_dir, 'model_eval_metrics.csv')
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metric_rows[0].keys()) if metric_rows else [])
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"\nSaved: {csv_path}")

    # ======================================================================
    # Figure A: Grouped bar charts of metrics
    # ======================================================================
    metric_names_cvr = ['cvr_pearson', 'cvr_ssim', 'cvr_psnr', 'cvr_rmse']
    metric_names_bat = ['bat_pearson', 'bat_ssim', 'bat_psnr', 'bat_rmse']
    display_names = ['Pearson r', 'SSIM', 'PSNR (dB)', 'RMSE']

    fig_bar, axes_bar = plt.subplots(2, 4, figsize=(22, 9))
    fig_bar.suptitle('DLRS vs GLM: Image Quality Metrics by Group',
                     fontsize=15, fontweight='bold')
    colors_grp = {'HC': '#4DBEEE', 'SCI': '#FF6B6B'}

    for col, (mkey, dname) in enumerate(zip(metric_names_cvr, display_names)):
        ax = axes_bar[0, col]
        for gi, glabel in enumerate(['HC', 'SCI']):
            vals = [r[mkey] for r in metric_rows
                    if r['group'] == glabel and np.isfinite(r[mkey])]
            if vals:
                bp = ax.boxplot([vals], positions=[gi], widths=0.45,
                                patch_artist=True, showmeans=True,
                                meanprops=dict(marker='D', markerfacecolor='k', markersize=6))
                bp['boxes'][0].set_facecolor(colors_grp[glabel])
                bp['boxes'][0].set_alpha(0.7)
                x_jitter = np.random.normal(gi, 0.04, len(vals))
                ax.scatter(x_jitter, vals, color=colors_grp[glabel],
                           edgecolors='k', s=40, zorder=5, alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['HC', 'SCI'])
        ax.set_title(f'CVR {dname}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    for col, (mkey, dname) in enumerate(zip(metric_names_bat, display_names)):
        ax = axes_bar[1, col]
        for gi, glabel in enumerate(['HC', 'SCI']):
            vals = [r[mkey] for r in metric_rows
                    if r['group'] == glabel and np.isfinite(r[mkey])]
            if vals:
                bp = ax.boxplot([vals], positions=[gi], widths=0.45,
                                patch_artist=True, showmeans=True,
                                meanprops=dict(marker='D', markerfacecolor='k', markersize=6))
                bp['boxes'][0].set_facecolor(colors_grp[glabel])
                bp['boxes'][0].set_alpha(0.7)
                x_jitter = np.random.normal(gi, 0.04, len(vals))
                ax.scatter(x_jitter, vals, color=colors_grp[glabel],
                           edgecolors='k', s=40, zorder=5, alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['HC', 'SCI'])
        ax.set_title(f'BAT {dname}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    bar_path = os.path.join(figure_dir, 'model_eval_barplots.png')
    plt.savefig(bar_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {bar_path}")

    # ======================================================================
    # 2. Bland-Altman Plots (pooled across subjects per group)
    # ======================================================================
    print(f"\n{'-' * 70}")
    print('2. Bland-Altman Analysis')
    print(f"{'-' * 70}")

    for modality, mod_label in [('cvr', 'CVR'), ('bat', 'BAT')]:
        n_groups = len([g for g in ['HC', 'SCI'] if any(r['group'] == g for r in metric_rows)])
        fig_ba, axes_ba = plt.subplots(1, n_groups, figsize=(7 * n_groups, 5.5))
        if n_groups == 1:
            axes_ba = [axes_ba]
        fig_ba.suptitle(f'Bland-Altman: DLRS {mod_label} vs GLM {mod_label} '
                        f'(within-brain robust z-scores)',
                        fontsize=14, fontweight='bold')

        idx = 0
        for glabel, (gdir, subjs) in groups.items():
            if not subjs:
                continue
            all_dlrs, all_gt = [], []
            for sid in subjs:
                dlrs = load_nifti(os.path.join(output_dir, gdir, sid,
                                               f'DLRS_{mod_label.upper()}.nii'))
                gt_cvr_data, gt_bat_data = load_gt(data_dir, gdir, sid)
                gt = gt_cvr_data if modality == 'cvr' else gt_bat_data
                if gt is None:
                    continue
                mask = get_brain_mask(data_dir, gdir, sid)
                if mask is None:
                    mask = np.abs(dlrs) > 0.01
                # Normalise each subject's DLRS and GLM map to within-brain
                # robust z-scores so the difference axis is meaningful and
                # not dominated by the GLM reference's native scale.
                dlrs_z = zscore_in_mask(dlrs, mask)
                gt_z = zscore_in_mask(gt, mask)
                brain_vox = mask.ravel().astype(bool)
                all_dlrs.append(dlrs_z.ravel()[brain_vox])
                all_gt.append(gt_z.ravel()[brain_vox])
            if all_dlrs:
                dlrs_pool = np.concatenate(all_dlrs)
                gt_pool = np.concatenate(all_gt)
                # Exclude non-physiological GLM fit-artifact voxels so the
                # agreement statistics reflect the bulk of brain tissue.
                finite = np.isfinite(dlrs_pool) & np.isfinite(gt_pool)
                keep = (finite & (np.abs(dlrs_pool) <= BA_ZCLIP)
                        & (np.abs(gt_pool) <= BA_ZCLIP))
                n_excl = int(finite.sum() - keep.sum())
                pct_excl = 100.0 * n_excl / max(int(finite.sum()), 1)
                dlrs_pool = dlrs_pool[keep]
                gt_pool = gt_pool[keep]
                print(f"    {glabel} {modality.upper()}: excluded "
                      f"{n_excl} outlier voxels (|z|>{BA_ZCLIP:g}, "
                      f"{pct_excl:.2f}% of brain)")
                # Subsample for plotting speed (max 50k points)
                if len(dlrs_pool) > 50000:
                    idx_sub = np.random.choice(len(dlrs_pool), 50000, replace=False)
                    dlrs_pool = dlrs_pool[idx_sub]
                    gt_pool = gt_pool[idx_sub]
                bland_altman_plot(axes_ba[idx], dlrs_pool, gt_pool,
                                  title=f'{glabel} (n={len(subjs)})',
                                  color=colors_grp[glabel],
                                  ylabel_units=f'({mod_label}, z-score)')
            idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        ba_path = os.path.join(figure_dir, f'bland_altman_{modality}.png')
        plt.savefig(ba_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {ba_path}")

    # ======================================================================
    # 3. Hemispheric comparison + two-way ANOVA (Group x Hemisphere)
    # ======================================================================
    print(f"\n{'-' * 70}")
    print('3. Hemispheric Comparison (Left vs Right) & Two-Way ANOVA')
    print(f"{'-' * 70}")

    hemi_data = {'HC': {'L_cvr': [], 'R_cvr': [], 'L_bat': [], 'R_bat': []},
                 'SCI': {'L_cvr': [], 'R_cvr': [], 'L_bat': [], 'R_bat': []}}

    for glabel, (gdir, subjs) in groups.items():
        for sid in subjs:
            dlrs_cvr = load_nifti(os.path.join(output_dir, gdir, sid, 'DLRS_CVR.nii'))
            dlrs_bat = load_nifti(os.path.join(output_dir, gdir, sid, 'DLRS_BAT.nii'))
            mask = get_brain_mask(data_dir, gdir, sid)
            if mask is None:
                mask = np.abs(dlrs_cvr) > 0.01

            left_mask = np.zeros_like(mask)
            left_mask[:midline_x, :, :] = mask[:midline_x, :, :]
            right_mask = np.zeros_like(mask)
            right_mask[midline_x:, :, :] = mask[midline_x:, :, :]

            hemi_data[glabel]['L_cvr'].append(np.nanmean(dlrs_cvr[left_mask > 0]))
            hemi_data[glabel]['R_cvr'].append(np.nanmean(dlrs_cvr[right_mask > 0]))
            hemi_data[glabel]['L_bat'].append(np.nanmean(dlrs_bat[left_mask > 0]))
            hemi_data[glabel]['R_bat'].append(np.nanmean(dlrs_bat[right_mask > 0]))

    # Print hemispheric means
    for glabel in ['HC', 'SCI']:
        hd = hemi_data[glabel]
        print(f"\n  {glabel}:")
        print(f"    CVR  Left={np.mean(hd['L_cvr']):.4f}+/-{np.std(hd['L_cvr']):.4f}  "
              f"Right={np.mean(hd['R_cvr']):.4f}+/-{np.std(hd['R_cvr']):.4f}")
        print(f"    BAT  Left={np.mean(hd['L_bat']):.4f}+/-{np.std(hd['L_bat']):.4f}  "
              f"Right={np.mean(hd['R_bat']):.4f}+/-{np.std(hd['R_bat']):.4f}")

    # Two-way ANOVA (Group x Hemisphere) for CVR and BAT
    print(f"\n  Two-way ANOVA (Group x Hemisphere):")
    for modality in ['cvr', 'bat']:
        # Collect factors
        values, factor_group, factor_hemi = [], [], []
        for glabel in ['HC', 'SCI']:
            for val in hemi_data[glabel][f'L_{modality}']:
                values.append(val)
                factor_group.append(glabel)
                factor_hemi.append('Left')
            for val in hemi_data[glabel][f'R_{modality}']:
                values.append(val)
                factor_group.append(glabel)
                factor_hemi.append('Right')

        values = np.array(values)
        # Manual two-way ANOVA (Type I, balanced or near-balanced)
        grand_mean = np.mean(values)
        n_total = len(values)

        # Group means
        hc_vals = values[np.array(factor_group) == 'HC']
        sci_vals = values[np.array(factor_group) == 'SCI']
        left_vals = values[np.array(factor_hemi) == 'Left']
        right_vals = values[np.array(factor_hemi) == 'Right']

        # Cell means
        cells = {}
        for g in ['HC', 'SCI']:
            for h in ['Left', 'Right']:
                cell_mask = (np.array(factor_group) == g) & (np.array(factor_hemi) == h)
                cells[(g, h)] = values[cell_mask]

        n_g = 2  # groups
        n_h = 2  # hemispheres
        n_per_cell = len(cells[('HC', 'Left')])  # subjects per cell

        # SS_group
        ss_group = sum(len(values[np.array(factor_group) == g]) *
                       (np.mean(values[np.array(factor_group) == g]) - grand_mean) ** 2
                       for g in ['HC', 'SCI'])
        # SS_hemi
        ss_hemi = sum(len(values[np.array(factor_hemi) == h]) *
                      (np.mean(values[np.array(factor_hemi) == h]) - grand_mean) ** 2
                      for h in ['Left', 'Right'])
        # SS_interaction
        ss_inter = 0
        for g in ['HC', 'SCI']:
            for h in ['Left', 'Right']:
                cell = cells[(g, h)]
                expected = (np.mean(values[np.array(factor_group) == g])
                            + np.mean(values[np.array(factor_hemi) == h])
                            - grand_mean)
                ss_inter += len(cell) * (np.mean(cell) - expected) ** 2
        # SS_error
        ss_error = sum(np.sum((cells[(g, h)] - np.mean(cells[(g, h)])) ** 2)
                       for g in ['HC', 'SCI'] for h in ['Left', 'Right'])

        df_group = n_g - 1
        df_hemi = n_h - 1
        df_inter = df_group * df_hemi
        df_error = n_total - n_g * n_h

        ms_group = ss_group / df_group if df_group > 0 else 0
        ms_hemi = ss_hemi / df_hemi if df_hemi > 0 else 0
        ms_inter = ss_inter / df_inter if df_inter > 0 else 0
        ms_error = ss_error / df_error if df_error > 0 else 1e-15

        f_group = ms_group / ms_error
        f_hemi = ms_hemi / ms_error
        f_inter = ms_inter / ms_error

        p_group = 1 - sp_stats.f.cdf(f_group, df_group, df_error)
        p_hemi = 1 - sp_stats.f.cdf(f_hemi, df_hemi, df_error)
        p_inter = 1 - sp_stats.f.cdf(f_inter, df_inter, df_error)

        sig = lambda p: '*' if p < 0.05 else ('**' if p < 0.01 else 'ns')
        print(f"\n    {modality.upper()}:")
        print(f"      Group effect:       F({df_group},{df_error})={f_group:.3f}, "
              f"p={p_group:.4f} {sig(p_group)}")
        print(f"      Hemisphere effect:  F({df_hemi},{df_error})={f_hemi:.3f}, "
              f"p={p_hemi:.4f} {sig(p_hemi)}")
        print(f"      Interaction:        F({df_inter},{df_error})={f_inter:.3f}, "
              f"p={p_inter:.4f} {sig(p_inter)}")

    # Figure: Hemispheric comparison
    fig_hemi, axes_h = plt.subplots(1, 2, figsize=(14, 6))
    fig_hemi.suptitle('Hemispheric Comparison: DLRS CVR and BAT (Left vs Right)',
                      fontsize=14, fontweight='bold')

    for mi, (modality, ax, ylabel) in enumerate(
            [('cvr', axes_h[0], 'Mean CVR'), ('bat', axes_h[1], 'Mean BAT (s)')]):
        positions = []
        box_data = []
        box_colors = []
        tick_labels = []
        for gi, glabel in enumerate(['HC', 'SCI']):
            for hi, hemi in enumerate(['Left', 'Right']):
                pos = gi * 3 + hi
                positions.append(pos)
                box_data.append(hemi_data[glabel][f'{"L" if hemi == "Left" else "R"}_{modality}'])
                box_colors.append('#4DBEEE' if glabel == 'HC' else '#FF6B6B')
                tick_labels.append(f'{glabel}\n{hemi}')

        bp = ax.boxplot(box_data, positions=positions, widths=0.55,
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='k', markersize=7))
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for i, vals in enumerate(box_data):
            x_j = np.random.normal(positions[i], 0.05, len(vals))
            ax.scatter(x_j, vals, color=box_colors[i], edgecolors='k',
                       s=50, zorder=5, alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{modality.upper()}: Left vs Right Hemisphere', fontsize=12,
                     fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    hemi_path = os.path.join(figure_dir, 'hemispheric_comparison.png')
    plt.savefig(hemi_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {hemi_path}")

    # ======================================================================
    # 4. Regional MCA-territory analysis
    # ======================================================================
    print(f"\n{'-' * 70}")
    print('4. MCA Territory - Regional CVR and BAT')
    print(f"{'-' * 70}")

    mca_mask_L = np.isin(atlas_data, MCA_LABELS_LEFT)
    mca_mask_R = np.isin(atlas_data, MCA_LABELS_RIGHT)
    mca_mask_all = np.isin(atlas_data, MCA_LABELS_ALL)

    print(f"  MCA voxels: Left={mca_mask_L.sum()}, Right={mca_mask_R.sum()}, "
          f"Total={mca_mask_all.sum()}")

    mca_results = []
    for glabel, (gdir, subjs) in groups.items():
        for sid in subjs:
            dlrs_cvr = load_nifti(os.path.join(output_dir, gdir, sid, 'DLRS_CVR.nii'))
            dlrs_bat = load_nifti(os.path.join(output_dir, gdir, sid, 'DLRS_BAT.nii'))
            mask = get_brain_mask(data_dir, gdir, sid)
            if mask is None:
                mask = np.abs(dlrs_cvr) > 0.01

            mca_brain_L = mca_mask_L & mask
            mca_brain_R = mca_mask_R & mask
            mca_brain = mca_mask_all & mask

            row = {
                'group': glabel, 'subject': sid,
                'mca_cvr_L': float(np.nanmean(dlrs_cvr[mca_brain_L])) if mca_brain_L.sum() > 0 else np.nan,
                'mca_cvr_R': float(np.nanmean(dlrs_cvr[mca_brain_R])) if mca_brain_R.sum() > 0 else np.nan,
                'mca_cvr':   float(np.nanmean(dlrs_cvr[mca_brain]))   if mca_brain.sum() > 0 else np.nan,
                'mca_bat_L': float(np.nanmean(dlrs_bat[mca_brain_L])) if mca_brain_L.sum() > 0 else np.nan,
                'mca_bat_R': float(np.nanmean(dlrs_bat[mca_brain_R])) if mca_brain_R.sum() > 0 else np.nan,
                'mca_bat':   float(np.nanmean(dlrs_bat[mca_brain]))   if mca_brain.sum() > 0 else np.nan,
            }
            mca_results.append(row)
            print(f"  {glabel:<6} {sid:<18} "
                  f"CVR(L={row['mca_cvr_L']:.4f} R={row['mca_cvr_R']:.4f}) "
                  f"BAT(L={row['mca_bat_L']:.4f} R={row['mca_bat_R']:.4f})")

    # Figure: MCA regional analysis
    fig_mca, axes_mca = plt.subplots(1, 2, figsize=(14, 6))
    fig_mca.suptitle('MCA Territory: Regional CVR and BAT by Group & Hemisphere',
                     fontsize=14, fontweight='bold')

    for mi, (mod, ax, ylabel) in enumerate(
            [('cvr', axes_mca[0], 'Mean CVR (MCA territory)'),
             ('bat', axes_mca[1], 'Mean BAT (MCA territory, s)')]):
        positions = []
        box_data = []
        box_colors = []
        tick_labels = []
        for gi, glabel in enumerate(['HC', 'SCI']):
            for hi, (hemi, hkey) in enumerate([('Left', 'L'), ('Right', 'R')]):
                pos = gi * 3 + hi
                positions.append(pos)
                vals = [r[f'mca_{mod}_{hkey}'] for r in mca_results
                        if r['group'] == glabel and np.isfinite(r[f'mca_{mod}_{hkey}'])]
                box_data.append(vals)
                box_colors.append('#4DBEEE' if glabel == 'HC' else '#FF6B6B')
                tick_labels.append(f'{glabel}\n{hemi} MCA')

        bp = ax.boxplot(box_data, positions=positions, widths=0.55,
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='k', markersize=7))
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for i, vals in enumerate(box_data):
            x_j = np.random.normal(positions[i], 0.05, len(vals))
            ax.scatter(x_j, vals, color=box_colors[i], edgecolors='k',
                       s=50, zorder=5, alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{mod.upper()} in MCA Territory', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    mca_path = os.path.join(figure_dir, 'mca_regional_analysis.png')
    plt.savefig(mca_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {mca_path}")

    # ======================================================================
    # 5. DLRS vs GLM - Per-subject overlay maps (axial mid-slice)
    # ======================================================================
    print(f"\n{'-' * 70}")
    print('5. DLRS vs GLM Visual Comparison Maps')
    print(f"{'-' * 70}")

    # One figure per group (HC -> panel 9a, SCI -> panel 9b).  Columns:
    #   BOLD structural reference | DLRS CVR | GLM CVR | DLRS BAT | GLM BAT.
    # All CVR/BAT maps are within-brain robust z-scores on a common scale.
    col_titles = ['BOLD (tSD)', 'DLRS CVR', 'GLM CVR', 'DLRS BAT', 'GLM BAT']

    for glabel, (gdir, subjs) in groups.items():
        if not subjs:
            continue
        lbl_color = '#0072BD' if glabel == 'HC' else '#D32F2F'
        n_subj = len(subjs)
        fig_comp, axes_comp = plt.subplots(n_subj, 5, figsize=(15, n_subj * 2.9))
        if n_subj == 1:
            axes_comp = axes_comp.reshape(1, -1)
        fig_comp.suptitle(
            f'DLRS vs GLM Ground Truth, {glabel} '
            f'(axial mid-slice; CVR/BAT as within-brain robust z-scores)',
            fontsize=14, fontweight='bold')

        im_cvr = im_bat = None
        for ri, sid in enumerate(subjs):
            dlrs_cvr = load_nifti(os.path.join(output_dir, gdir, sid, 'DLRS_CVR.nii'))
            dlrs_bat = load_nifti(os.path.join(output_dir, gdir, sid, 'DLRS_BAT.nii'))
            gt_cvr, gt_bat = load_gt(data_dir, gdir, sid)
            mask = get_brain_mask(data_dir, gdir, sid)
            if mask is None:
                mask = np.abs(dlrs_cvr) > 0.01

            z_mid = dlrs_cvr.shape[2] // 2
            m_sl = mask[:, :, z_mid]

            # Robust within-brain z-scores so DLRS and GLM share one scale.
            dlrs_cvr_z = zscore_in_mask(dlrs_cvr, mask)
            dlrs_bat_z = zscore_in_mask(dlrs_bat, mask)
            gt_cvr_z = zscore_in_mask(gt_cvr, mask) if gt_cvr is not None else None
            gt_bat_z = zscore_in_mask(gt_bat, mask) if gt_bat is not None else None

            # Column 0: BOLD temporal-SD structural reference (grayscale).
            ax = axes_comp[ri, 0]
            anat = bold_tsd_slice(data_dir, gdir, sid, z=z_mid)
            if anat is not None:
                inside = anat[m_sl]
                lo, hi = (np.percentile(inside, [2, 98]) if inside.size
                          else (anat.min(), anat.max()))
                anat_m = np.where(m_sl, np.clip(anat, lo, hi), np.nan)
                ax.imshow(anat_m.T, cmap='gray', origin='lower', vmin=lo, vmax=hi)
            ax.axis('off')
            if ri == 0:
                ax.set_title(col_titles[0], fontsize=11, fontweight='bold', pad=8)
            ax.text(-0.12, 0.5, f'[{glabel}] {sid}', transform=ax.transAxes,
                    rotation=90, va='center', ha='center', fontsize=8.5,
                    fontweight='bold', color=lbl_color)

            # Columns 1-4: DLRS/GLM CVR and BAT (z-scored, shared scale).
            panels = [
                (1, dlrs_cvr_z, CVR_CMAP, 'cvr'),
                (2, gt_cvr_z,   CVR_CMAP, 'cvr'),
                (3, dlrs_bat_z, BAT_CMAP, 'bat'),
                (4, gt_bat_z,   BAT_CMAP, 'bat'),
            ]
            for col, vol_z, cmap, kind in panels:
                ax = axes_comp[ri, col]
                if vol_z is not None:
                    sl = np.where(m_sl, vol_z[:, :, z_mid], np.nan)
                else:
                    sl = np.full_like(dlrs_cvr[:, :, z_mid], np.nan)
                im = ax.imshow(sl.T, cmap=cmap, origin='lower',
                               vmin=Z_VMIN, vmax=Z_VMAX, interpolation='bilinear')
                if kind == 'cvr':
                    im_cvr = im
                else:
                    im_bat = im
                if ri == 0:
                    ax.set_title(col_titles[col], fontsize=11, fontweight='bold', pad=8)
                ax.axis('off')

        # Two shared colour bars (CVR and BAT) on the common z-score scale.
        if im_cvr is not None:
            cax_c = fig_comp.add_axes([0.92, 0.55, 0.012, 0.32])
            style_cbar(fig_comp.colorbar(im_cvr, cax=cax_c, extend='both'),
                       Z_TICKS, 'CVR (within-brain z-score)')
        if im_bat is not None:
            cax_b = fig_comp.add_axes([0.92, 0.14, 0.012, 0.32])
            style_cbar(fig_comp.colorbar(im_bat, cax=cax_b, extend='both'),
                       Z_TICKS, 'BAT (within-brain z-score)')

        fig_comp.subplots_adjust(left=0.06, right=0.90, top=0.94, bottom=0.02,
                                 wspace=0.05, hspace=0.12)
        comp_path = os.path.join(figure_dir,
                                 f'dlrs_vs_glm_comparison_{glabel.lower()}.png')
        plt.savefig(comp_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {comp_path}")

    # ======================================================================
    # Full report CSV
    # ======================================================================
    report_path = os.path.join(figure_dir, 'model_eval_full_report.csv')
    with open(report_path, 'w', newline='') as fh:
        # Merge metric_rows and mca_results
        all_keys = list(metric_rows[0].keys()) if metric_rows else ['group', 'subject']
        mca_keys = ['mca_cvr_L', 'mca_cvr_R', 'mca_cvr', 'mca_bat_L', 'mca_bat_R', 'mca_bat']
        hemi_keys = ['hemi_cvr_L', 'hemi_cvr_R', 'hemi_bat_L', 'hemi_bat_R']
        all_fields = all_keys + mca_keys + hemi_keys

        writer = csv.DictWriter(fh, fieldnames=all_fields)
        writer.writeheader()

        for mr in metric_rows:
            row = dict(mr)
            # Find matching MCA row
            mca_match = next((m for m in mca_results
                              if m['group'] == mr['group'] and m['subject'] == mr['subject']),
                             {})
            for k in mca_keys:
                row[k] = mca_match.get(k, '')
            # Hemispheric data
            glabel = mr['group']
            subjs_list = groups[glabel][1]
            si = subjs_list.index(mr['subject']) if mr['subject'] in subjs_list else -1
            if si >= 0:
                row['hemi_cvr_L'] = hemi_data[glabel]['L_cvr'][si]
                row['hemi_cvr_R'] = hemi_data[glabel]['R_cvr'][si]
                row['hemi_bat_L'] = hemi_data[glabel]['L_bat'][si]
                row['hemi_bat_R'] = hemi_data[glabel]['R_bat'][si]
            writer.writerow(row)

    print(f"\nSaved: {report_path}")

    print(f"\n{'=' * 70}")
    print(f"All outputs saved to: {figure_dir}/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
