#!/usr/bin/env python3
"""
Visualize and Compare HC vs SCI Group Results from DLRS_CVR_BAT Pipeline.

Creates:
  1. Per-subject CVR/BAT maps in 2-column layout (HC left, SCI right)
  2. Group mean CVR/BAT maps
  3. HC vs SCI statistical comparison (box plots with t-tests)
  4. group_summary_statistics.csv

Usage:
    conda activate cvr_bat
    python visualize_group_comparison.py
"""

import os
import csv
import glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

# CVR and BAT display settings
# These are the canonical display ranges for the DLRS outputs (arbitrary
# units).  The network's 5*tanh() output layer can emit values well outside
# them, so imshow() clips; the colour bars below therefore (a) always label
# the exact end points and (b) carry extend arrows to declare the clipping.
CVR_VMIN, CVR_VMAX = -0.8, 0.8
BAT_VMIN, BAT_VMAX = -1.2, 1.2
# Explicit tick locations so the end points -0.8/+0.8 and -1.2/+1.2 are
# always printed.  Matplotlib's automatic locator picks "nice" round values
# (e.g. -1.0 ... 1.0 for BAT) and silently drops the true limits.
CVR_TICKS = [-0.8, -0.4, 0.0, 0.4, 0.8]
BAT_TICKS = [-1.2, -0.6, 0.0, 0.6, 1.2]
CVR_CMAP = 'hot'
BAT_CMAP = 'jet'
HC_COLOR = '#4DBEEE'
SCI_COLOR = '#FF6B6B'


def style_cbar(cb, ticks, label=None, fontsize=10, label_as_title=False):
    """Force a colour bar to print its exact end points.

    `ticks` must include vmin and vmax; labels are formatted with %g so
    -0.8 stays "-0.8" rather than being rounded away by the auto-locator.

    `label_as_title` puts the label above the bar instead of rotated to its
    right.  Use it where two bars sit side by side: the default position
    lands in the neighbouring bar's axes and the label is painted over.
    """
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{t:g}' for t in ticks])
    cb.ax.tick_params(labelsize=max(fontsize - 2, 6))
    if label is not None:
        if label_as_title:
            cb.ax.set_title(label, fontsize=fontsize, pad=8)
        else:
            cb.set_label(label, fontsize=fontsize)
    return cb


def load_nifti(filepath):
    return nib.load(filepath).get_fdata()


def get_slice(data, axis=2, idx=None):
    if idx is None:
        idx = data.shape[axis] // 2
    if axis == 0:
        return data[idx, :, :]
    elif axis == 1:
        return data[:, idx, :]
    else:
        return data[:, :, idx]


def clean_mask_slice(mask_2d):
    """Keep only the largest connected region via flood-fill from centroid.
    This eliminates salt-and-pepper noise (isolated bright/dark voxels)
    that can appear in the 2-D mask slice due to atlas boundary mismatches
    or interpolation artefacts during resampling."""
    mask = mask_2d.astype(bool).copy()
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return mask
    cy, cx = int(ys.mean()), int(xs.mean())
    if not mask[cy, cx]:
        dists = (ys - cy) ** 2 + (xs - cx) ** 2
        cy, cx = ys[np.argmin(dists)], xs[np.argmin(dists)]
    visited = np.zeros_like(mask, dtype=bool)
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


def bold_tsd_slice(data_dir, group_name, subj_id, axis=2, idx=None, n_frames=120):
    """Return a 2-D BOLD temporal-standard-deviation slice as an anatomical
    reference.  The preprocessed resting-state BOLD series
    (``PreprocessedData_RS/drwarfunc_run_combined-*-001.img``) is already
    warped to the 2 mm MNI grid that the DLRS maps live on, so the tSD map
    shows tissue/vascular structure (gyral pattern, ventricles) without any
    resampling.  This replaces the earlier plain white brain-mask silhouette,
    which carried no anatomical information and did not visually correspond to
    the extent of the predicted CVR/BAT maps."""
    fs = sorted(glob.glob(os.path.join(
        data_dir, group_name, subj_id, 'PreprocessedData_RS',
        'drwarfunc_run_combined-*-001.img')))
    if not fs:
        return None
    sel = np.linspace(0, len(fs) - 1, min(n_frames, len(fs))).astype(int)
    stack = np.stack([np.squeeze(nib.load(fs[i]).get_fdata()) for i in sel],
                     axis=-1)
    tsd = stack.std(axis=-1)
    return get_slice(tsd, axis=axis, idx=idx)


def representative_subject(subjects, cvr_means, bat_means):
    """Pick the subject whose (mean CVR, mean BAT) is closest to the group
    mean - the medoid - so the displayed maps are typical of the group rather
    than an outlier."""
    if not subjects:
        return None
    c = np.asarray(cvr_means, float)
    b = np.asarray(bat_means, float)
    d = (c - c.mean()) ** 2 + (b - b.mean()) ** 2
    return subjects[int(np.argmin(d))]


def find_subjects(output_dir, group_name):
    """Find all subjects in a group's output directory."""
    group_out = os.path.join(output_dir, group_name)
    if not os.path.isdir(group_out):
        return []
    subjects = []
    for s in sorted(os.listdir(group_out)):
        sp = os.path.join(group_out, s)
        if os.path.isdir(sp) and os.path.exists(os.path.join(sp, 'DLRS_CVR.nii')):
            subjects.append(s)
    return subjects


def get_brain_mask(data_dir, group_name, subj_id):
    """Load the atlas-based brain mask created during preprocessing.

    The mask was built by intersecting the Neuromorphometrics atlas (all
    labelled voxels = brain) with the BOLD field-of-view detected via
    temporal standard deviation > 1e-6.  Morphological closing (sphere
    r=2) and hole-filling were applied in 3-D to produce a smooth,
    anatomically shaped mask - replacing the earlier rectangular bounding
    box that resulted from a simple BOLD intensity threshold."""
    mask_file = os.path.join(data_dir, group_name, subj_id, 'mask', 'brainMask_RS.nii')
    if os.path.exists(mask_file):
        return load_nifti(mask_file) > 0.5
    return None


def compute_subject_stats(cvr_data, bat_data, brain_vox):
    """Return dict of per-subject whole-brain statistics."""
    cvr_brain = cvr_data[brain_vox]
    bat_brain = bat_data[brain_vox]
    return {
        'n_voxels': int(np.sum(brain_vox)),
        'cvr_mean': float(np.nanmean(cvr_brain)),
        'cvr_std':  float(np.nanstd(cvr_brain)),
        'cvr_median': float(np.nanmedian(cvr_brain)),
        'bat_mean': float(np.nanmean(bat_brain)),
        'bat_std':  float(np.nanstd(bat_brain)),
        'bat_median': float(np.nanmedian(bat_brain)),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    output_dir = os.path.join(data_dir, 'output')
    figure_dir = os.path.join(script_dir, 'figure')
    os.makedirs(figure_dir, exist_ok=True)

    # Find subjects in each group
    hc_subjects = find_subjects(output_dir, 'HC_subj')
    sci_subjects = find_subjects(output_dir, 'SCI_subj')

    print(f"HC subjects ({len(hc_subjects)}): {', '.join(hc_subjects)}")
    print(f"SCI subjects ({len(sci_subjects)}): {', '.join(sci_subjects)}")

    if not hc_subjects and not sci_subjects:
        print("No results found. Run the pipeline first.")
        return

    all_groups = []
    if hc_subjects:
        all_groups.append(('HC_subj', hc_subjects, '#4DBEEE'))
    if sci_subjects:
        all_groups.append(('SCI_subj', sci_subjects, '#FF6B6B'))

    # ------------------------------------------------------------------
    # Collect per-subject statistics (used by figures, table, and CSV)
    # ------------------------------------------------------------------
    stats_rows = []          # list of dicts for CSV
    group_cvr_means = {}     # group_label -> [per-subj mean CVR]
    group_bat_means = {}

    for group_name, subjects, color in all_groups:
        glabel = group_name.replace('_subj', '')
        cvr_subj_means = []
        bat_subj_means = []

        for subj_id in subjects:
            cvr_data = load_nifti(os.path.join(output_dir, group_name, subj_id, 'DLRS_CVR.nii'))
            bat_data = load_nifti(os.path.join(output_dir, group_name, subj_id, 'DLRS_BAT.nii'))

            mask_3d = get_brain_mask(data_dir, group_name, subj_id)
            brain_vox = mask_3d if mask_3d is not None else (np.abs(cvr_data) > 0.01)

            st = compute_subject_stats(cvr_data, bat_data, brain_vox)
            st['group'] = glabel
            st['subject'] = subj_id
            stats_rows.append(st)

            cvr_subj_means.append(st['cvr_mean'])
            bat_subj_means.append(st['bat_mean'])

        group_cvr_means[glabel] = cvr_subj_means
        group_bat_means[glabel] = bat_subj_means

    # ========================================================================
    # FIGURE 1 (MAIN): Representative maps + group distributions
    #   Style follows Hou et al. (npj Digit. Med. 2023) Fig. 2:
    #     - rows  = one representative subject per group (medoid)
    #     - cols  = BOLD structural reference | DLRS CVR | DLRS BAT
    #     - a SINGLE shared colour bar for CVR and one for BAT
    #     - box plots of whole-brain CVR/BAT (HC vs SCI) beneath the maps
    #   No plain "mask" column and no per-panel colour bars.
    # ========================================================================
    reps = []  # (group_dir, glabel, color, subj_id)
    for group_name, subjects, color in all_groups:
        glabel = group_name.replace('_subj', '')
        rep = representative_subject(subjects, group_cvr_means[glabel],
                                     group_bat_means[glabel])
        if rep is not None:
            reps.append((group_name, glabel, color, rep))

    if reps:
        ref_img = nib.load(os.path.join(output_dir, reps[0][0], reps[0][3],
                                        'DLRS_CVR.nii'))
        z_idx = ref_img.shape[2] // 2
        n_rows = len(reps)

        fig1 = plt.figure(figsize=(11, 4.7 * n_rows + 2.6))
        outer = gridspec.GridSpec(
            2, 1, height_ratios=[2.15 * n_rows / 2 + 0.0, 1.0],
            hspace=0.26, left=0.07, right=0.90, top=0.92, bottom=0.07)

        # --- Top: representative maps ---
        gs_top = gridspec.GridSpecFromSubplotSpec(
            n_rows, 3, subplot_spec=outer[0], wspace=0.06, hspace=0.12)
        col_titles = ['BOLD (temporal SD)', 'DLRS CVR', 'DLRS BAT']
        im_cvr = im_bat = None

        for r, (group_name, glabel, color, subj_id) in enumerate(reps):
            cvr_data = load_nifti(os.path.join(output_dir, group_name, subj_id, 'DLRS_CVR.nii'))
            bat_data = load_nifti(os.path.join(output_dir, group_name, subj_id, 'DLRS_BAT.nii'))
            cvr_slice = get_slice(cvr_data)
            bat_slice = get_slice(bat_data)

            mask_3d = get_brain_mask(data_dir, group_name, subj_id)
            if mask_3d is not None:
                brain_mask = clean_mask_slice(get_slice(mask_3d) > 0.5)
            else:
                brain_mask = np.abs(cvr_slice) > 0.01

            # anatomical reference (BOLD temporal SD)
            ax = fig1.add_subplot(gs_top[r, 0])
            anat = bold_tsd_slice(data_dir, group_name, subj_id, idx=z_idx)
            if anat is not None:
                inside = anat[brain_mask]
                lo, hi = (np.percentile(inside, [2, 98]) if inside.size
                          else (anat.min(), anat.max()))
                anat_m = np.where(brain_mask, np.clip(anat, lo, hi), np.nan)
                ax.imshow(anat_m.T, cmap='gray', origin='lower', vmin=lo, vmax=hi)
            ax.axis('off')
            if r == 0:
                ax.set_title(col_titles[0], fontsize=11, fontweight='bold', pad=8)
            ax.text(-0.09, 0.5, glabel, transform=ax.transAxes, rotation=90,
                    va='center', ha='center', fontsize=13, fontweight='bold',
                    color=color)
            ax.text(0.03, 0.96, chr(ord('a') + r), transform=ax.transAxes,
                    fontsize=13, fontweight='bold', va='top', ha='left',
                    color='white')

            # DLRS CVR
            ax = fig1.add_subplot(gs_top[r, 1])
            cvr_masked = np.where(brain_mask, cvr_slice, np.nan)
            im_cvr = ax.imshow(cvr_masked.T, cmap=CVR_CMAP, origin='lower',
                               vmin=CVR_VMIN, vmax=CVR_VMAX,
                               interpolation='bilinear')
            ax.axis('off')
            if r == 0:
                ax.set_title(col_titles[1], fontsize=11, fontweight='bold', pad=8)

            # DLRS BAT
            ax = fig1.add_subplot(gs_top[r, 2])
            bat_masked = np.where(brain_mask, bat_slice, np.nan)
            im_bat = ax.imshow(bat_masked.T, cmap=BAT_CMAP, origin='lower',
                               vmin=BAT_VMIN, vmax=BAT_VMAX,
                               interpolation='bilinear')
            ax.axis('off')
            if r == 0:
                ax.set_title(col_titles[2], fontsize=11, fontweight='bold', pad=8)

        # shared colour bars (author style, far right)
        cax_cvr = fig1.add_axes([0.915, 0.60, 0.016, 0.24])
        cb1 = fig1.colorbar(im_cvr, cax=cax_cvr, extend='both')
        style_cbar(cb1, CVR_TICKS, 'CVR (a.u.)')
        cax_bat = fig1.add_axes([0.915, 0.30, 0.016, 0.24])
        cb2 = fig1.colorbar(im_bat, cax=cax_bat, extend='both')
        style_cbar(cb2, BAT_TICKS, 'BAT (a.u.)')

        # --- Bottom: box plots (HC vs SCI) ---
        gs_bot = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[1], wspace=0.28)
        box_labels = list(group_cvr_means.keys())
        box_colors = [HC_COLOR if l == 'HC' else SCI_COLOR for l in box_labels]
        panels = [('CVR', group_cvr_means, 'Mean CVR (a.u.)', 'c'),
                  ('BAT', group_bat_means, 'Mean BAT (a.u.)', 'd')]
        for j, (metric, gdict, ylab, panel) in enumerate(panels):
            ax = fig1.add_subplot(gs_bot[0, j])
            data = [gdict[l] for l in box_labels]
            bp = ax.boxplot(data, labels=box_labels, patch_artist=True,
                            widths=0.55, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='black',
                                           markeredgecolor='black', markersize=6),
                            medianprops=dict(color='black', lw=1.4))
            for patch, c in zip(bp['boxes'], box_colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.55)
            for i, l in enumerate(box_labels):
                x = np.random.normal(i + 1, 0.05, size=len(gdict[l]))
                ax.scatter(x, gdict[l], color=box_colors[i], edgecolors='black',
                           s=42, alpha=0.85, zorder=5)
            ax.set_ylabel(ylab, fontsize=10)
            ax.grid(axis='y', alpha=0.25)
            ax.text(-0.02, 1.04, panel, transform=ax.transAxes, fontsize=13,
                    fontweight='bold', va='bottom', ha='right')
            try:
                from scipy import stats as _sp
                if len(box_labels) == 2:
                    _, p = _sp.ttest_ind(gdict[box_labels[0]], gdict[box_labels[1]],
                                         equal_var=False)  # Welch (unequal variances)
                    sig = '*' if p < 0.05 else 'n.s.'
                    ax.set_title(f'$P$ = {p:.2f} ({sig})', fontsize=9.5)
            except Exception:
                pass

        fig1.suptitle('Representative DLRS CVR / BAT maps and whole-brain '
                      'distributions (HC vs SCI)',
                      fontsize=13.5, fontweight='bold')
        fig1_path = os.path.join(figure_dir, 'representative_maps.png')
        plt.savefig(fig1_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {fig1_path}")

    # ========================================================================
    # FIGURE 1b (SUPPLEMENTARY): All subjects - clean layout
    #   HC (left) | SCI (right); each subject shows BOLD | CVR | BAT, matching
    #   the column layout of the representative Figure 3.
    #   Mask column removed; a single shared CVR / BAT colour bar is used so
    #   every panel is on an identical scale (fixes the earlier per-panel
    #   colour bars and the non-informative white-mask column).
    # ========================================================================
    n_hc = len(hc_subjects)
    n_sci = len(sci_subjects)
    max_rows = max(n_hc, n_sci, 1)

    fig1b = plt.figure(figsize=(17, max_rows * 3.0 + 1.2))
    gs = gridspec.GridSpec(max_rows, 6, figure=fig1b, wspace=0.04, hspace=0.18,
                           left=0.04, right=0.9, top=0.9, bottom=0.03)
    fig1b.suptitle('DLRS CVR and BAT: All Subjects\n'
                   'HC (left)  |  SCI (right);  each: BOLD | CVR | BAT',
                   fontsize=15, fontweight='bold')

    im_c_all = im_b_all = None
    sub_titles = ['BOLD (tSD)', 'CVR', 'BAT']

    def _plot_triple(ax_bold, ax_cvr, ax_bat, group_name, subj_id,
                     label_color, header):
        nonlocal im_c_all, im_b_all
        cvr_data = load_nifti(os.path.join(output_dir, group_name, subj_id, 'DLRS_CVR.nii'))
        bat_data = load_nifti(os.path.join(output_dir, group_name, subj_id, 'DLRS_BAT.nii'))
        cvr_slice = get_slice(cvr_data)
        bat_slice = get_slice(bat_data)
        mask_3d = get_brain_mask(data_dir, group_name, subj_id)
        if mask_3d is not None:
            brain_mask = clean_mask_slice(get_slice(mask_3d) > 0.5)
        else:
            brain_mask = np.abs(cvr_slice) > 0.01
        cvr_masked = np.where(brain_mask, cvr_slice, np.nan)
        bat_masked = np.where(brain_mask, bat_slice, np.nan)

        # BOLD temporal-SD anatomical reference (same as Figure 3)
        anat = bold_tsd_slice(data_dir, group_name, subj_id)
        if anat is not None:
            inside = anat[brain_mask]
            lo, hi = (np.percentile(inside, [2, 98]) if inside.size
                      else (anat.min(), anat.max()))
            anat_m = np.where(brain_mask, np.clip(anat, lo, hi), np.nan)
            ax_bold.imshow(anat_m.T, cmap='gray', origin='lower', vmin=lo, vmax=hi)
        ax_bold.axis('off')
        ax_bold.set_title(f'{subj_id}\n{sub_titles[0]}', fontsize=8.5,
                          fontweight='bold', color=label_color)

        im_c_all = ax_cvr.imshow(cvr_masked.T, cmap=CVR_CMAP, origin='lower',
                                 vmin=CVR_VMIN, vmax=CVR_VMAX, interpolation='bilinear')
        ax_cvr.axis('off')
        ax_cvr.set_title(f'{subj_id}\n{sub_titles[1]}', fontsize=8.5,
                         fontweight='bold', color=label_color)
        im_b_all = ax_bat.imshow(bat_masked.T, cmap=BAT_CMAP, origin='lower',
                                 vmin=BAT_VMIN, vmax=BAT_VMAX, interpolation='bilinear')
        ax_bat.axis('off')
        ax_bat.set_title(f'{subj_id}\n{sub_titles[2]}', fontsize=8.5,
                         fontweight='bold', color=label_color)

    for r in range(max_rows):
        if r < n_hc:
            _plot_triple(fig1b.add_subplot(gs[r, 0]), fig1b.add_subplot(gs[r, 1]),
                         fig1b.add_subplot(gs[r, 2]),
                         'HC_subj', hc_subjects[r], '#0072BD', r == 0)
        if r < n_sci:
            _plot_triple(fig1b.add_subplot(gs[r, 3]), fig1b.add_subplot(gs[r, 4]),
                         fig1b.add_subplot(gs[r, 5]),
                         'SCI_subj', sci_subjects[r], '#D32F2F', r == 0)

    fig1b.text(0.27, 0.925, 'HC (Healthy Controls)', ha='center', fontsize=13,
               fontweight='bold', color='#0072BD')
    fig1b.text(0.67, 0.925, 'SCI (Spinal Cord Injury)', ha='center', fontsize=13,
               fontweight='bold', color='#D32F2F')

    if im_c_all is not None:
        # The two bars sit side by side, so their labels go ABOVE them:
        # a rotated right-hand label on the CVR bar is drawn inside the BAT
        # bar's axes and ends up hidden behind it.
        cax_c = fig1b.add_axes([0.910, 0.55, 0.011, 0.30])
        style_cbar(fig1b.colorbar(im_c_all, cax=cax_c, extend='both'),
                   CVR_TICKS, 'CVR (a.u.)', label_as_title=True)
        cax_b = fig1b.add_axes([0.958, 0.55, 0.011, 0.30])
        style_cbar(fig1b.colorbar(im_b_all, cax=cax_b, extend='both'),
                   BAT_TICKS, 'BAT (a.u.)', label_as_title=True)

    fig1b_path = os.path.join(figure_dir, 'group_individual_maps.png')
    plt.savefig(fig1b_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig1b_path}")

    # ========================================================================
    # FIGURE 2: Group mean maps
    # ========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    fig2.suptitle('Group Mean CVR and BAT Maps', fontsize=16, fontweight='bold')

    for gi, (group_name, subjects, color) in enumerate(all_groups):
        if not subjects:
            continue
        cvr_stack = []
        bat_stack = []
        for subj_id in subjects:
            cvr_stack.append(load_nifti(os.path.join(output_dir, group_name, subj_id, 'DLRS_CVR.nii')))
            bat_stack.append(load_nifti(os.path.join(output_dir, group_name, subj_id, 'DLRS_BAT.nii')))

        cvr_mean = np.mean(cvr_stack, axis=0)
        bat_mean = np.mean(bat_stack, axis=0)
        cvr_slice = get_slice(cvr_mean)
        bat_slice = get_slice(bat_mean)

        mask_3d = get_brain_mask(data_dir, group_name, subjects[0])
        if mask_3d is not None:
            brain_mask = clean_mask_slice(get_slice(mask_3d) > 0.5)
        else:
            brain_mask = np.abs(cvr_slice) > 0.01

        cvr_masked = np.where(brain_mask, cvr_slice, np.nan)
        bat_masked = np.where(brain_mask, bat_slice, np.nan)
        glabel = group_name.replace('_subj', '')

        im_cvr = axes2[gi, 0].imshow(cvr_masked.T, cmap=CVR_CMAP, origin='lower',
                                      vmin=CVR_VMIN, vmax=CVR_VMAX, interpolation='bilinear')
        axes2[gi, 0].set_title(f'{glabel} Mean CVR (n={len(subjects)})', fontsize=12, fontweight='bold')
        axes2[gi, 0].axis('off')
        style_cbar(plt.colorbar(im_cvr, ax=axes2[gi, 0], fraction=0.046,
                                pad=0.04, extend='both'),
                   CVR_TICKS, 'CVR (a.u.)')

        im_bat = axes2[gi, 1].imshow(bat_masked.T, cmap=BAT_CMAP, origin='lower',
                                      vmin=BAT_VMIN, vmax=BAT_VMAX, interpolation='bilinear')
        axes2[gi, 1].set_title(f'{glabel} Mean BAT (n={len(subjects)})', fontsize=12, fontweight='bold')
        axes2[gi, 1].axis('off')
        style_cbar(plt.colorbar(im_bat, ax=axes2[gi, 1], fraction=0.046,
                                pad=0.04, extend='both'),
                   BAT_TICKS, 'BAT (a.u.)')

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig2_path = os.path.join(figure_dir, 'group_mean_maps.png')
    plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig2_path}")

    # ========================================================================
    # FIGURE 3: Box plot comparison
    # ========================================================================
    fig3, (ax_cvr, ax_bat) = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle('HC vs SCI: Whole-Brain CVR and BAT Distributions', fontsize=14, fontweight='bold')

    labels = list(group_cvr_means.keys())
    colors = ['#4DBEEE', '#FF6B6B']

    # CVR box plot
    cvr_data_list = [group_cvr_means[l] for l in labels]
    bp1 = ax_cvr.boxplot(cvr_data_list, labels=labels, patch_artist=True,
                          widths=0.5, showmeans=True,
                          meanprops=dict(marker='D', markerfacecolor='black', markersize=8))
    for patch, c in zip(bp1['boxes'], colors[:len(labels)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for i, (lbl, vals) in enumerate(group_cvr_means.items()):
        x = np.random.normal(i + 1, 0.04, size=len(vals))
        ax_cvr.scatter(x, vals, alpha=0.6, color=colors[i], edgecolors='black', zorder=5, s=50)
    ax_cvr.set_ylabel('Mean CVR (a.u.)', fontsize=12)
    ax_cvr.set_title('Whole-Brain Mean CVR per Subject', fontsize=12, fontweight='bold')
    ax_cvr.grid(axis='y', alpha=0.3)

    # BAT box plot
    bat_data_list = [group_bat_means[l] for l in labels]
    bp2 = ax_bat.boxplot(bat_data_list, labels=labels, patch_artist=True,
                          widths=0.5, showmeans=True,
                          meanprops=dict(marker='D', markerfacecolor='black', markersize=8))
    for patch, c in zip(bp2['boxes'], colors[:len(labels)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for i, (lbl, vals) in enumerate(group_bat_means.items()):
        x = np.random.normal(i + 1, 0.04, size=len(vals))
        ax_bat.scatter(x, vals, alpha=0.6, color=colors[i], edgecolors='black', zorder=5, s=50)
    ax_bat.set_ylabel('Mean BAT (a.u.)', fontsize=12)
    ax_bat.set_title('Whole-Brain Mean BAT per Subject', fontsize=12, fontweight='bold')
    ax_bat.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig3_path = os.path.join(figure_dir, 'group_comparison_boxplot.png')
    plt.savefig(fig3_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig3_path}")

    # ========================================================================
    # Console summary table
    # ========================================================================
    print(f"\n{'=' * 80}")
    print(f"GROUP SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    header = (f"{'Group':<8} {'Subject':<20} {'Voxels':>8} "
              f"{'CVR Mean':>9} {'CVR Std':>9} {'CVR Med':>9} "
              f"{'BAT Mean':>9} {'BAT Std':>9} {'BAT Med':>9}")
    print(header)
    print(f"{'-' * 80}")

    for group_name, subjects, color in all_groups:
        glabel = group_name.replace('_subj', '')
        g_rows = [r for r in stats_rows if r['group'] == glabel]
        for r in g_rows:
            print(f"{r['group']:<8} {r['subject']:<20} {r['n_voxels']:>8} "
                  f"{r['cvr_mean']:>9.4f} {r['cvr_std']:>9.4f} {r['cvr_median']:>9.4f} "
                  f"{r['bat_mean']:>9.4f} {r['bat_std']:>9.4f} {r['bat_median']:>9.4f}")
        grp_cvr = [r['cvr_mean'] for r in g_rows]
        grp_bat = [r['bat_mean'] for r in g_rows]
        print(f"{glabel:<8} {'GROUP MEAN':<20} {'':>8} "
              f"{np.mean(grp_cvr):>9.4f} {'':>9} {'':>9} "
              f"{np.mean(grp_bat):>9.4f}")
        print(f"{'-' * 80}")

    # t-test
    try:
        from scipy import stats as sp_stats
        hc_cvr_vals = group_cvr_means.get('HC', [])
        sci_cvr_vals = group_cvr_means.get('SCI', [])
        hc_bat_vals = group_bat_means.get('HC', [])
        sci_bat_vals = group_bat_means.get('SCI', [])
        if len(hc_cvr_vals) >= 2 and len(sci_cvr_vals) >= 2:
            t_cvr, p_cvr = sp_stats.ttest_ind(hc_cvr_vals, sci_cvr_vals, equal_var=False)
            t_bat, p_bat = sp_stats.ttest_ind(hc_bat_vals, sci_bat_vals, equal_var=False)
            print(f"\nIndependent Welch t-test (HC vs SCI):")
            print(f"  CVR: t={t_cvr:.3f}, p={p_cvr:.4f} {'*' if p_cvr < 0.05 else 'ns'}")
            print(f"  BAT: t={t_bat:.3f}, p={p_bat:.4f} {'*' if p_bat < 0.05 else 'ns'}")
    except ImportError:
        print("\n(Install scipy for t-test: pip install scipy)")

    # ========================================================================
    # Write CSV
    # ========================================================================
    csv_path = os.path.join(figure_dir, 'group_summary_statistics.csv')
    fieldnames = ['group', 'subject', 'n_voxels',
                  'cvr_mean', 'cvr_std', 'cvr_median',
                  'bat_mean', 'bat_std', 'bat_median']

    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in stats_rows:
            writer.writerow({k: r[k] for k in fieldnames})

        # Group summary rows
        for group_name, subjects, color in all_groups:
            glabel = group_name.replace('_subj', '')
            g_rows = [r for r in stats_rows if r['group'] == glabel]
            writer.writerow({
                'group': glabel,
                'subject': 'GROUP_MEAN',
                'n_voxels': '',
                'cvr_mean':   f"{np.mean([r['cvr_mean'] for r in g_rows]):.6f}",
                'cvr_std':    f"{np.std([r['cvr_mean'] for r in g_rows], ddof=1):.6f}",
                'cvr_median': '',
                'bat_mean':   f"{np.mean([r['bat_mean'] for r in g_rows]):.6f}",
                'bat_std':    f"{np.std([r['bat_mean'] for r in g_rows], ddof=1):.6f}",
                'bat_median': '',
            })

        # t-test row
        try:
            from scipy import stats as sp_stats
            hc_cvr_vals = group_cvr_means.get('HC', [])
            sci_cvr_vals = group_cvr_means.get('SCI', [])
            hc_bat_vals = group_bat_means.get('HC', [])
            sci_bat_vals = group_bat_means.get('SCI', [])
            if len(hc_cvr_vals) >= 2 and len(sci_cvr_vals) >= 2:
                t_cvr, p_cvr = sp_stats.ttest_ind(hc_cvr_vals, sci_cvr_vals, equal_var=False)
                t_bat, p_bat = sp_stats.ttest_ind(hc_bat_vals, sci_bat_vals, equal_var=False)
                writer.writerow({
                    'group': 'T-TEST',
                    'subject': 'HC_vs_SCI',
                    'n_voxels': '',
                    'cvr_mean':   f"t={t_cvr:.4f}",
                    'cvr_std':    f"p={p_cvr:.6f}",
                    'cvr_median': '*' if p_cvr < 0.05 else 'ns',
                    'bat_mean':   f"t={t_bat:.4f}",
                    'bat_std':    f"p={p_bat:.6f}",
                    'bat_median': '*' if p_bat < 0.05 else 'ns',
                })
        except ImportError:
            pass

    print(f"\nSaved: {csv_path}")
    print(f"\nAll outputs saved to: {figure_dir}/")


if __name__ == '__main__':
    main()
