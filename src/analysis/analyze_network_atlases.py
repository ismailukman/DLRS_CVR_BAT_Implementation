#!/usr/bin/env python3
"""
analyze_network_atlases.py - Exploratory network-level comparison of DLRS
CVR and BAT between Healthy Controls (HC) and Spinal Cord Injury (SCI),
using functional atlases (Schaefer-2018 200/7-network cortex + Tian Scale-I
subcortex) instead of the ad-hoc MCA labels.

Design notes / correctness safeguards
-------------------------------------
* The DLRS output NIfTI carries a non-standard affine, but its voxel grid
  matches the brain mask and Neuromorphometrics atlas (the existing pipeline
  aligns atlas<->data by array index).  We therefore resample every new
  atlas to the *brain-mask* grid (91x109x91, FSL-MNI152 2 mm) and apply it
  by array index, exactly as the current MCA analysis does.
* Schaefer is provided in FSL-MNI152 (1 mm) - same template family as the
  study data, so downsampling to 2 mm is well posed.
* Tian S1 is provided in MNI152NLin2009cAsym space (a *different* template
  generation, with flipped x-orientation).  resample_from_to() handles the
  geometry, but a residual few-mm template mismatch remains; subcortical
  results are therefore flagged as approximate.
* n = 5 per group: all group comparisons are exploratory.  We report Welch t,
  Cohen's d, and Benjamini-Hochberg FDR across the region set, and interpret
  nothing as confirmatory.

Outputs (figure/):
  network_regional_stats.csv     per-subject and group region means + tests
  network_regional_analysis.png  box plots (HC vs SCI) per network/nucleus
  atlas_alignment_check.png      atlas overlays on a subject mid-slice (QC)
"""

import os
import csv
import sys
import numpy as np
import nibabel as nib
import nibabel.processing as niproc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# This script lives in src/analysis, so the project root is two levels up.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(DATA, 'output')
FIG = os.path.join(ROOT, 'figure')
ATLAS = os.path.join(ROOT, 'preprocessing', 'atlas')

HC_COLOR, SCI_COLOR = '#4DBEEE', '#FF6B6B'
GROUPS = [('HC_subj', 'HC'), ('SCI_subj', 'SCI')]

SCHAEFER = os.path.join(ATLAS, 'schaefer_2018', 'schaefer_2018',
                        'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz')
SCHAEFER_TXT = os.path.join(ATLAS, 'schaefer_2018', 'schaefer_2018',
                            'Schaefer2018_200Parcels_7Networks_order.txt')
TIAN = os.path.join(ATLAS, 'tian_s1', 'Tian_Subcortex_S1_3T_2009cAsym.nii.gz')
TIAN_TXT = os.path.join(ATLAS, 'tian_s1', 'Tian_Subcortex_S1_3T_label.txt')

YEO7 = {'Vis': 'Visual', 'SomMot': 'Somatomotor', 'DorsAttn': 'DorsAttn',
        'SalVentAttn': 'VentAttn/Sal', 'Limbic': 'Limbic', 'Cont': 'Control/FP',
        'Default': 'Default'}
YEO7_ORDER = ['Visual', 'Somatomotor', 'DorsAttn', 'VentAttn/Sal',
              'Limbic', 'Control/FP', 'Default']
# Tian S1 bilateral grouping (combine L/R and aTHA+pTHA -> THA)
TIAN_GROUP = {'HIP': 'Hippocampus', 'AMY': 'Amygdala', 'pTHA': 'Thalamus',
              'aTHA': 'Thalamus', 'NAc': 'Accumbens', 'GP': 'Pallidum',
              'PUT': 'Putamen', 'CAU': 'Caudate'}
TIAN_ORDER = ['Thalamus', 'Caudate', 'Putamen', 'Pallidum', 'Accumbens',
              'Hippocampus', 'Amygdala']


def L(p):
    return np.squeeze(nib.load(p).get_fdata())


def first_mask_path():
    """Path to any available subject brain mask (all share the same MNI grid).

    Discovered dynamically so the script does not depend on a specific subject.
    """
    for gdir, _ in GROUPS:
        gpath = os.path.join(DATA, gdir)
        if not os.path.isdir(gpath):
            continue
        for sid in sorted(os.listdir(gpath)):
            mp = os.path.join(gpath, sid, 'mask', 'brainMask_RS.nii')
            if os.path.exists(mp):
                return mp, sid
    raise FileNotFoundError('No subject brainMask_RS.nii found under data/*_subj/*/mask/')


def ref_grid():
    """Target grid = the study's brain-mask / Neuromorphometrics grid."""
    mp, _ = first_mask_path()
    img = nib.load(mp)
    return img.shape, img.affine


def resample_labels(atlas_path, shape, affine):
    src = nib.load(atlas_path)
    res = niproc.resample_from_to(src, (shape, affine), order=0)  # nearest
    return np.rint(res.get_fdata()).astype(int)


def build_cortex_networkvol(shape, affine):
    """Resample Schaefer, then collapse 200 parcels -> 7 Yeo networks."""
    parc = resample_labels(SCHAEFER, shape, affine)
    parcel2net = {}
    with open(SCHAEFER_TXT) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            pid = int(parts[0])
            net = parts[1].split('_')[2]  # 7Networks_LH_<Net>_k
            parcel2net[pid] = YEO7.get(net, net)
    netvol = np.zeros(parc.shape, dtype=int)
    net_names = YEO7_ORDER
    for pid, name in parcel2net.items():
        if name in net_names:
            netvol[parc == pid] = net_names.index(name) + 1
    return netvol, net_names


def build_subcortex_vol(shape, affine):
    parc = resample_labels(TIAN, shape, affine)
    names = [l.strip() for l in open(TIAN_TXT) if l.strip()]  # 16, index = label-1
    vol = np.zeros(parc.shape, dtype=int)
    struct_names = TIAN_ORDER
    for i, lab_name in enumerate(names, start=1):
        base = lab_name.split('-')[0]              # e.g. 'pTHA-rh' -> 'pTHA'
        grp = TIAN_GROUP.get(base)
        if grp in struct_names:
            vol[parc == i] = struct_names.index(grp) + 1
    return vol, struct_names


def regional_means(cvr, bat, brain, labelvol, n_labels):
    """Mean CVR/BAT within each label intersected with the brain mask."""
    cvr_m, bat_m, nvox = [], [], []
    for k in range(1, n_labels + 1):
        reg = (labelvol == k) & brain
        n = int(reg.sum())
        nvox.append(n)
        if n >= 10:
            cvr_m.append(float(np.nanmean(cvr[reg])))
            bat_m.append(float(np.nanmean(bat[reg])))
        else:
            cvr_m.append(np.nan)
            bat_m.append(np.nan)
    return cvr_m, bat_m, nvox


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    m = len(idx)
    if m == 0:
        return q
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[order] = np.clip(adj, 0, 1)
    return q


def main():
    shape, affine = ref_grid()
    netvol, net_names = build_cortex_networkvol(shape, affine)
    subvol, sub_names = build_subcortex_vol(shape, affine)

    # QC: how many voxels each region has inside a representative brain mask
    qc_mask_path, qc_sid = first_mask_path()
    brain0 = L(qc_mask_path) > 0.5
    print(f'Alignment QC (voxels in-brain per region, subject {qc_sid}):')
    for k, nm in enumerate(net_names, 1):
        print(f'  [cortex] {nm:14s}: {int(((netvol==k)&brain0).sum()):6d}')
    for k, nm in enumerate(sub_names, 1):
        print(f'  [subcx]  {nm:14s}: {int(((subvol==k)&brain0).sum()):6d}')

    regions = [('cortex', net_names, netvol), ('subcortex', sub_names, subvol)]

    # Collect per-subject regional means
    rows = []
    data = {}  # (compartment, region, metric, group) -> list
    for gdir, glabel in GROUPS:
        gpath = os.path.join(OUT, gdir)
        subs = sorted([s for s in os.listdir(gpath)
                       if os.path.exists(os.path.join(gpath, s, 'DLRS_CVR.nii'))])
        for sid in subs:
            cvr = L(os.path.join(OUT, gdir, sid, 'DLRS_CVR.nii'))
            bat = L(os.path.join(OUT, gdir, sid, 'DLRS_BAT.nii'))
            mf = os.path.join(DATA, gdir, sid, 'mask', 'brainMask_RS.nii')
            brain = (L(mf) > 0.5) if os.path.exists(mf) else (np.abs(cvr) > 0.01)
            row = {'group': glabel, 'subject': sid}
            for comp, names, vol in regions:
                cvr_m, bat_m, nvox = regional_means(cvr, bat, brain, vol, len(names))
                for nm, cv, bv in zip(names, cvr_m, bat_m):
                    row[f'{comp}:{nm}:CVR'] = cv
                    row[f'{comp}:{nm}:BAT'] = bv
                    data.setdefault((comp, nm, 'CVR', glabel), []).append(cv)
                    data.setdefault((comp, nm, 'BAT', glabel), []).append(bv)
            rows.append(row)

    # Group comparison per region/metric
    stat_rows = []
    for comp, names, _ in regions:
        for metric in ['CVR', 'BAT']:
            pvals, tmp = [], []
            for nm in names:
                hc = [v for v in data[(comp, nm, metric, 'HC')] if np.isfinite(v)]
                sci = [v for v in data[(comp, nm, metric, 'SCI')] if np.isfinite(v)]
                if len(hc) >= 2 and len(sci) >= 2:
                    t, p = sp_stats.ttest_ind(hc, sci, equal_var=False)
                    d = cohen_d(hc, sci)
                else:
                    t, p, d = np.nan, np.nan, np.nan
                pvals.append(p)
                tmp.append(dict(compartment=comp, region=nm, metric=metric,
                                hc_mean=np.nanmean(hc) if hc else np.nan,
                                sci_mean=np.nanmean(sci) if sci else np.nan,
                                t=t, p=p, cohen_d=d))
            q = bh_fdr(pvals)
            for r, qq in zip(tmp, q):
                r['p_fdr'] = qq
                stat_rows.append(r)

    # Write CSV
    os.makedirs(FIG, exist_ok=True)
    csv_path = os.path.join(FIG, 'network_regional_stats.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['compartment', 'region', 'metric',
                                           'hc_mean', 'sci_mean', 't', 'p',
                                           'cohen_d', 'p_fdr'])
        w.writeheader()
        for r in stat_rows:
            w.writerow({k: (f'{r[k]:.4f}' if isinstance(r[k], float) and np.isfinite(r[k]) else r[k])
                        for k in w.fieldnames})
    print(f'\nSaved {csv_path}')

    # Print a compact stat summary
    print('\nGroup comparison (Welch t, Cohen d, BH-FDR q):')
    for r in stat_rows:
        if np.isfinite(r['p']):
            print(f"  {r['compartment']:9s} {r['region']:14s} {r['metric']} "
                  f"HC={r['hc_mean']:+.3f} SCI={r['sci_mean']:+.3f} "
                  f"t={r['t']:+.2f} d={r['cohen_d']:+.2f} p={r['p']:.3f} q={r['p_fdr']:.3f}")

    # ---- Figure: box plots per compartment/metric ----
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    panels = [('cortex', net_names, 'CVR', axes[0, 0]),
              ('cortex', net_names, 'BAT', axes[0, 1]),
              ('subcortex', sub_names, 'CVR', axes[1, 0]),
              ('subcortex', sub_names, 'BAT', axes[1, 1])]
    for comp, names, metric, ax in panels:
        pos = np.arange(len(names))
        for gi, glabel, color, off in [(0, 'HC', HC_COLOR, -0.18),
                                       (1, 'SCI', SCI_COLOR, 0.18)]:
            box_data = [[v for v in data[(comp, nm, metric, glabel)] if np.isfinite(v)]
                        for nm in names]
            bp = ax.boxplot(box_data, positions=pos + off, widths=0.32,
                            patch_artist=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='k',
                                           markeredgecolor='k', markersize=5),
                            medianprops=dict(color='k', lw=1.1))
            for b in bp['boxes']:
                b.set_facecolor(color); b.set_alpha(0.55)
            for j, vals in enumerate(box_data):
                xs = np.random.normal(pos[j] + off, 0.03, size=len(vals))
                ax.scatter(xs, vals, color=color, edgecolors='k', s=26,
                           alpha=0.8, zorder=5)
        ax.set_xticks(pos)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.axhline(0, color='gray', lw=0.7, ls=':')
        ax.set_ylabel(f'Mean {metric} (a.u.)', fontsize=11)
        ax.set_title(f'{comp.capitalize()}: {metric}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.25)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=HC_COLOR, alpha=0.55, label='HC'),
                        Patch(facecolor=SCI_COLOR, alpha=0.55, label='SCI')],
               loc='upper right', fontsize=11)
    fig.suptitle('Network-level DLRS CVR and BAT by group: '
                 'Schaefer-200/Yeo-7 cortex and Tian S1 subcortex (exploratory)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fpath = os.path.join(FIG, 'network_regional_analysis.png')
    fig.savefig(fpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved {fpath}')

    # ---- Atlas overview figure: cortex, subcortex, and combined ----
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    yeo_colors = ['#781286', '#4682B4', '#00760E', '#C43AFA',
                  '#C8F0A0', '#E69422', '#CD3E4E']            # Yeo-7 (cortex)
    sub_colors = ['#8C510A', '#D8B365', '#5AB4AC', '#01665E',
                  '#762A83', '#1B7837', '#E7298A']            # Tian structures

    # Functional affiliation of each Tian nucleus to a Yeo-7 network
    # (Tian et al., 2020): the cortical network whose resting-state signal the
    # nucleus most closely tracks.
    affil_map = {'PUT': 'Somatomotor', 'GP': 'Somatomotor', 'pTHA': 'Somatomotor',
                 'AMY': 'Limbic', 'aTHA': 'Limbic', 'NAc': 'Limbic',
                 'CAU': 'Control/FP', 'HIP': 'Default'}
    struct_affil = {'Thalamus': 'SMN/LN', 'Caudate': 'FPN', 'Putamen': 'SMN',
                    'Pallidum': 'SMN', 'Accumbens': 'LN', 'Hippocampus': 'DMN',
                    'Amygdala': 'LN'}
    tian_parc = resample_labels(TIAN, shape, affine)
    tian_names = [l.strip() for l in open(TIAN_TXT) if l.strip()]
    affil_vol = np.zeros(tian_parc.shape, int)  # subcortex -> Yeo network id
    for i, lab in enumerate(tian_names, start=1):
        net = affil_map.get(lab.split('-')[0])
        if net in net_names:
            affil_vol[tian_parc == i] = net_names.index(net) + 1

    yeo_cmap = ListedColormap(yeo_colors)
    yeo_norm = BoundaryNorm(np.arange(0.5, len(net_names) + 1.5), yeo_cmap.N)
    sub_cmap = ListedColormap(sub_colors)
    sub_norm = BoundaryNorm(np.arange(0.5, len(sub_names) + 1.5), sub_cmap.N)

    combined_func = netvol.copy()                # cortex by network ...
    combined_func[affil_vol > 0] = affil_vol[affil_vol > 0]   # ... subcortex by affiliation

    z_slices = [34, 42, 50, 58]
    rows = [('A', 'Cortex: Schaefer-200 parcels grouped into Yeo-7 networks',
             netvol, yeo_cmap, yeo_norm, 'yeo'),
            ('B', 'Subcortex: Tian Scale I structures',
             subvol, sub_cmap, sub_norm, 'sub'),
            ('C', 'Combined: cortex by network, subcortex by functional affiliation',
             combined_func, yeo_cmap, yeo_norm, 'yeo')]

    fig3 = plt.figure(figsize=(14.5, 10.2))
    gs3 = fig3.add_gridspec(3, len(z_slices) + 1,
                            width_ratios=[1] * len(z_slices) + [1.05],
                            wspace=0.04, hspace=0.30)
    for r, (tag, title, vol, cm, nm, legkind) in enumerate(rows):
        for i, zz in enumerate(z_slices):
            ax = fig3.add_subplot(gs3[r, i])
            ax.imshow(np.where(brain0[:, :, zz], 0.4, np.nan).T, cmap='Greys',
                      origin='lower', vmin=0, vmax=1)
            lab = vol[:, :, zz].astype(float)
            lab[lab == 0] = np.nan
            ax.imshow(lab.T, cmap=cm, norm=nm, origin='lower',
                      interpolation='nearest')
            if i == 0:
                ax.set_title(f'{tag}.  {title}', loc='left', fontsize=11,
                             fontweight='bold', pad=6)
            ax.text(0.5, -0.04, f'z = {zz}', transform=ax.transAxes, ha='center',
                    va='top', fontsize=8, color='0.35')
            ax.axis('off')
        axl = fig3.add_subplot(gs3[r, -1]); axl.axis('off')
        if legkind == 'sub':
            handles = [Patch(facecolor=sub_colors[k],
                             label=f'{sub_names[k]}  (~{struct_affil[sub_names[k]]})')
                       for k in range(len(sub_names))]
            axl.legend(handles=handles, title='Tian S1 structure (affiliation)',
                       loc='center left', fontsize=8.5, frameon=False,
                       title_fontsize=9.5)
        else:
            handles = [Patch(facecolor=yeo_colors[k], label=net_names[k])
                       for k in range(len(net_names))]
            ttl = 'Yeo-7 network' if r == 0 else 'Yeo-7 network (cortex + subcortex)'
            axl.legend(handles=handles, title=ttl, loc='center left',
                       fontsize=8.5, frameon=False, title_fontsize=9.5)

    fig3.suptitle('Functional atlas on the study grid: cortex, subcortex, and '
                  'combined (2 mm MNI)', fontsize=14, fontweight='bold')
    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    qpath = os.path.join(FIG, 'atlas_alignment_check.png')
    fig3.savefig(qpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved {qpath}')


if __name__ == '__main__':
    main()
