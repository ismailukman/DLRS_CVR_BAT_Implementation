"""Main figures (no-covariate model, 41 HC vs 29 SCI).

    fig1_representative_maps   T1, mean BOLD, DLRS CVR and DLRS BAT for one HC and one SCI
                               participant (closest to their group's whole-brain mean)
    fig2_bat_networks          cortical networks with a DLRS BAT difference (p < 0.05) on the
                               inflated surface, and the nucleus accumbens
    fig3_regional_effects      SCI - HC differences in all 14 regions, and participant values
                               for the four regions with the smallest p

Inputs: results/regional_means.csv, results/group_comparison.csv.
Figure 1 also needs the fMRIPrep output (FMRIPREP_DIR) and the MATLAB input folders
(DLRS_INPUT_DIR); it is skipped when they are not available.

    python src/analysis/make_figures.py [1 2 3]
"""
import io
import os
import sys
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageChops

from common import (CORTEX, FIGURES, GROUPS, NEUROMORPH, RESULTS, SUBCORTEX, brain_mask, dlrs_map, grid,
                    nice, region_masks)
from cohort_exclusions import EXCLUDE

MODEL = "none"
FMRIPREP = os.environ.get("FMRIPREP_DIR", "")
DLRS_INPUT = os.environ.get("DLRS_INPUT_DIR", "")

HC_C, SCI_C = "#2F6DB5", "#D1495B"
SIG_C, NOM_C, NS_C = "#1B1B1B", "#8C8C8C", "#C8C8C8"
DIV = LinearSegmentedColormap.from_list("div", ["#2C5F8A", "#7FA8C9", "#F4F4F4", "#E69A8D", "#B2182B"])
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
})

T = pd.read_csv(os.path.join(RESULTS, "group_comparison.csv"))
T = T[(T.model == MODEL) & (T.family != "whole-brain")]
D = pd.read_csv(os.path.join(RESULTS, "regional_means.csv"))
D = D[~D.subject.isin(EXCLUDE)].reset_index(drop=True)
D["g"] = (D.group == "SCI").astype(int)
N_TXT = f"{(D.g == 0).sum()} HC vs {(D.g == 1).sum()} SCI; no covariates"


def qfmt(q):
    """q to 3 decimals, or 4 when it would round to 0.050."""
    return f"{q:.4f}" if 0.0495 <= q < 0.0505 else f"{q:.3f}"


def minus(s):
    return s.replace("-", "−")


def panel(ax, letter, x, y):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, f"{name}.png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", name)


def fig_maps():
    """Representative participants: closest to their group mean of whole-brain CVR and BAT (z-scored)."""
    if not (os.path.isdir(FMRIPREP) and os.path.isdir(DLRS_INPUT)):
        print("figure 1 skipped: set FMRIPREP_DIR and DLRS_INPUT_DIR")
        return
    atlas = nib.load(NEUROMORPH)
    shape, affine = grid()
    lab = np.rint(atlas.get_fdata()).astype(int)
    z = int(np.argmax(np.isin(lab, [1, 2, 21, 22]).sum(axis=(0, 1))))      # lateral-ventricle level
    zmm = int(affine[2, 2] * z + affine[2, 3])
    reps = {}
    for g in ("HC", "SCI"):
        d = D[D.group == g]
        x = np.column_stack([(d[c] - d[c].mean()) / d[c].std() for c in ("whole-brain|CVR", "whole-brain|BAT")])
        reps[g] = d.subject.values[int(np.argmin((x ** 2).sum(1)))]
    fig = plt.figure(figsize=(183 * MM, 92 * MM))
    gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.04, hspace=0.06,
                           left=0.05, right=0.93, top=0.9, bottom=0.03)
    for ri, g in enumerate(("HC", "SCI")):
        sid = reps[g]
        m = brain_mask(g, sid)
        t1f = sorted(glob.glob(os.path.join(FMRIPREP, f"sub-{sid}", "**", "anat",
                                            "*space-MNI152NLin6Asym_res-2_desc-preproc_T1w.nii.gz"), recursive=True))[0]
        t1 = np.squeeze(resample_from_to(nib.load(t1f), (shape, affine), order=1).get_fdata())
        bimg = nib.load(os.path.join(DLRS_INPUT, GROUPS[g], sid, "func_run_combined.nii"))
        sel = np.linspace(0, bimg.shape[3] - 1, min(60, bimg.shape[3])).astype(int)
        bm = np.mean([np.asanyarray(bimg.dataobj[..., i]) for i in sel], axis=0).astype(np.float32)
        bold = np.squeeze(resample_from_to(nib.Nifti1Image(bm, bimg.affine), (shape, affine), order=1).get_fdata())
        ins = m[:, :, z]

        def sl(v):
            a = v[:, :, z].astype(float).copy()
            a[~ins] = np.nan
            return np.rot90(a)

        panels = [("T1-weighted", sl(t1), "gray", np.percentile(t1[:, :, z][ins], [1, 99.5])),
                  ("BOLD (mean)", sl(bold), "gray", np.percentile(bold[:, :, z][ins], [1, 93])),
                  ("DLRS CVR", sl(dlrs_map(g, sid, "CVR")), "hot", (-0.8, 0.8)),
                  ("DLRS BAT", sl(dlrs_map(g, sid, "BAT")), "jet", (-1.2, 1.2))]
        for ci, (title, img, cmap, (lo, hi)) in enumerate(panels):
            ax = fig.add_subplot(gs[ri, ci])
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("black")
            im = ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="bilinear")
            ys, xs = np.where(np.isfinite(img))
            ax.set_xlim(xs.min() - 3, xs.max() + 3)
            ax.set_ylim(ys.max() + 3, ys.min() - 3)
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ri == 0:
                ax.set_title(title, fontsize=7.5, fontweight="bold", pad=3)
            if ci == 0:
                ax.text(-0.08, 0.5, g, transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=9, fontweight="bold", color=HC_C if g == "HC" else SCI_C)
            if ci == 2:
                im_cvr = im
            if ci == 3:
                im_bat = im
    for k, (im, label, ticks) in enumerate(((im_cvr, "CVR (a.u.)", [-0.8, -0.4, 0, 0.4, 0.8]),
                                            (im_bat, "BAT (a.u.)", [-1.2, -0.6, 0, 0.6, 1.2]))):
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[k, 4]), extend="both")
        cb.set_ticks(ticks)
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=5.5)
        cb.set_label(label, fontsize=6)
    fig.text(0.05, 0.975, minus(f"Representative participants, axial slice at z = {zmm} mm"),
             fontsize=8, fontweight="bold", va="top")
    save(fig, "fig1_representative_maps")


def network_labels_on_surface(fsa, masks):
    """Yeo-7 network of every fsaverage vertex (most frequent label along the cortical ribbon)."""
    from nilearn import surface
    shape, affine = grid()
    netvol = np.zeros(shape, np.int16)
    for k, net in enumerate(CORTEX, 1):
        netvol[masks[net]] = k
    img = nib.Nifti1Image(netvol, affine)
    return {h: np.rint(surface.vol_to_surf(img, fsa[f"pial_{h}"], inner_mesh=fsa[f"white_{h}"],
                                           interpolation="nearest_most_frequent")).astype(int)
            for h in ("left", "right")}


def fig_networks():
    from nilearn import datasets, plotting
    vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fsa = datasets.fetch_surf_fsaverage("fsaverage")
    masks = region_masks()
    lab = network_labels_on_surface(fsa, masks)
    t = T[(T.metric == "BAT") & (T.family == "cortex")].set_index("region")

    def render(h, view):
        tex = np.full(lab[h].shape, np.nan)
        for k, net in enumerate(CORTEX, 1):
            if t.loc[net].p < 0.05:
                tex[lab[h] == k] = t.loc[net].es
        f = plt.figure(figsize=(6, 6), dpi=300)
        ax = f.add_subplot(projection="3d")
        plotting.plot_surf_stat_map(fsa[f"infl_{h}"], tex, hemi=h, view=view, colorbar=False, cmap=DIV,
                                    vmin=-vmax, vmax=vmax, symmetric_cbar=True, threshold=None,
                                    bg_map=fsa[f"sulc_{h}"], bg_on_data=False, darkness=0.55, axes=ax, figure=f)
        buf = io.BytesIO()
        f.savefig(buf, format="png", facecolor="white")
        plt.close(f)
        im = Image.open(buf).convert("RGB")
        return np.asarray(im.crop(ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()))

    W, H = 183 * MM, 112 * MM
    fig = plt.figure(figsize=(W, H))
    fig.text(0.04, 0.975, "Regional differences in DLRS bolus arrival time (DLRS BAT) between SCI and HC",
             fontsize=10.4, fontweight="bold", va="top")
    fig.text(0.04, 0.915, "DLRS BAT. Red = higher in SCI than in HC; blue = lower", fontsize=8.6,
             fontweight="bold", va="top", color="#B2182B")
    sig = t[t.p < 0.05].sort_values("p")
    lines = [minus(f"{nice(r)}: {x.es:+.2f} SD difference, ") + (f"q = {qfmt(x.q)}" if x.q < 0.05 else f"p = {x.p:.3f}")
             for r, x in sig.iterrows()] or ["No network differed at p < 0.05"]
    fig.text(0.04, 0.868, "\n".join(lines), fontsize=7.8, va="top", color="#222222", linespacing=1.3)

    x0, x1, y_top, y_bot = 0.075, 0.665, 0.735, 0.125
    cw, rh = (x1 - x0) / 2, (y_top - y_bot) / 2
    for c, txt in enumerate(["Left hemisphere (L)", "Right hemisphere (R)"]):
        fig.text(x0 + cw * (c + 0.5), y_top + 0.012, txt, ha="center", va="bottom", fontsize=8.6, fontweight="bold")
    for r, txt in enumerate(["Lateral", "Medial"]):
        fig.text(x0 - 0.035, y_top - rh * (r + 0.5), txt, ha="center", va="center", rotation=90, fontsize=8.6,
                 fontweight="bold")
    layout = [("left", "lateral", 0, 0, "A", "P"), ("right", "lateral", 1, 0, "P", "A"),
              ("left", "medial", 0, 1, "P", "A"), ("right", "medial", 1, 1, "A", "P")]
    for h, view, c, r, left, right in layout:
        pad = 0.028
        ax = fig.add_axes([x0 + cw * c + pad, y_top - rh * (r + 1) + 0.005, cw - 2 * pad, rh - 0.01])
        ax.imshow(render(h, view))
        ax.axis("off")
        ax.text(-0.015, 0.5, left, transform=ax.transAxes, ha="right", va="center", fontsize=8.6,
                fontweight="bold", color="#555555")
        ax.text(1.015, 0.5, right, transform=ax.transAxes, ha="left", va="center", fontsize=8.6,
                fontweight="bold", color="#555555")

    # nucleus accumbens on the 1 mm MNI template
    tmpl = datasets.load_mni152_template(resolution=1)
    g1 = ((181, 217, 181), np.array([[-1., 0, 0, 90], [0, 1., 0, -126], [0, 0, 1., -72], [0, 0, 0, 1]]))
    t1 = np.squeeze(resample_from_to(tmpl, g1, order=3).get_fdata())
    acc_img = nib.Nifti1Image(masks["Accumbens"].astype(np.float32), grid()[1])
    acc = np.squeeze(resample_from_to(acc_img, g1, order=1).get_fdata()) > 0.5
    zi = int(np.round(np.argwhere(acc)[:, 2].mean()))
    zmm = int(g1[1][2, 2] * zi + g1[1][2, 3])
    row = T[(T.region == "Accumbens") & (T.metric == "BAT")].iloc[0]
    ax = fig.add_axes([0.70, 0.30, 0.20, 0.20 * W / H * 0.85])
    sl, m = np.rot90(t1[:, :, zi]), np.rot90(acc[:, :, zi])
    ax.imshow(sl, cmap="gray", vmin=np.percentile(sl[sl > 0], 2), vmax=np.percentile(sl, 99.5))
    ax.imshow(np.ma.masked_where(~m, np.full(m.shape, row.es)), cmap=DIV, norm=norm, interpolation="nearest")
    ax.contour(m, levels=[0.5], colors=["#111111" if row.q < 0.05 else "#8C8C8C"],
               linewidths=1.0 if row.q < 0.05 else 0.5)
    ys, xs = np.where(m)
    ax.set_xlim(xs.mean() - 40, xs.mean() + 40)
    ax.set_ylim(ys.mean() + 34, ys.mean() - 34)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(minus(f"Nucleus accumbens\nDLRS BAT, axial z = {zmm} mm"), fontsize=8.5, fontweight="bold", pad=3)
    ax.text(0.5, -0.06, f"{row.es:+.2f} SD, higher in SCI\np = {row.p:.3f}, FDR q = {qfmt(row.q)}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.1,
            fontweight="bold" if row.q < 0.05 else "normal")
    ax.text(0.04, 0.96, "R", transform=ax.transAxes, color="#111111", fontsize=8.5, fontweight="bold", va="top")
    ax.text(0.96, 0.96, "L", transform=ax.transAxes, color="#111111", fontsize=8.5, fontweight="bold", va="top",
            ha="right")

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=DIV), cax=fig.add_axes([0.935, 0.14, 0.013, 0.60]))
    cb.set_ticks([-1.0, -0.5, 0, 0.5, 1.0])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=7.8)
    cb.set_label("Difference, SCI − HC (SD units)", fontsize=8.5)
    axl = fig.add_axes([0.04, 0.0, 0.88, 0.10])
    axl.axis("off")
    axl.legend([(Patch(facecolor="#D96B6B"), Patch(facecolor="#5F8FB8")), Patch(facecolor="#BDBDBD")],
               ["SCI differs from HC (p < 0.05)", "no significant difference"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, ncol=3, frameon=False, fontsize=7.8,
               loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=2.2, columnspacing=1.6)
    axl.text(0.0, 0.18, f"{N_TXT}. q (FDR) is given where q < 0.05; otherwise p. "
             "A, anterior; P, posterior; L, left; R, right.", fontsize=7.3, color="#555555",
             transform=axl.transAxes, va="center")
    save(fig, "fig2_bat_networks")


def fig_effects():
    plt.rcParams.update({"xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.labelsize": 9,
                         "axes.titlesize": 9.5, "legend.fontsize": 8.5})
    fig = plt.figure(figsize=(183 * MM, 135 * MM))
    gs = gridspec.GridSpec(1, 2, wspace=0.38, left=0.16, right=0.9, top=0.96, bottom=0.47)
    order = CORTEX + SUBCORTEX
    for k, met in enumerate(("BAT", "CVR")):
        ax = fig.add_subplot(gs[k])
        t = T[T.metric == met].set_index("region").loc[order]
        y = np.arange(len(order))[::-1].astype(float)
        y[7:] -= 0.8
        for yi, (r, row) in zip(y, t.iterrows()):
            c, fc = (SIG_C, SIG_C) if row.q < 0.05 else (NOM_C, NOM_C) if row.p < 0.05 else (NS_C, "white")
            ax.plot([row.es_lo, row.es_hi], [yi, yi], color=c, lw=0.9)
            ax.scatter(row.es, yi, s=15, color=fc, edgecolor=c, lw=0.8, zorder=3)
            if row.p < 0.05:
                ax.annotate(f"q = {qfmt(row.q)}" if row.q < 0.05 else f"p = {row.p:.3f}", xy=(1.01, yi),
                            xycoords=("axes fraction", "data"), va="center", ha="left", fontsize=6.9, color=c,
                            fontweight="bold" if row.q < 0.05 else "normal", annotation_clip=False)
        ax.axvline(0, color="#777777", lw=0.5)
        ax.axhspan(y[6] - 0.65, y[0] + 0.6, color="#F5F7FA", zorder=0, lw=0)
        ax.set_xlim(-2, 2)
        ax.set_ylim(y[-1] - 0.8, y[0] + 0.8)
        ax.set_yticks(y)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticklabels([nice(r) for r in order] if k == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Difference, SCI − HC (SD units)")
        ax.set_title("DLRS BAT: positive = higher in SCI" if met == "BAT" else "DLRS CVR: negative = lower in SCI",
                     fontweight="bold")
        for yy, label in ((y[0] + 0.55, "Cortical networks"), (y[7] + 0.55, "Subcortical structures")):
            ax.text(0.99, yy, label, transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=7.3,
                    style="italic", color="#7A8794")
        if k == 0:
            panel(ax, "a", -0.62, 1.02)
    fig.legend(handles=[Line2D([], [], marker="o", color=SIG_C, mfc=SIG_C, lw=0.9, ms=4, label="FDR q < 0.05"),
                        Line2D([], [], marker="o", color=NOM_C, mfc=NOM_C, lw=0.9, ms=4, label="p < 0.05"),
                        Line2D([], [], marker="o", color=NS_C, mfc="white", lw=0.9, ms=4, label="not significant")],
               frameon=False, loc="upper center", bbox_to_anchor=(0.53, 0.405), ncol=3, handlelength=1.6)

    sub = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=gridspec.GridSpec(1, 1, left=0.08, right=0.99, top=0.32, bottom=0.04)[0], wspace=0.55)
    for i, (_, row) in enumerate(T.sort_values("p").head(4).iterrows()):
        ax = fig.add_subplot(sub[i])
        col = f"{row.region}|{row.metric}"
        for j, (g, c) in enumerate(((0, HC_C), (1, SCI_C))):
            v = D[D.g == g][col].values
            vp = ax.violinplot(v, positions=[j], widths=0.75, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(c)
                b.set_alpha(0.22)
                b.set_edgecolor("none")
            rng = np.random.default_rng(10 + j)
            ax.scatter(j + rng.uniform(-0.13, 0.13, len(v)), v, s=5, color=c, lw=0, zorder=2)
            q1, md, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([j - 0.22, j + 0.22], [md, md], color="black", lw=1.1, zorder=3)
            ax.plot([j, j], [q1, q3], color="black", lw=0.6, zorder=3)
        ymax, ymin = D[col].max(), D[col].min()
        span = ymax - ymin
        ax.plot([0, 0, 1, 1], [ymax + .06 * span, ymax + .1 * span, ymax + .1 * span, ymax + .06 * span],
                color="k", lw=0.5)
        stars = "**" if row.p < 0.01 else ("*" if row.p < 0.05 else "")
        ax.text(0.5, ymax + 0.12 * span, f"{stars} p = {row.p:.3f}" + (f", q = {qfmt(row.q)}" if row.q < 0.05 else ""),
                ha="center", va="bottom", fontsize=7.5, fontweight="bold" if row.q < 0.05 else "normal")
        ax.set_ylim(ymin - 0.05 * span, ymax + 0.33 * span)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["HC", "SCI"])
        for tl, c in zip(ax.get_xticklabels(), (HC_C, SCI_C)):
            tl.set_color(c)
            tl.set_fontweight("bold")
        ax.set_title(f"{nice(row.region)} DLRS {row.metric}", fontsize=8.2, fontweight="bold")
        ax.set_ylabel(f"DLRS {row.metric} (a.u.)", fontsize=7.5)
        if i == 0:
            panel(ax, "b", -0.42, 1.1)
    save(fig, "fig3_regional_effects")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    if "1" in which:
        fig_maps()
    if "2" in which:
        fig_networks()
    if "3" in which:
        fig_effects()
