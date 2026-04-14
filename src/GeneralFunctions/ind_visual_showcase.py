"""
spectral_individual.py
======================
Saves clean per-descriptor images coloured on one or TWO meshes, adds a
plain Laplace eigenvector panel, and finishes with a single conclusionary
overview figure combining everything.

Layout philosophy
-----------------
    Single mesh : one row of view panels.
    Two meshes  : Mesh A panels on the LEFT half, Mesh B panels on the RIGHT
                  half, separated by a thin vertical divider.  The figure
                  grows wider, not taller.

    Each per-descriptor PNG is displayed as one row, stacked top-to-bottom.
    A conclusion box sits at the bottom.

Outputs (in --outdir, default = current directory):
    laplace.png
    hks.png
    sihks.png
    wks.png
    summary.png

Usage (single mesh):
    python spectral_individual.py path/to/mesh.obj

Usage (TWO meshes - demonstrates isometric invariance):
    python spectral_individual.py path/to/mesh1.obj path/to/mesh2.ply

Extra options:
    --k 150              number of eigenpairs (default 120)
    --outdir figures/    output directory
    --views front side   which view angles to show (default: front side)
    --descriptors hks wks   subset of descriptors (default: all)
"""
import argparse
import os
import textwrap

import numpy as np
import trimesh
import igl
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings("ignore")

# ── palette ───────────────────────────────────────────────────────────────────
BG      = "#0d0f14"
ACCENT  = "#c8f542"
ACCENT2 = "#42c8f5"   # second mesh accent colour
TEXT    = "#e8eaf0"
SUBTEXT = "#8a8fa8"
FOOTER  = "#3d4260"

# ── font sizes ────────────────────────────────────────────────────────────────
FS_TITLE    = 22
FS_FORMULA  = 13
FS_PANEL    = 11
FS_CBAR     = 9
FS_FOOTER   = 11
FS_ISOMETRY = 10
FS_SUM_TTL  = 14
FS_SUM_TXT  = 9

DESCRIPTORS = {
    "laplace": {
        "cmap":    "coolwarm",
        "label":   "Laplace-Beltrami Eigenvectors",
        "short":   "Laplace",
        "formula": "LB·φ = λ M φ    shown: φ₂  (left)  and  φ₃  (right)",
        "caption": (
            "The 2nd and 3rd eigenvectors of the Laplace-Beltrami operator\n"
            "encode the principal 'frequency modes' of the surface geometry.\n"
            "They provide a canonical, intrinsic coordinate system for the mesh."
        ),
    },
    "hks": {
        "cmap":    "YlOrRd",
        "label":   "Heat Kernel Signature  (HKS)",
        "short":   "HKS",
        "formula": "HKS(x,t) = Σ exp(−λₖ t) φₖ(x)²   [mean over t]",
        "caption": (
            "Colour encodes how quickly heat diffuses away from each vertex.\n"
            "Warm tones → slow diffusion (flat / interior regions).\n"
            "Cool tones → fast diffusion (sharp features, tips)."
        ),
    },
    "sihks": {
        "cmap":    "viridis",
        "label":   "Scale-Invariant HKS  (SI-HKS)",
        "short":   "SI-HKS",
        "formula": "SI-HKS = HKS(λ̂ₖ, t) / Σ HKS   where  λ̂ₖ = λₖ/λ₁",
        "caption": (
            "HKS normalised by λ₁ and row-sum, removing scale dependency.\n"
            "Comparable across meshes of different physical sizes.\n"
            "Highlights intrinsic curvature independently of scale."
        ),
    },
    "wks": {
        "cmap":    "plasma",
        "label":   "Wave Kernel Signature  (WKS)",
        "short":   "WKS",
        "formula": "WKS(x,e) = Σ wₖ(e) φₖ(x)²   wₖ ∝ exp(-(e-log λₖ)²/2σ²)",
        "caption": (
            "Colour encodes quantum particle localisation probability.\n"
            "Sharper frequency discrimination than HKS.\n"
            "Distinguishes geometrically similar but topologically different regions."
        ),
    },
}

VIEW_PRESETS = {
    "front": (20, -60),
    "side":  (20,  30),
    "back":  (20, 120),
    "top":   (85, -60),
    "iso":   (30, -45),
}

# ── spectral math ──────────────────────────────────────────────────────────────
def _laplace_eigenpairs(V, F, k):
    n = V.shape[0]
    L = igl.cotmatrix(V, F)
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
    k_eff = min(k, n - 2)
    try:
        evals, evecs = eigsh(
            A=-L, M=M, k=k_eff, sigma=0, which="LM",
            maxiter=10_000, tol=1e-6, return_eigenvectors=True,
        )
    except ArpackNoConvergence as err:
        evals, evecs = err.eigenvalues, err.eigenvectors
    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx]


def _hks(evals, evecs, n_times=50):
    evals = np.maximum(evals, 1e-12)
    t_min = 4 * np.log(10) / evals[-1]
    t_max = 4 * np.log(10) / evals[1]
    times = np.logspace(np.log10(t_min), np.log10(t_max), n_times)
    return ((evecs ** 2) @ np.exp(-np.outer(evals, times))).mean(1)


def _sihks(evals, evecs, n_times=50):
    evals = np.maximum(evals, 1e-12)
    lam1  = evals[1]
    ev_n  = evals / lam1
    t_min = 4 * np.log(10) / ev_n[-1]
    t_max = 4 * np.log(10) / ev_n[1]
    times = np.logspace(np.log10(t_min), np.log10(t_max), n_times)
    hks   = (evecs ** 2) @ np.exp(-np.outer(ev_n, times))
    return (hks / np.sum(hks, axis=1, keepdims=True)).mean(1)


def _wks(evals, evecs, n_energies=100, sigma_factor=6.0):
    evals    = np.maximum(evals, 1e-12)
    log_e    = np.log(evals)
    energies = np.linspace(log_e[1], log_e[-1], n_energies)
    sigma    = (energies[1] - energies[0]) * sigma_factor
    phi_sq   = evecs ** 2
    wks      = np.zeros(evecs.shape[0])
    for e in energies:
        w = np.exp(-(e - log_e) ** 2 / (2 * sigma ** 2))
        w /= w.sum()
        wks += phi_sq @ w
    return wks / n_energies


# ── rendering helper ───────────────────────────────────────────────────────────
def _normalise_verts(V):
    ctr = (V.max(0) + V.min(0)) / 2
    scl = (V.max(0) - V.min(0)).max()
    return (V - ctr) / (scl + 1e-12)


def _render_mesh(ax, V, F, scalar, cmap_name, elev, azim,
                 vmin=None, vmax=None):
    """Draw a depth-sorted coloured mesh on a 3-D axis.  Returns (norm, cmap).

    Uses percentile-based colour limits (2nd-98th) to prevent outlier vertices
    from collapsing the entire colormap to one hue.
    """
    Vn   = _normalise_verts(V)
    cmap = plt.get_cmap(cmap_name)

    # Percentile clipping so outliers don't flatten the colour range.
    if vmin is None:
        vmin = np.percentile(scalar, 2)
    if vmax is None:
        vmax = np.percentile(scalar, 98)
    if np.isclose(vmin, vmax):
        vmin, vmax = scalar.min(), scalar.max() + 1e-12

    norm  = Normalize(vmin=vmin, vmax=vmax)
    vcols = cmap(norm(scalar))
    fcols = vcols[F].mean(1)
    tris  = Vn[F]
    order = np.argsort(tris[:, :, 2].mean(1))

    poly = Poly3DCollection(
        tris[order],
        facecolors=fcols[order],
        edgecolors="none",
        linewidths=0,
        antialiased=True,
    )
    ax.add_collection3d(poly)
    lim = 0.68
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_facecolor(BG)
    ax.patch.set_alpha(0)
    return norm, cmap


# ── shared figure decorations ──────────────────────────────────────────────────
def _add_header(fig, meta, y_title=0.97, y_formula=0.91):
    fig.text(0.5, y_title, meta["label"],
             ha="center", va="top",
             color=ACCENT, fontsize=FS_TITLE, fontweight="bold",
             fontfamily="monospace")
    fig.text(0.5, y_formula, meta["formula"],
             ha="center", va="top",
             color=TEXT, fontsize=FS_FORMULA, fontfamily="monospace")


def _add_footer(fig, mesh_names, n_vs, n_fs, y=0.022):
    parts = []
    for name, nv, nf in zip(mesh_names, n_vs, n_fs):
        parts.append(f"{name}  ({nv:,}v · {nf:,}f)")
    fig.text(0.5, y, "   |   ".join(parts),
             ha="center", va="bottom",
             color=FOOTER, fontsize=FS_FOOTER, fontfamily="monospace")


def _add_colourbar(fig, norm, cmap_name, rect=(0.905, 0.22, 0.022, 0.64)):
    sm = ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes(rect)
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.ax.tick_params(labelcolor=SUBTEXT, labelsize=FS_CBAR, colors=SUBTEXT)
    cb.outline.set_edgecolor("#22263a")
    cb.set_label("descriptor value", color=SUBTEXT,
                 fontsize=FS_CBAR, fontfamily="monospace")


# ── per-descriptor figure ──────────────────────────────────────────────────────
# ───────────────────────────────
# Single mesh:  [ view1 | view2 | … ] 
#
# Two meshes:   [ Mesh A: view1 | view2 | … || Mesh B: view1 | view2 | … ]
#               A thin VERTICAL separator is drawn between the two mesh groups.
#
def save_descriptor_figure(
    meshes,           # list of (V, F, scalar) 1 or 2 entries
    meta,             # DESCRIPTORS entry
    mesh_names,       # list of str
    views,            # [(view_name, (elev, azim)), …]
    outdir,
    filename_override=None,
):
    n_mesh  = len(meshes)
    n_views = len(views)

    # Share percentile-clipped colour range across both meshes.
    all_scalars = np.concatenate([s for _, _, s in meshes])
    g_vmin = np.percentile(all_scalars, 2)
    g_vmax = np.percentile(all_scalars, 98)
    if np.isclose(g_vmin, g_vmax):
        g_vmin, g_vmax = all_scalars.min(), all_scalars.max() + 1e-12

    # ── figure geometry ────────────────────────────────────────────────────────
    panel_w_in = 4.5
    panel_h_in = 4.5
    cbar_w_in  = 1.2   # inches reserved on the right for the colourbar

    # Total panels = n_mesh groups × n_views each, all in ONE row.
    n_total_cols = n_mesh * n_views
    fig_w = panel_w_in * n_total_cols + cbar_w_in
    fig_h = panel_h_in + 1.8   # fixed height: one row + header/footer

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    header_frac = 0.16
    footer_frac = 0.08
    note_frac   = 0.06 if n_mesh > 1 else 0.0
    panel_area  = 1.0 - header_frac - footer_frac - note_frac

    cbar_frac  = cbar_w_in / fig_w
    plot_right = 1.0 - cbar_frac - 0.01
    left_pad   = 0.01
    # Width fraction of a single panel (all panels share one row)
    col_w      = (plot_right - left_pad) / n_total_cols
    row_bottom = footer_frac + note_frac
    row_h_frac = panel_area                  # single row takes all panel area

    _add_header(fig, meta, y_title=0.975, y_formula=0.920)

    norm = None
    for m_idx, (V, F, scalar) in enumerate(meshes):
        accent_color = ACCENT if (n_mesh == 1 or m_idx == 0) else ACCENT2
        mesh_label   = chr(65 + m_idx) if n_mesh > 1 else None

        for v_idx, (vname, (elev, azim)) in enumerate(views):
            # Global column index for this panel
            col_idx = m_idx * n_views + v_idx
            left    = left_pad + col_idx * col_w

            ax = fig.add_axes(
                [left + 0.005, row_bottom + 0.015,
                 col_w - 0.010, row_h_frac - 0.04],
                projection="3d",
            )
            norm, _ = _render_mesh(
                ax, V, F, scalar, meta["cmap"], elev, azim,
                vmin=g_vmin, vmax=g_vmax,
            )

            title_txt = (f"Mesh {mesh_label}  ·  {vname}"
                         if mesh_label else vname)
            ax.set_title(title_txt, color=accent_color,
                         fontsize=FS_PANEL, fontfamily="monospace", pad=3)

        # Vertical separator between mesh groups (drawn after all views of
        # mesh A are placed, before mesh B begins).
        if n_mesh > 1 and m_idx < n_mesh - 1:
            sep_x = left_pad + (m_idx + 1) * n_views * col_w
            fig.add_artist(
                matplotlib.lines.Line2D(
                    [sep_x, sep_x],
                    [row_bottom, row_bottom + row_h_frac],
                    color="#2a2f4a", linewidth=1.2,
                    transform=fig.transFigure, clip_on=False,
                )
            )

    if n_mesh > 1:
        note_y = footer_frac + note_frac * 0.55
        fig.text(
            0.5, note_y,
            "↑  identical colour pattern on both meshes confirms isometric invariance",
            ha="center", va="center",
            color=ACCENT, fontsize=FS_ISOMETRY, fontfamily="monospace",
            alpha=0.90,
        )

    # Colourbar centred vertically over the panel area
    cbar_left   = plot_right + 0.015
    cbar_bottom = footer_frac + note_frac + 0.03
    cbar_height = panel_area - 0.06
    _add_colourbar(fig, norm, meta["cmap"],
                   rect=(cbar_left, cbar_bottom, 0.025, cbar_height))

    n_vs = [V.shape[0] for V, _, _ in meshes]
    n_fs = [F.shape[0] for _, F, _ in meshes]
    _add_footer(fig, mesh_names, n_vs, n_fs, y=footer_frac * 0.4)

    fname = (filename_override
             or f"{meta['short'].lower().replace('-','').replace(' ','')}.png")
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    return out


def save_laplace_figure(
    meshes,       # list of (V, F, evecs)
    mesh_names,
    views,
    outdir,
):
    meta    = DESCRIPTORS["laplace"]
    n_mesh  = len(meshes)
    eig_ids = [1, 2]           # φ₂, φ₃
    n_views = len(views)

    # Columns per mesh group: one column per (eigenvector × view) combination.
    cols_per_mesh = len(eig_ids) * n_views
    n_total_cols  = n_mesh * cols_per_mesh

    panel_w_in = 3.8
    panel_h_in = 4.0
    cbar_w_in  = 1.2

    fig_w = panel_w_in * n_total_cols + cbar_w_in
    fig_h = panel_h_in + 1.8

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    header_frac = 0.16
    footer_frac = 0.08
    note_frac   = 0.06 if n_mesh > 1 else 0.0
    panel_area  = 1.0 - header_frac - footer_frac - note_frac

    cbar_frac  = cbar_w_in / fig_w
    plot_right = 1.0 - cbar_frac - 0.01
    left_pad   = 0.01
    col_w      = (plot_right - left_pad) / n_total_cols
    row_bottom = footer_frac + note_frac
    row_h_frac = panel_area

    _add_header(fig, meta, y_title=0.975, y_formula=0.920)

    norm_last = None
    for m_idx, (V, F, evecs) in enumerate(meshes):
        accent_color = ACCENT if (n_mesh == 1 or m_idx == 0) else ACCENT2
        mesh_label   = chr(65 + m_idx) if n_mesh > 1 else None

        local_col = 0
        for eid in eig_ids:
            scalar = evecs[:, eid]
            lim_v  = np.abs(scalar).max()   # symmetric colourbar for eigenvecs

            for vname, (elev, azim) in views:
                col_idx = m_idx * cols_per_mesh + local_col
                left    = left_pad + col_idx * col_w

                ax = fig.add_axes(
                    [left + 0.005, row_bottom + 0.015,
                     col_w - 0.010, row_h_frac - 0.04],
                    projection="3d",
                )
                norm_last, _ = _render_mesh(
                    ax, V, F, scalar, meta["cmap"], elev, azim,
                    vmin=-lim_v, vmax=lim_v,
                )

                label_parts = []
                if mesh_label:
                    label_parts.append(f"Mesh {mesh_label}")
                label_parts += [f"φ{eid + 1}", vname]
                ax.set_title(
                    "  ·  ".join(label_parts),
                    color=accent_color, fontsize=FS_PANEL,
                    fontfamily="monospace", pad=3,
                )
                local_col += 1

        # Vertical separator between mesh groups
        if n_mesh > 1 and m_idx < n_mesh - 1:
            sep_x = left_pad + (m_idx + 1) * cols_per_mesh * col_w
            fig.add_artist(
                matplotlib.lines.Line2D(
                    [sep_x, sep_x],
                    [row_bottom, row_bottom + row_h_frac],
                    color="#2a2f4a", linewidth=1.2,
                    transform=fig.transFigure, clip_on=False,
                )
            )

    if n_mesh > 1:
        note_y = footer_frac + note_frac * 0.55
        fig.text(
            0.5, note_y,
            "↑  same spectral decomposition on isometric shapes → "
            "intrinsic coordinate system is shape-invariant",
            ha="center", va="center",
            color=ACCENT, fontsize=FS_ISOMETRY, fontfamily="monospace",
            alpha=0.90,
        )

    cbar_left   = plot_right + 0.015
    cbar_bottom = footer_frac + note_frac + 0.03
    cbar_height = panel_area - 0.06
    _add_colourbar(fig, norm_last, meta["cmap"],
                   rect=(cbar_left, cbar_bottom, 0.025, cbar_height))

    n_vs = [V.shape[0] for V, _, _ in meshes]
    n_fs = [F.shape[0] for _, F, _ in meshes]
    _add_footer(fig, mesh_names, n_vs, n_fs, y=footer_frac * 0.4)

    out = os.path.join(outdir, "laplace.png")
    fig.savefig(out, dpi=200, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    return out


# ── conclusionary summary figure ───────────────────────────────────────────────
#
# Each per-descriptor PNG (which is now wide/landscape) is shown as one row,
# stacked top-to-bottom.  A conclusion box sits at the bottom.
# Target: roughly square overall figure.
#
def save_summary_figure(image_paths, mesh_names, n_mesh, outdir):
    """
    Compose all per-descriptor PNG panels into one tall summary figure.
    Each source image occupies one row (displayed at full width).
    """
    import matplotlib.image as mpimg

    n_rows = len(image_paths)
    imgs   = [mpimg.imread(p) for p in image_paths]

    # Derive figure width from the widest source image's aspect ratio,
    # then size each row's height to match that ratio.
    # Cap the total image area so the figure stays roughly square.
    fig_w     = 18.0 
    max_ar    = max(img.shape[1] / img.shape[0] for img in imgs)
    row_h_in  = fig_w / max_ar
    margin_in = 2.8    # inches for title + conclusion box

    # Don't let the rows alone exceed fig_w (square cap)
    max_img_h = fig_w - margin_in
    if row_h_in * n_rows > max_img_h:
        row_h_in = max_img_h / n_rows

    fig_h = row_h_in * n_rows + margin_in
    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    # ── title ─────────────────────────────────────────────────────────────────
    title = "Spectral Geometry Descriptors (Summary)"
    if n_mesh == 2:
        title += f"\n(Isometric Invariance:  {mesh_names[0]}   vs   {mesh_names[1]})"
    fig.text(0.5, 0.997, title,
             ha="center", va="top",
             color=ACCENT, fontsize=FS_SUM_TTL, fontweight="bold",
             fontfamily="monospace")

    # ── image rows ────────────────────────────────────────────────────────────
    top_pad    = 0.04
    bot_pad    = margin_in / fig_h
    usable_h   = 1.0 - top_pad - bot_pad
    row_h_frac = usable_h / n_rows

    for i, img in enumerate(imgs):
        bottom = 1.0 - top_pad - (i + 1) * row_h_frac
        ax = fig.add_axes([0.0, bottom, 1.0, row_h_frac])
        ax.imshow(img, aspect="auto", interpolation="bilinear")
        ax.set_axis_off()

    # ── conclusion box ────────────────────────────────────────────────────────
    # conclusion = textwrap.dedent("""\
    #     KEY CONCLUSIONS
    #     Isometric Invariance: The LB operator and all descriptors are intrinsic (geodesic-only).  Identical colour patterns on Mesh A & B confirm this.
    #     Laplace Eigenvectors: Low-frequency modes give a canonical, pose-invariant coordinate system; foundational for spectral mesh processing.
    #     HKS:  Multi-scale heat diffusion fingerprint.  Stable & easy to compute; scale-dependent.  Best for coarse shape matching.
    #     SI-HKS: HKS normalised by λ₁; removes global scale.  Retains multi-scale sensitivity for cross-shape comparison.
    #     WKS: Quantum particle localisation model.  Finer frequency resolution than HKS; better at discriminating locally similar regions.
    # """)

    # box_bottom = 0.006
    # box_height = bot_pad - 0.014
    # fig.patches.extend([
    #     matplotlib.patches.FancyBboxPatch(
    #         (0.02, box_bottom), 0.96, box_height,
    #         boxstyle="round,pad=0.005",
    #         linewidth=1, edgecolor="#22263a", facecolor="#12151e",
    #         transform=fig.transFigure, clip_on=False, zorder=3,
    #     )
    # ])
    # fig.text(
    #     0.04, box_bottom + box_height - 0.005,
    #     conclusion,
    #     ha="left", va="top",
    #     color=TEXT, fontsize=FS_SUM_TXT, fontfamily="monospace",
    #     linespacing=1.5, zorder=4,
    # )

    out = os.path.join(outdir, "summary.png")
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    return out


# ── helpers ────────────────────────────────────────────────────────────────────
def _load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    V    = np.array(mesh.vertices, dtype=np.float64)
    F    = np.array(mesh.faces,    dtype=np.int32)
    return V, F


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Save per-descriptor spectral geometry images. "
            "Pass TWO meshes to demonstrate isometric invariance."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "meshes", nargs="+",
        help="One or two mesh files (.obj / .ply / .stl / .off / …)",
    )
    parser.add_argument("--k", type=int, default=120,
                        help="Number of Laplace-Beltrami eigenpairs (default 120)")
    parser.add_argument("--outdir", default=".",
                        help="Directory to write images into (default: cwd)")
    parser.add_argument(
        "--views", nargs="+",
        choices=list(VIEW_PRESETS.keys()), default=["front", "side"],
        help="View angles per panel  (default: front side)",
    )
    parser.add_argument(
        "--descriptors", nargs="+",
        choices=["hks", "sihks", "wks"],
        default=["hks", "sihks", "wks"],
        help="Descriptors to render  (default: all three)",
    )
    args = parser.parse_args()

    if len(args.meshes) > 2:
        parser.error("At most two mesh files can be supplied.")

    os.makedirs(args.outdir, exist_ok=True)

    views      = [(name, VIEW_PRESETS[name]) for name in args.views]
    mesh_names = [os.path.basename(p) for p in args.meshes]
    n_mesh     = len(args.meshes)

    print("=" * 60)
    print(f"  Meshes : {mesh_names}")
    print(f"  k      : {args.k}")
    print(f"  Views  : {args.views}")
    print(f"  Outdir : {os.path.abspath(args.outdir)}")
    print("=" * 60)

    # ── load meshes ────────────────────────────────────────────────────────────
    all_V, all_F = [], []
    for path in args.meshes:
        print(f"\n[load]  {path}")
        V, F = _load_mesh(path)
        print(f"        {V.shape[0]:,} vertices   {F.shape[0]:,} faces")
        all_V.append(V); all_F.append(F)

    # ── eigenpairs ─────────────────────────────────────────────────────────────
    all_evals, all_evecs = [], []
    for i, (V, F) in enumerate(zip(all_V, all_F)):
        print(f"\n[eigen] Mesh {chr(65+i)} computing {args.k} eigenpairs …")
        evals, evecs = _laplace_eigenpairs(V, F, args.k)
        all_evals.append(evals); all_evecs.append(evecs)
        print(f"        λ₁ = {evals[1]:.6f}   λ_max = {evals[-1]:.4f}")

    saved_paths = []

    # ── Laplace eigenvector figure ─────────────────────────────────────────────
    print("\n[render]  Laplace eigenvectors …")
    laplace_meshes = [(all_V[i], all_F[i], all_evecs[i]) for i in range(n_mesh)]
    p = save_laplace_figure(laplace_meshes, mesh_names, views, args.outdir)
    saved_paths.append(p)
    print(f"          saved to {os.path.basename(p)}")

    # ── descriptor figures ─────────────────────────────────────────────────────
    SCALAR_FNS = {"hks": _hks, "sihks": _sihks, "wks": _wks}
    for key in args.descriptors:
        print(f"\n[render]  {DESCRIPTORS[key]['label']} …")
        mesh_tuples = []
        for i in range(n_mesh):
            scalar = SCALAR_FNS[key](all_evals[i], all_evecs[i])
            mesh_tuples.append((all_V[i], all_F[i], scalar))
        p = save_descriptor_figure(
            meshes=mesh_tuples,
            meta=DESCRIPTORS[key],
            mesh_names=mesh_names,
            views=views,
            outdir=args.outdir,
        )
        saved_paths.append(p)
        print(f"          saved  to  {os.path.basename(p)}")

    # ── conclusionary summary ──────────────────────────────────────────────────
    print("\n[summary]  Composing conclusionary figure …")
    p = save_summary_figure(saved_paths, mesh_names, n_mesh, args.outdir)
    saved_paths.append(p)
    print(f"           saved  to  {os.path.basename(p)}")

    print("\n" + "=" * 60)
    print("  All files saved:")
    for p in saved_paths:
        print(f"    {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()