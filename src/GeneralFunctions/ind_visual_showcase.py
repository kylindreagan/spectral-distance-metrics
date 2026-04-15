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
    sihks_norm.png (RGB from 3 time points)
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
import sys
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
        "formula": "SI-HKS = mean_{f} |FFT(HKS(t))| / DC   [freqs 1..16]",
        "caption": (
            "HKS spectrum normalised by its DC component, removing scale dependency.\n"
            "Each vertex is represented by the mean magnitude across 16 low frequencies.\n"
            "Invariant to global scaling; highlights intrinsic curvature."
        ),
    },
    
    "sihks_norm": {
        "cmap":    None,   # RGB visualisation – no colormap
        "label":   "Scale-Invariant HKS  (SI-HKS, eigenvalue normalised)",
        "short":   "SI-HKS-norm",
        "formula": "SI-HKS(t) = HKS(λ̂ₖ, t) / Σ HKS   where  λ̂ₖ = λₖ/λ₁\nRGB = first 3 time points",
        "caption": (
            "HKS normalised by λ₁ and row-sum, removing scale dependency.\n"
            "Each vertex's time series is mapped to RGB using the first 3 time points.\n"
            "Colour variation reveals intrinsic curvature changes over diffusion time."
        ),
        "rgb": True,   # special flag
    },
    "wks": {
        "cmap":    "plasma",
        "label":   "Wave Kernel Signature  (WKS)",
        "short":   "WKS",
        "formula": "WKS(x,e) = Σ wₖ(e) φₖ(x)²   wₖ ∝ exp(-(e-log λₖ)²/2σ²)\n[mean over energies]",
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your actual descriptor functions
from ShapeDNA.laplaceBeltramiShape import laplace_beltrami_eigenvalues
from HKS.heatKernelSignatures import compute_hks
from WKS.waveKernelSignatures import compute_wks
from HKS.scaleInvariantHKS import compute_scale_invariant_hks
from HKS.sihkseignorm import compute_sihks_norm


# ── helper to ensure 1D descriptor (for non‑RGB descriptors) ─────────────────
def _to_1d(scalar):
    """If scalar is 2D (n_vertices × n_features), reduce by mean along features."""
    if scalar is None:
        return None
    if isinstance(scalar, tuple):
        scalar = scalar[0]
    if hasattr(scalar, 'ndim') and scalar.ndim == 2:
        scalar = scalar.mean(axis=1)
    return scalar


# ── rendering helper (supports vertex colours directly) ───────────────────────
def _normalise_verts(V):
    ctr = (V.max(0) + V.min(0)) / 2
    scl = (V.max(0) - V.min(0)).max()
    return (V - ctr) / (scl + 1e-12)


def _render_mesh(ax, V, F, scalar=None, cmap_name=None, elev=20, azim=-60,
                 vmin=None, vmax=None, vertex_colors=None):
    """
    Draw a depth-sorted coloured mesh.
    If vertex_colors is provided (N×3 RGB), use those directly.
    Otherwise, use scalar + colormap.
    """
    Vn = _normalise_verts(V)
    tris = Vn[F]
    order = np.argsort(tris[:, :, 2].mean(1))

    if vertex_colors is not None:
        # Use per‑vertex RGB colours
        vcols = vertex_colors
        if vcols.max() <= 1.0:
            vcols = np.clip(vcols, 0, 1)
        else:
            vcols = vcols / 255.0
        fcols = vcols[F].mean(1)
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
        return None, None   # no norm/cmap for RGB

    # Otherwise, scalar + colormap
    if scalar is None:
        raise ValueError("Either vertex_colors or scalar must be provided")
    cmap = plt.get_cmap(cmap_name)
    if vmin is None:
        vmin = np.percentile(scalar, 2)
    if vmax is None:
        vmax = np.percentile(scalar, 98)
    if np.isclose(vmin, vmax):
        vmin, vmax = scalar.min(), scalar.max() + 1e-12

    norm = Normalize(vmin=vmin, vmax=vmax)
    vcols = cmap(norm(scalar))
    fcols = vcols[F].mean(1)
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
    if cmap_name is None:
        return
    sm = ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes(rect)
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.ax.tick_params(labelcolor=SUBTEXT, labelsize=FS_CBAR, colors=SUBTEXT)
    cb.outline.set_edgecolor("#22263a")
    cb.set_label("descriptor value", color=SUBTEXT,
                 fontsize=FS_CBAR, fontfamily="monospace")


# ── per-descriptor figure ──────────────────────────────────────────────────────
def save_descriptor_figure(
    meshes,           # list of (V, F, data) where data is either 1D scalar or 2D (n,3) RGB
    meta,             # DESCRIPTORS entry
    mesh_names,       # list of str
    views,            # [(view_name, (elev, azim)), …]
    outdir,
    filename_override=None,
):
    n_mesh  = len(meshes)
    n_views = len(views)
    is_rgb = meta.get("rgb", False)

    # For scalar descriptors, compute global colour limits across all meshes
    if not is_rgb:
        all_scalars = np.concatenate([s for _, _, s in meshes])
        g_vmin = np.percentile(all_scalars, 2)
        g_vmax = np.percentile(all_scalars, 98)
        if np.isclose(g_vmin, g_vmax):
            g_vmin, g_vmax = all_scalars.min(), all_scalars.max() + 1e-12
    else:
        g_vmin = g_vmax = None   # not used

    # ── figure geometry ────────────────────────────────────────────────────────
    panel_w_in = 4.5
    panel_h_in = 4.5
    cbar_w_in  = 1.2 if not is_rgb else 0.0   # no colourbar for RGB

    n_total_cols = n_mesh * n_views
    fig_w = panel_w_in * n_total_cols + cbar_w_in
    fig_h = panel_h_in + 1.8

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    header_frac = 0.16
    footer_frac = 0.08
    note_frac   = 0.06 if n_mesh > 1 else 0.0
    panel_area  = 1.0 - header_frac - footer_frac - note_frac

    plot_right = 1.0 - (cbar_w_in / fig_w) - 0.01 if not is_rgb else 0.99
    left_pad   = 0.01
    col_w      = (plot_right - left_pad) / n_total_cols
    row_bottom = footer_frac + note_frac
    row_h_frac = panel_area

    _add_header(fig, meta, y_title=0.975, y_formula=0.920)

    norm = None
    for m_idx, (V, F, data) in enumerate(meshes):
        accent_color = ACCENT if (n_mesh == 1 or m_idx == 0) else ACCENT2
        mesh_label   = chr(65 + m_idx) if n_mesh > 1 else None

        # For RGB descriptors, data is already vertex colours (N×3)
        # For scalar, data is 1D array
        for v_idx, (vname, (elev, azim)) in enumerate(views):
            col_idx = m_idx * n_views + v_idx
            left    = left_pad + col_idx * col_w

            ax = fig.add_axes(
                [left + 0.005, row_bottom + 0.015,
                 col_w - 0.010, row_h_frac - 0.04],
                projection="3d",
            )

            if is_rgb:
                # data is already RGB array (n_vertices, 3)
                _render_mesh(ax, V, F, vertex_colors=data,
                             elev=elev, azim=azim)
            else:
                norm, _ = _render_mesh(
                    ax, V, F, scalar=data, cmap_name=meta["cmap"],
                    elev=elev, azim=azim,
                    vmin=g_vmin, vmax=g_vmax,
                )

            title_txt = (f"Mesh {mesh_label}  ·  {vname}"
                         if mesh_label else vname)
            ax.set_title(title_txt, color=accent_color,
                         fontsize=FS_PANEL, fontfamily="monospace", pad=3)

        # Vertical separator between mesh groups
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

    if not is_rgb:
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

    plot_right = 1.0 - (cbar_w_in / fig_w) - 0.01
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
            lim_v  = np.abs(scalar).max()

            for vname, (elev, azim) in views:
                col_idx = m_idx * cols_per_mesh + local_col
                left    = left_pad + col_idx * col_w

                ax = fig.add_axes(
                    [left + 0.005, row_bottom + 0.015,
                     col_w - 0.010, row_h_frac - 0.04],
                    projection="3d",
                )
                norm_last, _ = _render_mesh(
                    ax, V, F, scalar=scalar, cmap_name=meta["cmap"],
                    elev=elev, azim=azim,
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
def save_summary_figure(image_paths, mesh_names, n_mesh, outdir):
    import matplotlib.image as mpimg

    n_rows = len(image_paths)
    imgs   = [mpimg.imread(p) for p in image_paths]

    fig_w     = 18.0 
    max_ar    = max(img.shape[1] / img.shape[0] for img in imgs)
    row_h_in  = fig_w / max_ar
    margin_in = 2.8

    max_img_h = fig_w - margin_in
    if row_h_in * n_rows > max_img_h:
        row_h_in = max_img_h / n_rows

    fig_h = row_h_in * n_rows + margin_in
    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    title = "Spectral Geometry Descriptors (Summary)"
    if n_mesh == 2:
        title += f"\n(Isometric Invariance:  {mesh_names[0]}   vs   {mesh_names[1]})"
    fig.text(0.5, 0.997, title,
             ha="center", va="top",
             color=ACCENT, fontsize=FS_SUM_TTL, fontweight="bold",
             fontfamily="monospace")

    top_pad    = 0.04
    bot_pad    = margin_in / fig_h
    usable_h   = 1.0 - top_pad - bot_pad
    row_h_frac = usable_h / n_rows

    for i, img in enumerate(imgs):
        bottom = 1.0 - top_pad - (i + 1) * row_h_frac
        ax = fig.add_axes([0.0, bottom, 1.0, row_h_frac])
        ax.imshow(img, aspect="auto", interpolation="bilinear")
        ax.set_axis_off()

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


# ── RGB conversion for sihks_norm ─────────────────────────────────────────────
def _sihks_norm_to_rgb(hks_si):
    """
    Convert a 2D descriptor (n_vertices × n_times) to RGB using the first 3 time points.
    Each channel is normalised independently to [0,1] across vertices.
    """
    # Take first 3 time points (or pad with zeros if fewer)
    n_times = hks_si.shape[1]
    if n_times < 3:
        # Pad with zeros
        pad = 3 - n_times
        hks_si = np.pad(hks_si, ((0,0), (0,pad)), constant_values=0)
    rgb = hks_si[:, :3].copy()
    # Normalise each channel to [0,1] (per‑channel min‑max)
    for c in range(3):
        ch = rgb[:, c]
        minc, maxc = ch.min(), ch.max()
        if maxc > minc:
            rgb[:, c] = (ch - minc) / (maxc - minc)
        else:
            rgb[:, c] = 0.5
    return rgb


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
        choices=["hks", "sihks", "sihks_norm", "wks"],
        default=["hks", "sihks", "sihks_norm", "wks"],
        help="Descriptors to render  (default: all four)",
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
        evals, evecs = laplace_beltrami_eigenvalues(
            (V, F), k=args.k, return_eigenvectors=True
        )
        all_evals.append(evals)
        all_evecs.append(evecs)
        print(f"        λ₁ = {evals[1]:.6f}   λ_max = {evals[-1]:.4f}")

    saved_paths = []

    # ── Laplace eigenvector figure ─────────────────────────────────────────────
    print("\n[render]  Laplace eigenvectors …")
    laplace_meshes = [(all_V[i], all_F[i], all_evecs[i]) for i in range(n_mesh)]
    p = save_laplace_figure(laplace_meshes, mesh_names, views, args.outdir)
    saved_paths.append(p)
    print(f"          saved to {os.path.basename(p)}")

    # ── descriptor figures ─────────────────────────────────────────────────────
    SCALAR_FNS = {
        "hks": compute_hks,
        "sihks": compute_scale_invariant_hks,
        "sihks_norm": compute_sihks_norm,
        "wks": compute_wks
    }
    for key in args.descriptors:
        print(f"\n[render]  {DESCRIPTORS[key]['label']} …")
        mesh_tuples = []
        for i in range(n_mesh):
            raw = SCALAR_FNS[key](all_evals[i], all_evecs[i])
            # Some functions return a tuple; extract the array
            if isinstance(raw, tuple):
                raw = raw[0]
            if key == "sihks_norm":
                # raw is (n_vertices, n_times) – convert to RGB using first 3 time points
                rgb = _sihks_norm_to_rgb(raw)
                mesh_tuples.append((all_V[i], all_F[i], rgb))
            else:
                # For other descriptors, reduce to 1D if needed
                data = _to_1d(raw)
                mesh_tuples.append((all_V[i], all_F[i], data))
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