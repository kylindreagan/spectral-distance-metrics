"""
Shape Descriptor Benchmark Suite
Tests: ShapeDNA, HKS, SIHKS, WKS, AMKS
Outputs: cluster PNGs per descriptor + runtime table (terminal + PNG)
"""

import os, sys, time, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from collections import defaultdict

warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from batch_processor import load_meshes_cached
from ShapeDNA.laplaceBeltramiShape import laplace_beltrami_eigenvalues
from HKS.heatKernelSignatures import compute_hks
from WKS.waveKernelSignatures import compute_wks
from HKS.scaleInvariantHKS import compute_scale_invariant_hks
from HKS.sihkseignorm import compute_sihks_norm
from modern.averageMixingKernelSignature import compute_amks

# ── config ─────────────────────────────────────────────────────────────────────
DATA_DIR       = "././data/SHREC2011/"
MESH_CACHE     = "shrec_meshes_shrec11.pkl"
GT_FILE        = "././data/SHREC2011/test.cla"
K_EVALS        = 100
OUT_DIR        = "benchmark_outputs"
THUMB_SIZE     = 120      # px per thumbnail in scatter
TSNE_PERP      = 40
N_SAMPLE       = 60     # meshes to show in t-SNE (keep readable); set to None for all

os.makedirs(OUT_DIR, exist_ok=True)

# ── ground truth ───────────────────────────────────────────────────────────────
def load_cla_ground_truth(cla_file):
    gt = {}
    with open(cla_file) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 2
    current_class = None
    expect_count  = False
    count_remaining = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 3 and parts[1].lstrip('-').isdigit() and parts[2].isdigit():
            current_class   = parts[0]
            expect_count    = True
            count_remaining = 0
            i += 1
        elif expect_count and len(parts) == 1 and parts[0].isdigit():
            count_remaining = int(parts[0])
            expect_count    = False
            i += 1
        elif current_class and not expect_count and count_remaining > 0:
            for idx_str in parts:
                if idx_str.isdigit():
                    gt[int(idx_str)] = current_class
                    count_remaining -= 1
            i += 1
        else:
            i += 1
    return gt

def build_ground_truth(names, raw_gt):
    gt = {}
    for name in names:
        base = os.path.splitext(os.path.basename(name))[0]
        try:
            idx = int(base.replace('T','').replace('t',''))
            if idx in raw_gt:
                gt[name] = raw_gt[idx]
        except ValueError:
            pass
    return gt

# ── mesh thumbnail renderer ────────────────────────────────────────────────────
from matplotlib.colors import LightSource

def render_mesh_thumbnail(V, F, size=THUMB_SIZE):
    """Render mesh as a shell with surface normal shading no edges, no solid fill."""
    fig = plt.figure(figsize=(1, 1), dpi=size)
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Center and normalize the mesh
    center = V.mean(axis=0)
    V = V - center
    scale = np.abs(V).max()
    if scale > 0:
        V = V / scale

    # Sample faces for speed
    if len(F) > 1200:
        idx = np.random.choice(len(F), 1200, replace=False)
        F_draw = F[idx]
    else:
        F_draw = F

    # Compute per-face normals for shading
    v0 = V[F_draw[:, 0]]
    v1 = V[F_draw[:, 1]]
    v2 = V[F_draw[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-12)

    # Light from upper-left
    ls = LightSource(azdeg=225, altdeg=45)
    light_dir = np.array([-0.5, 0.5, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Diffuse intensity per face
    intensity = np.clip(normals @ light_dir, 0, 1)
    # Ambient + diffuse
    shading = 0.15 + 0.85 * intensity

    # Map shading to a cool blue-gray color
    base_color = np.array([0.22, 0.35, 0.55]) #deep navy-slate
    face_colors = shading[:, None] * base_color
    face_colors = np.clip(face_colors, 0, 1)
    # Add alpha channel
    face_colors_rgba = np.concatenate(
        [face_colors, np.ones((len(face_colors), 1))], axis=1
    )

    tris = [V[f] for f in F_draw]
    poly = Poly3DCollection(tris, alpha=1.0, linewidths=0,
                            edgecolors='none')
    poly.set_facecolor(face_colors_rgba)
    ax.add_collection3d(poly)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.view_init(elev=20, azim=30)
    fig.tight_layout(pad=0)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    rgba = np.concatenate([buf[:,:,1:], buf[:,:,:1]], axis=-1)
    plt.close(fig)
    return rgba

# ── t-SNE cluster plot ─────────────────────────────────────────────────────────
# ── t-SNE cluster plots ────────────────────────────────────────────────────────
def _compute_tsne_projection(descriptors, names, n_sample=N_SAMPLE):
    """Aggregate descriptors and run t-SNE. Returns (proj, sampled_names)."""
    def aggregate(d):
        if d.ndim == 1:
            return d
        return np.concatenate([d.mean(axis=0), d.std(axis=0), d.max(axis=0)])

    feats = np.array([aggregate(d) for d in descriptors])

    if n_sample and n_sample < len(names):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(names), n_sample, replace=False)
    else:
        idx = np.arange(len(names))

    feats_sub = feats[idx]
    names_sub = [names[i] for i in idx]
    feats_sub = feats_sub / (np.linalg.norm(feats_sub, axis=1, keepdims=True) + 1e-12)

    tsne = TSNE(n_components=2, perplexity=min(TSNE_PERP, len(names_sub) - 1),
                random_state=42, max_iter=1000)
    proj = tsne.fit_transform(feats_sub)
    return proj, names_sub


def plot_cluster(descriptors, names, gt, meshes, title, out_path,
                 n_sample=N_SAMPLE, thumb_size=THUMB_SIZE):
    """t-SNE scatter with 3-D mesh thumbnails coloured by class."""
    print(f"  Running t-SNE for {title}...")
    proj, names_sub = _compute_tsne_projection(descriptors, names, n_sample)

    classes     = sorted(set(gt.values()))
    cmap        = plt.cm.get_cmap('tab20', len(classes))
    class_color = {c: cmap(i) for i, c in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(20, 16))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_title(title, color='black', fontsize=20, fontweight='bold', pad=20)
    ax.tick_params(colors='gray', labelsize=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('lightgray')

    for j, name in enumerate(names_sub):
        cls   = gt.get(name, 'unknown')
        color = class_color.get(cls, (0.5, 0.5, 0.5, 1))
        x, y  = proj[j]

        circle = plt.Circle(
            (x, y),
            radius=0.012 * (np.ptp(proj[:, 0]) + np.ptp(proj[:, 1])) / 2,
            color=color, alpha=0.35, zorder=1,
        )
        ax.add_patch(circle)

        mesh = meshes.get(name)
        if mesh is not None:
            V, F = mesh['V'], mesh['F']
            try:
                thumb = render_mesh_thumbnail(V, F, size=thumb_size)
                im = OffsetImage(thumb, zoom=0.55)
                ab = AnnotationBbox(
                    im, (x, y), frameon=True,
                    bboxprops=dict(edgecolor=color, linewidth=1.5,
                                   facecolor='white', boxstyle='round,pad=0.1'),
                )
                ax.add_artist(ab)
            except Exception:
                ax.scatter(x, y, color=color, s=40, zorder=3)
        else:
            ax.scatter(x, y, color=color, s=40, zorder=3)

    handles = [mpatches.Patch(color=class_color[c], label=c) for c in classes]
    ax.legend(handles=handles, loc='upper left', fontsize=6,
              facecolor='#161b22', edgecolor='#333', labelcolor='white',
              ncol=3, framealpha=0.85)
    ax.autoscale()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_cluster_points(descriptors, names, gt, title, out_path,
                        n_sample=N_SAMPLE):
    """t-SNE scatter with plain colored point"""
    print(f"  Running t-SNE (points) for {title}...")
    proj, names_sub = _compute_tsne_projection(descriptors, names, n_sample)

    classes     = sorted(set(gt.values()))
    cmap        = plt.cm.get_cmap('tab20', len(classes))
    class_color = {c: cmap(i) for i, c in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f8f8')
    ax.set_title(title + 'points', color='black', fontsize=16,
                 fontweight='bold', pad=14)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Group by class so each class gets a single legend entry
    class_points = defaultdict(list)
    for j, name in enumerate(names_sub):
        cls = gt.get(name, 'unknown')
        class_points[cls].append(proj[j])

    for cls in classes:
        pts = np.array(class_points[cls])
        if len(pts) == 0:
            continue
        color = class_color[cls]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=[color],
            s=55,
            alpha=0.82,
            edgecolors='white',
            linewidths=0.5,
            label=cls,
            zorder=3,
        )

    ax.legend(loc='upper left', fontsize=7, ncol=3,
              facecolor='#161b22', edgecolor='#444', labelcolor='white',
              framealpha=0.88, markerscale=1.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")

# ── runtime table PNG ──────────────────────────────────────────────────────────
def save_runtime_table(rows, out_path):
    """
    rows: list of (descriptor, n_meshes, total_s, per_mesh_s, mAP, p10)
    """
    col_labels = ['Descriptor', 'Meshes', 'Total (s)', 'Per mesh (s)', 'mAP', 'P@10']
    cell_text  = [[r[0], str(r[1]), f'{r[2]:.1f}', f'{r[3]:.2f}',
                   f'{r[4]:.4f}', f'{r[5]:.4f}'] for r in rows]

    fig, ax = plt.subplots(figsize=(26, 20))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.axis('off')

    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1f6feb')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#161b22')
            cell.set_text_props(color='#e6edf3')
        else:
            cell.set_facecolor('#0d1117')
            cell.set_text_props(color='#e6edf3')
        cell.set_edgecolor('#30363d')

    ax.set_title('Shape Descriptor Benchmark: Runtime & Retrieval',
                 color='white', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved runtime table → {out_path}")

# ── retrieval metrics ──────────────────────────────────────────────────────────
def average_precision(query_name, results, gt):
    q_class = gt[query_name]
    n_rel   = sum(1 for n in gt if gt[n] == q_class and n != query_name)
    if n_rel == 0: return 0.0
    hits, ap = 0, 0.0
    for rank, (name, _) in enumerate(results, 1):
        if gt.get(name) == q_class:
            hits += 1
            ap   += hits / rank
    return ap / n_rel

def precision_at_k(query_name, results, gt, k=10):
    q_class = gt[query_name]
    return sum(1 for name, _ in results[:k] if gt.get(name) == q_class) / k

def evaluate(descriptors_dict, names, gt):
    valid = [n for n in names if n in gt and n in descriptors_dict]
    feats = []
    for n in valid:
        d = descriptors_dict[n]
        if d.ndim == 2:
            feats.append(np.concatenate([d.mean(axis=0), d.std(axis=0), d.max(axis=0)]))
        else:
            feats.append(d)
    feats = np.array(feats)
    D     = cdist(feats, feats, metric='euclidean')
    aps, p10s = [], []
    for qi, qname in enumerate(valid):
        ranked = [(valid[i], D[qi,i]) for i in np.argsort(D[qi]) if valid[i] != qname]
        aps.append(average_precision(qname, ranked, gt))
        p10s.append(precision_at_k(qname, ranked, gt))
    return float(np.mean(aps)), float(np.mean(p10s))


def zoom_on_cluster(names, gt, meshes, class_names=None, t_sne_coords=None,
                    bounding_box=None, n_max=30, out_path="zoom_cluster.png",
                    thumb_size=200, cols=5):

    # Create a grid of larger mesh thumbnails for a specific cluster.

    # ---- 1. Select meshes ----
    selected = []
    for name in names:
        if name not in gt or name not in meshes:
            continue
        if class_names is not None and gt[name] not in class_names:
            continue
        if bounding_box is not None and t_sne_coords is not None:
            x, y = t_sne_coords[names.index(name)]
            xmin, xmax, ymin, ymax = bounding_box
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                continue
        selected.append(name)

    if len(selected) > n_max:
        # Optionally, sample randomly or pick the first n_max
        rng = np.random.default_rng(42)
        selected = rng.choice(selected, n_max, replace=False).tolist()

    if not selected:
        print("No meshes matched the selection criteria.")
        return

    # ---- 2. Prepare class colors (same as in your cluster plot) ----
    classes = sorted(set(gt.values()))
    cmap = plt.cm.get_cmap('tab20', len(classes))
    class_color = {c: cmap(i) for i, c in enumerate(classes)}

    # ---- 3. Create grid layout ----
    n = len(selected)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5),
                             squeeze=False)
    fig.patch.set_facecolor('white')

    # ---- 4. Place each mesh thumbnail ----
    for idx, name in enumerate(selected):
        ax = axes[idx // cols, idx % cols]
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        cls = gt[name]
        color = class_color[cls]

        # Render the mesh thumbnail
        V, F = meshes[name]['V'], meshes[name]['F']
        try:
            thumb = render_mesh_thumbnail(V, F, size=thumb_size)
        except Exception as e:
            print(f"  Render failed for {name}: {e}")
            thumb = np.ones((thumb_size, thumb_size, 4), dtype=np.uint8) * 240
            thumb[:, :, 3] = 255

        # Show thumbnail with a colored border
        im = OffsetImage(thumb, zoom=1.0)
        ab = AnnotationBbox(
            im, (0.5, 0.5), frameon=True,
            bboxprops=dict(edgecolor=color, linewidth=3,
                           facecolor='white', boxstyle='round,pad=0.05'),
            xycoords='axes fraction', box_alignment=(0.5, 0.5)
        )
        ax.add_artist(ab)

        # Optional: add class label below each image
        ax.text(0.5, -0.05, cls, transform=ax.transAxes,
                ha='center', va='top', fontsize=8, color='#333')

    # ---- 5. Remove empty subplots ----
    for j in range(idx + 1, rows * cols):
        axes[j // cols, j % cols].axis('off')

    plt.suptitle(f"Zoom on Cluster: {', '.join(class_names) if class_names else 'Selected region'}",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Zoomed cluster view saved → {out_path}")

# ── descriptor pipelines ───────────────────────────────────────────────────────
def get_evals_evecs(mesh, k=K_EVALS):
    return laplace_beltrami_eigenvalues(mesh, k=k, return_eigenvectors=True,
                                        mass_matrix_type='voronoi')

def pipeline_shapedna(mesh):
    evals, _ = get_evals_evecs(mesh)
    evals = np.abs(evals)
    nonzero = evals[evals > 1e-6]
    if len(nonzero) == 0: return evals
    return evals / nonzero[0]

def pipeline_hks(mesh):
    evals, evecs = get_evals_evecs(mesh)
    evals = np.abs(evals)
    hks, _ = compute_hks(evals, evecs, n_times=20)
    return hks         # (n_times,) global descriptor

def pipeline_sihks(mesh):
    evals, evecs = get_evals_evecs(mesh)
    evals = np.abs(evals)
    sihks, _ = compute_scale_invariant_hks(evals, evecs, n_times=20)
    return sihks

def pipeline_sihks_norm(mesh):
    evals, evecs = get_evals_evecs(mesh)
    evals = np.abs(evals)
    sihks, _ = compute_sihks_norm(evals, evecs, n_times=20)
    return sihks

def pipeline_wks(mesh):
    evals, evecs = get_evals_evecs(mesh)
    evals = np.abs(evals)
    wks, _ = compute_wks(evals, evecs, n_energies=100)
    return wks

def pipeline_amks(mesh):
    if isinstance(mesh, dict):
        V, F = mesh['V'], mesh['F']
    else:
        V, F = mesh
    desc = compute_amks(V, F, k=min(K_EVALS, V.shape[0]-2))
    return desc.mean(axis=0)

PIPELINES = {
    'ShapeDNA': pipeline_shapedna,
    'HKS':      pipeline_hks,
    'SIHKS':    pipeline_sihks,
    'SIHKSnorm':    pipeline_sihks_norm,
    'WKS':      pipeline_wks,
    #'AMKS':     pipeline_amks,
}

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    # Load meshes
    print("Loading meshes...")
    t0     = time.time()
    meshes = load_meshes_cached(DATA_DIR, cache_file=MESH_CACHE)
    print(f"  {len(meshes)} meshes in {time.time()-t0:.1f}s")

    # Ground truth
    raw_gt = load_cla_ground_truth(GT_FILE)
    names  = list(meshes.keys())
    gt     = build_ground_truth(names, raw_gt)
    print(f"  GT covers {len(gt)}/{len(names)} meshes, "
          f"{len(set(gt.values()))} classes")

    # Verify class sizes
    from collections import Counter
    cc = Counter(gt.values())
    print(f"  Class sizes: min={min(cc.values())} max={max(cc.values())}")

    runtime_rows = []   # (name, n_meshes, total_s, per_mesh_s, mAP, p10)

    for desc_name, pipeline in PIPELINES.items():
        print(f"\n{'='*60}")
        print(f"  {desc_name}")
        print(f"{'='*60}")

        desc_dict  = {}   # name → 1-D feature
        desc_list  = []   # same order as valid_names
        valid_names= []
        t_start    = time.time()

        for name, mesh in meshes.items():
            if name not in gt:
                continue
            try:
                feat = pipeline(mesh)
                desc_dict[name]  = feat
                desc_list.append(feat)
                valid_names.append(name)
            except Exception as e:
                print(f"    Failed {name}: {e}")

        elapsed     = time.time() - t_start
        n_ok        = len(valid_names)
        per_mesh    = elapsed / n_ok if n_ok else 0

        print(f"  Computed {n_ok} descriptors in {elapsed:.1f}s "
              f"({per_mesh:.2f}s/mesh)")

        # Metrics
        mAP, p10 = evaluate(desc_dict, valid_names, gt)
        print(f"  mAP={mAP:.4f}  P@10={p10:.4f}")

        runtime_rows.append((desc_name, n_ok, elapsed, per_mesh, mAP, p10))

        # Cluster plot
        out_png = os.path.join(OUT_DIR, f"cluster_{desc_name.lower()}.png")
        # Thumbnail cluster plot (existing)
        out_png = os.path.join(OUT_DIR, f"cluster_{desc_name.lower()}.png")
        plot_cluster(
            descriptors=[desc_dict[n] for n in valid_names],
            names=valid_names,
            gt=gt,
            meshes=meshes,
            title=f"{desc_name} t-SNE Cluster (SHREC-11)",
            out_path=out_png,
        )

        # Points-only cluster plot
        out_pts = os.path.join(OUT_DIR, f"cluster_{desc_name.lower()}_points.png")
        plot_cluster_points(
            descriptors=[desc_dict[n] for n in valid_names],
            names=valid_names,
            gt=gt,
            title=f"{desc_name} t-SNE Cluster (SHREC-11)",
            out_path=out_pts,
            n_sample= 542
        )

        # After running plot_cluster for ShapeDNA
        # zoom_on_cluster(
        #     names=valid_names,                 # the same list used for t‑SNE
        #     gt=gt,
        #     meshes=meshes,
        #     class_names=['gorilla'],             # single class or list of classes
        #     out_path=os.path.join(OUT_DIR, "zoom_human_class.png"),
        #     thumb_size=200,
        #     cols=5
        # )

    # ── runtime table ──────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    header = f"{'Descriptor':<12} {'Meshes':>7} {'Total(s)':>10} "
    header += f"{'Per mesh(s)':>12} {'mAP':>8} {'P@10':>8}"
    print(header)
    print("-"*65)
    for r in runtime_rows:
        print(f"{r[0]:<12} {r[1]:>7} {r[2]:>10.1f} {r[3]:>12.2f} "
              f"{r[4]:>8.4f} {r[5]:>8.4f}")
    print("="*65)

    table_png = os.path.join(OUT_DIR, "runtime_table.png")
    save_runtime_table(runtime_rows, table_png)

    print(f"\nAll outputs saved to ./{OUT_DIR}/")

if __name__ == '__main__':
    main()