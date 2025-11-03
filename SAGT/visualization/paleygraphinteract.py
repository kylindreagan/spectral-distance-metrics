# Paley graph 3D spectral embedding (interactive) + adjacency spectrum plot
# - Try Plotly for interactive 3D; fallback to matplotlib 3D scatter if Plotly isn't present.
# - q must be congruent to 1 (mod 4). Default q=29 is a good size for demonstration.

import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Try to import plotly for interactive 3D; if unavailable, we use matplotlib 3D.
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

from mpl_toolkits.mplot3d import Axes3D  # needed for matplotlib 3D

def quadratic_residues(q):
    """Return set of nonzero quadratic residues modulo q."""
    return {pow(x, 2, q) for x in range(1, q)}

def paley_graph(q):
    """Construct the Paley graph of order q (requires q % 4 == 1)."""
    if q % 4 != 1:
        raise ValueError("q must be congruent to 1 mod 4 for a Paley graph.")
    G = nx.Graph()
    residues = quadratic_residues(q)
    G.add_nodes_from(range(q))
    for a in range(q):
        for b in range(a+1, q):
            if ((a - b) % q) in residues:
                G.add_edge(a, b)
    return G

# ---- PARAMETERS ----
q = 13   # change this to any prime-power q ≡ 1 (mod 4).
# --------------------

G = paley_graph(q)
A = nx.to_numpy_array(G, dtype=float)

# adjacency eigen-decomposition
eigvals, eigvecs = np.linalg.eig(A)

# Sort eigenvalues by absolute value (largest first) so the top eigenvectors capture major structure:
order = np.argsort(np.abs(eigvals))[::-1]
eigvals_sorted = eigvals[order]
eigvecs_sorted = eigvecs[:, order]

# Use top 3 eigenvectors (real parts) to create a 3D spectral embedding:
k = 3
coords = np.real(eigvecs_sorted[:, :k])
coords = coords - coords.mean(axis=0)       # center
scale = np.max(np.abs(coords))
if scale > 0:
    coords /= scale                         # normalize to [-1,1] range roughly

# --- Interactive 3D (Plotly) if available ---
if PLOTLY_AVAILABLE:
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        x0,y0,z0 = coords[u]
        x1,y1,z1 = coords[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    node_x, node_y, node_z = coords[:,0], coords[:,1], coords[:,2]

    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', hoverinfo='none', line=dict(width=1))
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        text=[str(i) for i in range(q)],
        textposition='top center',
        marker=dict(size=6),
        hoverinfo='text'
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"3D spectral embedding of Paley graph (q={q}) — rotate/zoom to explore",
        scene=dict(xaxis=dict(title='Eigvec 1'), yaxis=dict(title='Eigvec 2'), zaxis=dict(title='Eigvec 3')),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.show()
else:
    # Matplotlib 3D fallback (interactive rotating in most notebook UIs)
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=40)
    for u,v in G.edges():
        xs = [coords[u,0], coords[v,0]]
        ys = [coords[u,1], coords[v,1]]
        zs = [coords[u,2], coords[v,2]]
        ax.plot(xs, ys, zs, linewidth=0.6, alpha=0.7)
    ax.set_title(f"3D spectral embedding (matplotlib) of Paley graph (q={q})")
    ax.set_xlabel("Eigvec 1")
    ax.set_ylabel("Eigvec 2")
    ax.set_zlabel("Eigvec 3")
    plt.tight_layout()
    plt.show()

# --- Separate adjacency spectrum plot (matplotlib; its own figure) ---
eigvals_real_sorted = np.sort(np.real(eigvals))   # Paley graphs yield real eigenvalues
plt.figure(figsize=(8,4))
plt.stem(range(len(eigvals_real_sorted)), eigvals_real_sorted, use_line_collection=True)
plt.title(f"Adjacency spectrum (sorted real parts) — Paley graph q={q}")
plt.xlabel("Index (sorted)")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.show()

# --- Numeric summary printed to console ---
unique_vals, counts = np.unique(np.round(eigvals_real_sorted, 8), return_counts=True)
print("Distinct eigenvalues (rounded) and multiplicities:")
for val, cnt in zip(unique_vals, counts):
    print(f"  {val}  multiplicity {cnt}")

# Optional: overlay theoretical eigenvalues lines (if you want)
theoretical_big = (q - 1) / 2
theoretical_other_plus = (-1 + math.sqrt(q)) / 2
theoretical_other_minus = (-1 - math.sqrt(q)) / 2
print("\nTheoretical eigenvalues for Paley graph of order q:")
print(f"  (q-1)/2 = {theoretical_big}")
print(f"  (-1 + sqrt(q))/2 = {theoretical_other_plus}")
print(f"  (-1 - sqrt(q))/2 = {theoretical_other_minus}")
