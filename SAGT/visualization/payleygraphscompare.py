import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# ---------- Paley Graph Definition ----------
def quadratic_residues(q):
    return {pow(x, 2, q) for x in range(1, q)}

def paley_graph(q):
    if q % 4 != 1:
        raise ValueError("q must be congruent to 1 mod 4 for a Paley graph.")
    G = nx.Graph()
    residues = quadratic_residues(q)
    G.add_nodes_from(range(q))
    for a in range(q):
        for b in range(a + 1, q):
            if ((a - b) % q) in residues:
                G.add_edge(a, b)
    return G

# ---------- Parameters ----------
q = 13  # prime ≡ 1 (mod 4)
G = paley_graph(q)
A = nx.to_numpy_array(G, dtype=float)
L = nx.laplacian_matrix(G).toarray()

# ---------- Spectral Embeddings ----------
adj_eigvals, adj_eigvecs = np.linalg.eigh(A)
idxA = np.argsort(np.abs(adj_eigvals))[::-1]
adj_coords = np.real(adj_eigvecs[:, idxA[:3]])

lap_eigvals, lap_eigvecs = np.linalg.eigh(L)
idxL = np.argsort(lap_eigvals)
lap_coords = np.real(lap_eigvecs[:, idxL[1:4]])  # skip the trivial eigenvector

# Normalize coordinates for display
def normalize(coords):
    coords -= coords.mean(axis=0)
    coords /= np.max(np.abs(coords))
    return coords

adj_coords = normalize(adj_coords)
lap_coords = normalize(lap_coords)

# ---------- Plot Setup ----------
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

def plot_graph(ax, coords, color, title):
    # Draw edges
    for u, v in G.edges():
        ax.plot([coords[u,0], coords[v,0]],
                [coords[u,1], coords[v,1]],
                [coords[u,2], coords[v,2]],
                c='lightgray', linewidth=0.5)
    # Draw nodes
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=color, s=50, edgecolor='k', depthshade=True)
    # Add labels
    for i, (x, y, z) in enumerate(coords):
        ax.text(x, y, z, str(i), fontsize=8, color='black', ha='center', va='center')
    # Style
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel('eigvec 1'); ax.set_ylabel('eigvec 2'); ax.set_zlabel('eigvec 3')
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

plot_graph(ax1, adj_coords, 'skyblue', f"Adjacency spectral embedding (q={q})")
plot_graph(ax2, lap_coords, 'lightgreen', f"Laplacian spectral embedding (q={q})")

# ---------- Animation Function ----------
def update(angle):
    ax1.view_init(30, angle)
    ax2.view_init(30, angle)
    return fig,

ani = animation.FuncAnimation(fig, update, frames=range(0, 360, 2), interval=50, blit=False)

# ---------- Save or Display ----------
# To save to video or gif, uncomment one:
# ani.save("paley_embeddings_labeled.mp4", fps=30, dpi=150)
# ani.save("paley_embeddings_labeled.gif", writer='pillow', fps=20)

plt.tight_layout()
plt.show()
