# diffusion_maps_demo.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import networkx as nx
from matplotlib import animation

# If sklearn exists, use it to make a data set; otherwise we make blobs
try:
    from sklearn.datasets import make_moons
    from sklearn.neighbors import kneighbors_graph
    SK = True
except Exception:
    SK = False

def build_affinity(X, sigma=0.5, k=10):
    sqdist = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    A_full = np.exp(-sqdist / (2 * sigma**2))
    if SK:
        knn = kneighbors_graph(X, n_neighbors=k, include_self=False).toarray()
        A = A_full * (knn + knn.T > 0)
    else:
        # fallback: sparsify symmetrically by nearest neighbors by distance
        n = X.shape[0]
        idx = np.argsort(sqdist, axis=1)[:, 1:k+1]
        mask = np.zeros_like(sqdist, dtype=bool)
        for i in range(n):
            mask[i, idx[i]] = True
        mask = np.logical_or(mask, mask.T)
        A = A_full * mask
    return A

def diffusion_map_embedding(A, m=3):
    # A: symmetric affinity
    D = np.diag(A.sum(axis=1))
    D_sqrt_inv = np.linalg.inv(np.sqrt(D))
    S = D_sqrt_inv @ A @ D_sqrt_inv  # symmetric normalized matrix
    evals, evecs = eigh(S)
    # sort descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    # convert to psi eigenvectors of the random-walk operator:
    psi = np.linalg.inv(np.sqrt(D)) @ evecs
    return evals, psi

def diffusion_map_coords(psi, evals, t=1, m=3):
    # skip the first trivial eigenvector (index 0)
    coords = np.zeros((psi.shape[0], m))
    for j in range(1, m+1):
        coords[:, j-1] = (evals[j]**t) * psi[:, j]
    return coords

def main():
    if SK:
        X, _ = make_moons(n_samples=200, noise=0.06, random_state=0)
    else:
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(loc=[-1,0], scale=0.3, size=(100,2)),
                       rng.normal(loc=[1,0], scale=0.3, size=(100,2))])

    A = build_affinity(X, sigma=0.5, k=10)
    evals, psi = diffusion_map_embedding(A, m=10)

    ts = [1, 2, 4, 8, 16]
    embeddings = [diffusion_map_coords(psi, evals, t=t, m=3) for t in ts]

    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(embeddings[0][:,0], embeddings[0][:,1], s=30)
    ax.set_title(f"Diffusion map (t={ts[0]})")

    def update(i):
        emb = embeddings[i]
        sc.set_offsets(emb[:, :2])
        ax.set_title(f"Diffusion map (t={ts[i]})")
        return sc,

    anim = animation.FuncAnimation(fig, update, frames=len(ts), interval=800, blit=True)

    # You can save the animation as mp4 if ffmpeg is installed:
    # anim.save('diffusion_map_evolution.mp4', writer='ffmpeg', fps=1.5)
    plt.show()

    # Also show first and last embedding side-by-side
    fig2, axes = plt.subplots(1,2, figsize=(10,4))
    axes[0].scatter(embeddings[0][:,0], embeddings[0][:,1], s=30)
    axes[0].set_title("Diffusion map (t=1)")
    axes[1].scatter(embeddings[-1][:,0], embeddings[-1][:,1], s=30)
    axes[1].set_title(f"Diffusion map (t={ts[-1]})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
