# spectral_clustering_demo.py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import eigh

# optional sklearn parts (make_moons, kneighbors_graph, KMeans)
try:
    from sklearn.datasets import make_moons
    from sklearn.neighbors import kneighbors_graph
    from sklearn.cluster import KMeans
    SK = True
except Exception:
    SK = False
    print("sklearn not available, will fall back to basic blobs and simple kmeans.")

def build_affinity(X, sigma=0.5, k=10):
    # RBF affinities, sparsified by k-NN
    sqdist = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)
    A_full = np.exp(-sqdist / (2 * sigma**2))
    if SK:
        knn = kneighbors_graph(X, n_neighbors=k, include_self=False).toarray()
        A = A_full * (knn + knn.T > 0)
    else:
        # fallback: threshold by distance to k-th nearest neighbor
        dists = np.sort(sqdist, axis=1)
        kth = np.sqrt(dists[:, k])
        mask = sqdist <= (kth[:, None]**2)
        A = A_full * mask
        A = np.maximum(A, A.T)
    return A

def main():
    if SK:
        X, true = make_moons(n_samples=200, noise=0.06, random_state=0)
    else:
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(loc=[-1,0], scale=0.3, size=(100,2)),
                       rng.normal(loc=[1,0], scale=0.3, size=(100,2))])
    A = build_affinity(X, sigma=0.5, k=10)
    # normalized Laplacian
    D = np.diag(A.sum(axis=1))
    D_sqrt_inv = np.linalg.inv(np.sqrt(D))
    L = np.eye(A.shape[0]) - D_sqrt_inv @ A @ D_sqrt_inv
    w, U = eigh(L)
    # embedding: use second and third eigenvectors (skip trivial first)
    embed = U[:, 1:3]
    # kmeans
    if SK:
        km = KMeans(n_clusters=2, random_state=0).fit(embed)
        labels = km.labels_
    else:
        # simple kmeans fallback on 2 clusters
        def simple_kmeans(X, k=2, n_iter=100):
            rng = np.random.default_rng(0)
            n = X.shape[0]
            idx = rng.choice(n, k, replace=False)
            centers = X[idx]
            for _ in range(n_iter):
                d = np.sum((X[:,None,:] - centers[None,:,:])**2, axis=2)
                labels = np.argmin(d, axis=1)
                new_centers = np.vstack([X[labels==j].mean(axis=0) for j in range(k)])
                if np.allclose(new_centers, centers, atol=1e-6):
                    break
                centers = new_centers
            return labels
        labels = simple_kmeans(embed, k=2)

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    axes[0].scatter(X[:,0], X[:,1], c=labels, s=30)
    axes[0].set_title("Spectral Clustering result")
    axes[1].scatter(embed[:,0], embed[:,1], c=labels, s=30)
    axes[1].set_title("Rows of eigenvector matrix (embedding)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
