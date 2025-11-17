import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Sample a noisy circle
n = 400
theta = np.linspace(0, 2*np.pi, n)
x = np.column_stack([np.cos(theta), np.sin(theta)])
x += 0.03*np.random.randn(*x.shape)

# Build kNN graph
k = 10
nbrs = NearestNeighbors(n_neighbors=k).fit(x)
distances, indices = nbrs.kneighbors(x)

# Build weight matrix W
eps = np.mean(distances)**2
W = np.zeros((n,n))
for i in range(n):
    for j in indices[i]:
        W[i,j] = np.exp(-np.linalg.norm(x[i]-x[j])**2/eps)

# Degree and Laplacian
D = np.diag(W.sum(axis=1))
L = D - W

# Compute eigenvalues/vectors
vals, vecs = np.linalg.eigh(L)

# Plot first three eigenfunctions
plt.figure(figsize=(12,4))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.scatter(x[:,0], x[:,1], c=vecs[:,i], cmap="coolwarm")
    plt.title(f"Eigenfunction {i}")
    plt.axis("equal")
plt.show()