import sys
import os
from time import time
from scipy.spatial.distance import pdist, squareform

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from laplaceBeltramiShape import laplace_beltrami_eigenvalues_vectors
from GeneralFunctions.batch_processor import load_all_meshes

isospectral_drums = load_all_meshes("C:/Users/kylin/Documents/GitHub/Spectral-Shape-Analysis/data/Isospectral_Drums")
all_shape_dnas = []

for mesh in isospectral_drums:
    all_shape_dnas.append(laplace_beltrami_eigenvalues_vectors(isospectral_drums[mesh]['V'], isospectral_drums[mesh]['F'], k=40))

for arr in all_shape_dnas:
    print(len(arr))
D = squareform(pdist(all_shape_dnas, metric="euclidean"))
print(D)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import MDS

# Create a clearer visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Heatmap of distance matrix
axes[0].set_title("Distance Matrix Heatmap")
im = axes[0].imshow(D, cmap='viridis', aspect='auto')
plt.colorbar(im, ax=axes[0])
axes[0].set_xlabel("Mesh Index")
axes[0].set_ylabel("Mesh Index")

# 2. Hierarchical clustering dendrogram
axes[1].set_title("Hierarchical Clustering")
linkage_matrix = linkage(D, method='ward')
dendrogram(linkage_matrix, ax=axes[1], 
           labels=[f"Mesh {i}" for i in range(len(isospectral_drums))])
axes[1].set_xlabel("Meshes")
axes[1].set_ylabel("Distance")

# 3. MDS for 2D embedding
axes[2].set_title("2D MDS Projection")
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_result = mds.fit_transform(D)
scatter = axes[2].scatter(mds_result[:, 0], mds_result[:, 1], 
                         c=range(len(D)), cmap='tab20', s=100)

# Add labels to points
for i, (x, y) in enumerate(mds_result):
    axes[2].text(x, y, f'{i}', fontsize=9, ha='center', va='center')

plt.tight_layout()
plt.show()