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