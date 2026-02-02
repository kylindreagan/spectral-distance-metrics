#Assistance from https://github.com/raphaelsulzer/ShapeDNA, "Discrete Laplace-Beltrami Operators for Shape Analysis and Segmentation" Reuter et al 2009
import igl
import numpy as np
from scipy.sparse.linalg import eigsh

def laplace_beltrami_eigenvalues(mesh, k=200, s=0, mass_matrix_type='voronoi'):
    #k = number of eigenvalues (Should be min ~50), s = sigma for eigsh (Find eigenvalues near this value), 
    if isinstance(mesh, dict):
        V = mesh['V']  # n x 3 array
        F = mesh['F']  # m x 3 array
    else:
        V, F = mesh  # Assume tuple
    
    n_vertices = V.shape[0]
    
    L = igl.cotmatrix(V, F) #Sparse cotangent Laplacian
    if mass_matrix_type.lower() == 'voronoi' or mass_matrix_type.lower() == 'lumped_voronoi':
        M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
        
    elif mass_matrix_type.lower() == 'barycentric' or mass_matrix_type.lower() == 'lumped_barycentric':
        M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC)
        
    elif mass_matrix_type.lower() == 'full':
        # Note: eigsh requires positive definite M, full matrix may cause issues
        M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_FULL)
        
    else:
        raise ValueError(f"Unknown mass_matrix_type: {mass_matrix_type}. "
                         f"Options: 'voronoi', 'barycentric', 'full'")

    num_eigenvalues = min(k, n_vertices - 1)

    eigenvalues, eigenvectors = eigsh(
        A=L, 
        M=M, 
        k=num_eigenvalues,
        sigma=0,  # Find eigenvalues near 0
        which='LM',  # Largest magnitude
        maxiter=3000
    )

    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    return eigenvalues, eigenvectors