import igl
import numpy as np
from scipy.sparse.linalg import eigsh
import trimesh

def laplace_beltrami_eigenvalues(mesh, k=200, return_eigenvectors=False, mass_matrix_type='voronoi'):
    #k = number of eigenvalues (Should be min ~50), 
    if isinstance(mesh, dict):
        V = mesh['V']  # n x 3 array
        F = mesh['F']  # m x 3 array
    elif isinstance(mesh, trimesh.Trimesh):
        V = mesh.vertices
        F = mesh.faces
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
        A=-L, 
        M=M, 
        k=num_eigenvalues,
        sigma=None, 
        which='SM',  # Smallest magnitude
        maxiter=5000,
        tol=1e-6
    )

    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]
    eigenvalues[0] = max(eigenvalues[0], 0) #Ensure 1st eigenvalue nonnegative

    if return_eigenvectors:
        return eigenvalues, eigenvectors
    return eigenvalues

def laplace_beltrami_eigenvalues_vectors(V, F, k=200, return_eigenvectors=False, mass_matrix_type='voronoi'):
    #k = number of eigenvalues (Should be min ~50), 
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
        A=-L, 
        M=M, 
        k=num_eigenvalues,
        sigma=None, 
        which='SM',  # Smallest magnitude
        maxiter=5000,
        tol=1e-6
    )

    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]
    eigenvalues[0] = max(eigenvalues[0], 0) #Ensure 1st eigenvalue nonnegative

    if return_eigenvectors:
        return eigenvalues, eigenvectors
    return eigenvalues