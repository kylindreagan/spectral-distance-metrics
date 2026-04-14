import igl
import numpy as np
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
import trimesh

def laplace_beltrami_eigenvalues(mesh, k=200, return_eigenvectors=False, mass_matrix_type='voronoi'):
    if isinstance(mesh, dict):
        V = mesh['V']
        F = mesh['F']
    elif isinstance(mesh, trimesh.Trimesh):
        V = mesh.vertices
        F = mesh.faces
    else:
        V, F = mesh

    n_vertices = V.shape[0]
    L = igl.cotmatrix(V, F)

    if mass_matrix_type.lower() in ('voronoi', 'lumped_voronoi'):
        M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
    elif mass_matrix_type.lower() in ('barycentric', 'lumped_barycentric'):
        M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    elif mass_matrix_type.lower() == 'full':
        M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_FULL)
    else:
        raise ValueError(f"Unknown mass_matrix_type: {mass_matrix_type}")

    num_eigenvalues = min(k, n_vertices - 2)

    try:
        result = eigsh(
            A=-L,
            M=M,
            k=num_eigenvalues,
            sigma=0,
            which='LM',        # LM after shift = smallest original eigenvalues
            maxiter=10000,
            tol=1e-6,
            return_eigenvectors=return_eigenvectors
        )
    except ArpackNoConvergence as e:
        # Rescue partial results rather than failing entirely
        if return_eigenvectors:
            eigenvalues, eigenvectors = e.eigenvalues, e.eigenvectors
        else:
            eigenvalues = e.eigenvalues

        partial = np.sort(np.abs(eigenvalues))
        if len(partial) < num_eigenvalues:
            partial = np.pad(partial, (0, num_eigenvalues - len(partial)),
                             constant_values=partial[-1] if len(partial) > 0 else 0.0)
        if return_eigenvectors:
            return partial[:num_eigenvalues], eigenvectors
        return partial[:num_eigenvalues]

    if return_eigenvectors:
        eigenvalues, eigenvectors = result
        sort_idx = np.argsort(eigenvalues)
        return eigenvalues[sort_idx], eigenvectors[:, sort_idx]
    else:
        return np.sort(result)