import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, spdiags
from scipy.sparse.linalg import eigsh

def cotangent_laplacian(vertices, faces):
    """
    Build cotangent stiffness matrix C and diagonal mass matrix M.
    vertices: (n, 3) array
    faces: (f, 3) array of vertex indices
    Returns:
        C: scipy.sparse.csr_matrix (n, n)
        M: numpy.ndarray (n,) diagonal entries
    """
    n = len(vertices)
    C = lil_matrix((n, n))
    M = np.zeros(n)

    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        # Edge vectors
        e01 = v1 - v0
        e12 = v2 - v1
        e20 = v0 - v2

        # Lengths
        l01 = np.linalg.norm(e01)
        l12 = np.linalg.norm(e12)
        l20 = np.linalg.norm(e20)

        # Angles and cotangents
        # Angle at v0
        cos0 = np.dot(e01, -e20) / (l01 * l20)
        sin0 = np.linalg.norm(np.cross(e01, -e20)) / (l01 * l20)
        cot0 = cos0 / sin0 if sin0 > 1e-10 else 0.0

        # Angle at v1
        cos1 = np.dot(e12, -e01) / (l12 * l01)
        sin1 = np.linalg.norm(np.cross(e12, -e01)) / (l12 * l01)
        cot1 = cos1 / sin1 if sin1 > 1e-10 else 0.0

        # Angle at v2
        cos2 = np.dot(e20, -e12) / (l20 * l12)
        sin2 = np.linalg.norm(np.cross(e20, -e12)) / (l20 * l12)
        cot2 = cos2 / sin2 if sin2 > 1e-10 else 0.0

        # Off‑diagonal contributions (negative cotangents)
        # Edge (v1,v2) opposite v0
        C[face[1], face[2]] += -cot0
        C[face[2], face[1]] += -cot0
        # Edge (v0,v1) opposite v2
        C[face[0], face[1]] += -cot2
        C[face[1], face[0]] += -cot2
        # Edge (v0,v2) opposite v1
        C[face[0], face[2]] += -cot1
        C[face[2], face[0]] += -cot1

        # Triangle area (1/2 * |cross|) and mass contribution (1/3 per vertex)
        area = 0.5 * np.linalg.norm(np.cross(e01, e12))
        M[face[0]] += area / 3.0
        M[face[1]] += area / 3.0
        M[face[2]] += area / 3.0

    # Diagonal entries: sum of absolute off‑diagonals
    for i in range(n):
        row_sum = -C[i, :].sum()   # off‑diagonals are negative
        C[i, i] = row_sum

    return csr_matrix(C), M


def compute_amks(vertices, faces, T=0.5, k=100, d=100, sigma=0.1):
    """
    Compute AMKS descriptor for each vertex.
    vertices: (n, 3) array
    faces: (f, 3) array
    T: stopping time
    k: number of eigenvectors to use
    d: number of energy levels (descriptor length)
    sigma: width of Gaussian filter in log-energy domain
    Returns:
        descriptors: (n, d) array, each row is the descriptor for a vertex
    """
    # --- 1. Scale mesh to unit area ---
    total_area = 0.0
    for face in faces:
        v0, v1, v2 = vertices[face]
        e01 = v1 - v0
        e12 = v2 - v1
        area = 0.5 * np.linalg.norm(np.cross(e01, e12))
        total_area += area
    scale = 1.0 / np.sqrt(total_area)
    vertices_scaled = vertices * scale

    # --- 2. Build Laplacian and mass matrices ---
    C, M = cotangent_laplacian(vertices_scaled, faces)
    n = len(vertices_scaled)

    # --- 3. Solve generalized eigenvalue problem via symmetric normalization ---
    # L_sym = M^{-1/2} C M^{-1/2}
    M_half_inv = 1.0 / np.sqrt(M)                # diagonal entries
    M_half_inv_diag = spdiags(M_half_inv, 0, n, n)
    L_sym = M_half_inv_diag @ C @ M_half_inv_diag

    # Compute smallest k+1 eigenvalues/vectors (to discard the zero eigenvalue)
    eigvals, eigvecs_sym = eigsh(L_sym, k=k+1, which='SM', tol=1e-6, maxiter=1000)

    # Sort and discard the first (zero) eigenpair
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx][1:]          # keep only non‑zero
    eigvecs_sym = eigvecs_sym[:, idx][:, 1:]

    # Transform to eigenvectors of the original Laplacian (M‑orthonormal)
    eigvecs = eigvecs_sym * M_half_inv[:, np.newaxis]   # (n, k)

    # --- 4. Precompute squared eigenfunctions ---
    # S_T: (k, n) where S_T[i,u] = φ_i(u)^2
    S_T = (eigvecs ** 2).T

    # --- 5. Sinc matrix A: A_{i,j} = sinc(T*(λ_j - λ_i)) ---
    lam = eigvals.reshape(-1, 1)
    diff = lam - lam.T
    # Use numpy.sinc with proper scaling: sinc(x) = sin(x)/x, but np.sinc(y) = sin(π y)/(π y)
    # We need y = x/π, so np.sinc(diff*T / np.pi) gives sin(diff*T)/(diff*T)
    A = np.sinc(diff * T / np.pi)
    np.fill_diagonal(A, 1.0)           # ensure diagonal is exactly 1

    # --- 6. Energy levels ---
    log_lam = np.log(eigvals)
    E_min = np.min(log_lam)
    E_max = np.max(log_lam)
    E_vals = np.linspace(E_min, E_max, d)
    delta_E = (E_max - E_min) / (d - 1)

    descriptors = np.zeros((n, d))

    # --- 7. Loop over energies ---
    for l, E in enumerate(E_vals):
        # Filter values f_i = exp(-(E - log λ_i)^2 / (2 σ^2))
        f = np.exp(-((E - log_lam) ** 2) / (2.0 * sigma ** 2))
        f_outer = np.outer(f, f)                # (k, k)
        B = A * f_outer                         # element‑wise product
        Y = B @ S_T                              # (k, n)
        N = np.sum(S_T * Y, axis=0)             # (n,)
        D = (np.sum(f)) ** 2
        descriptors[:, l] = N / D

    # --- 8. Normalize each vertex descriptor to integrate to 1 over energy ---
    integrals = np.sum(descriptors, axis=1) * delta_E
    descriptors = descriptors / integrals[:, np.newaxis]

    return descriptors