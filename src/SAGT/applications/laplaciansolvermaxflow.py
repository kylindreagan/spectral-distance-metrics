# ---------- illustrative electrical flow ----------
# Uses Laplacian matrices and linear algebra to approximate maximum s-t flow.

import numpy as np
import networkx as nx
from typing import List, Tuple

def build_incidence_matrix(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    """Return oriented incidence matrix B (m x n) for edges list (u,v)."""
    m = len(edges)
    B = np.zeros((m, n))
    for i, (u, v) in enumerate(edges):
        B[i, u] = 1
        B[i, v] = -1
    return B

def laplacian_from_B_and_weights(B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return Laplacian L = B^T * W * B where W = diag(w)."""
    W = np.diag(w)
    return B.T @ W @ B

def electrical_flow(B: np.ndarray, w: np.ndarray, s: int, t: int, F: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute electrical flow for demand F from s to t.
    Returns (f, x) where f is signed flow on each oriented edge.
    """
    n = B.shape[1]
    L = laplacian_from_B_and_weights(B, w)
    # demand vector b (n)
    b = np.zeros(n)
    b[s] = F
    b[t] = -F
    # solve L x = b in subspace orthonormal to ones: use pseudoinverse (educational)
    L_pinv = np.linalg.pinv(L)
    x = L_pinv @ b
    f = w * (B @ x)    # f_e = w_e * (x_u - x_v)
    return f, x

def greedy_electrical_augment(
    n: int,
    edges: List[Tuple[int, int]],
    capacities: List[float],
    s: int,
    t: int,
    max_iters: int = 1000
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Greedy augmentation using electrical flow directions.
    Returns (total_flow, flow_on_edges, remaining_cap, potentials).
    """
    m = len(edges)
    B = build_incidence_matrix(n, edges)
    flow_on_edges = np.zeros(m)
    remaining_cap = np.array(capacities, dtype=float)
    total_flow = 0.0

    for it in range(max_iters):
        # conductances proportional to remaining capacity (small floor to avoid zeros)
        conductances = np.maximum(remaining_cap, 1e-12)
        f_dir, potentials = electrical_flow(B, conductances, s, t, F=1.0)
        abs_f = np.abs(f_dir)
        if np.max(abs_f) < 1e-12:
            break
        with np.errstate(divide='ignore', invalid='ignore'):
            alphas = np.where(abs_f > 1e-14, remaining_cap / abs_f, np.inf)
        alpha = np.min(alphas)
        if not np.isfinite(alpha) or alpha <= 1e-12:
            break
        augment = alpha * f_dir
        flow_on_edges += augment
        remaining_cap -= np.abs(augment)
        total_flow += alpha
        if np.max(remaining_cap) < 1e-12:
            break
    return total_flow, flow_on_edges, remaining_cap, potentials

# --------- demo ----------
if __name__ == "__main__":
    # small demo graph (4 nodes)
    n = 4
    # oriented edges (0-based): 1-2,2-3,3-4,2-4 is edges list
    edges = [(0,1), (1,2), (2,3), (1,3)]
    capacities = [1.0, 1.0, 1.0, 1.0]    # unit capacities
    s, t = 0, 3

    F_total, flow_on_edges, remaining_cap, potentials = greedy_electrical_augment(
        n, edges, capacities, s, t
    )

    print("Approximate flow value (greedy electrical):", F_total)
    print("Edge flows (signed wrt orientation):")
    for i,(u,v) in enumerate(edges):
        print(f" edge {u+1}-{v+1}: flow = {flow_on_edges[i]:.6f}, remaining cap = {remaining_cap[i]:.6f}")
    print("Node potentials:", [float(v) for v in potentials])
