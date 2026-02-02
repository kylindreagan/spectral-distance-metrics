import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def quadratic_residues(q):
    """Return set of nonzero quadratic residues modulo q."""
    return {pow(x, 2, q) for x in range(1, q)}

def paley_graph(q):
    """Construct the Paley graph of order q (requires q % 4 == 1)."""
    if q % 4 != 1:
        raise ValueError("q must be congruent to 1 mod 4 for a Paley graph.")
    G = nx.Graph()
    residues = quadratic_residues(q)
    G.add_nodes_from(range(q))
    for a in range(q):
        for b in range(a+1, q):
            if ((a - b) % q) in residues:
                G.add_edge(a, b)
    return G

def random_walk_mixing(G, steps=50, start=None):
    A = nx.to_numpy_array(G)
    D = np.diag(A.sum(axis=1))
    P = np.linalg.inv(D) @ A  # Transition matrix

    n = len(G)
    if start is None:
        start = np.random.randint(0, n)
    dist = np.zeros(n)
    dist[start] = 1.0

    pi = A.sum(axis=1) / A.sum()  # stationary distribution

    tvd = []
    for _ in range(steps):
        dist = dist @ P
        tvd.append(0.5 * np.sum(np.abs(dist - pi)))

    eigvals = np.linalg.eigvals(P)
    eigvals = np.sort(np.abs(eigvals))[::-1]
    gap = 1 - eigvals[1]
    return tvd, eigvals, gap

graphs = {
    "Cycle": nx.cycle_graph(20),
    "Complete": nx.complete_graph(20),
    "Random (Gnp)": nx.gnp_random_graph(20, 0.3),
    "Paley (q=13)": paley_graph(13)
}

plt.figure(figsize=(7,5))
for name, G in graphs.items():
    tvd, eigvals, gap = random_walk_mixing(G)
    plt.plot(tvd, label=f"{name} (λ₂={eigvals[1]:.2f}, gap={gap:.2f})")

plt.title("Random Walk Convergence to Stable Distribution")
plt.xlabel("Steps")
plt.ylabel("Total Variation Distance")
plt.legend()
plt.show()
