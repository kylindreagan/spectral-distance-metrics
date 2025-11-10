# cheeger_sweep_demo.py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from matplotlib.colors import Normalize

def conductance(G, S):
    S = set(S)
    V = set(G.nodes())
    degs = dict(G.degree(weight='weight'))
    volS = sum(degs.get(v, 1) for v in S)
    volT = sum(degs.get(v, 1) for v in V - S)
    cut = 0.0
    for u in S:
        for v in G[u]:
            if v not in S:
                cut += G[u][v].get('weight', 1.0)
    if min(volS, volT) == 0:
        return np.inf
    return cut / min(volS, volT)

def main():
    # build a 2-block stochastic block model
    sizes = [30, 30]
    p_in = 0.15
    p_out = 0.01
    probs = [[p_in, p_out], [p_out, p_in]]
    G = nx.stochastic_block_model(sizes, probs, seed=42)

    # normalized Laplacian and Fiedler vector
    L = nx.normalized_laplacian_matrix(G).astype(float)
    vals, vecs = eigsh(L, k=2, which='SM')  # smallest eigenvalues
    lambda2 = vals[1]
    fiedler = vecs[:, 1]

    # sweep cut
    order = np.argsort(fiedler)[::-1]
    conds = []
    best_cond = np.inf
    best_set = None
    for r in range(1, len(order)):
        S = order[:r]
        cond = conductance(G, S)
        conds.append(cond)
        if cond < best_cond:
            best_cond = cond
            best_set = set(S)

    print("lambda2 (normalized) =", lambda2)
    print("best conductance found =", best_cond)

    # plotting
    pos = nx.spring_layout(G, seed=1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    cmap = plt.colormaps['viridis']
    nx.draw_networkx_edges(G, pos, ax=axs[0], alpha=0.3)
    nx.draw_networkx_nodes(G, pos,
                           node_color=fiedler,
                           cmap=cmap,
                           node_size=80,
                           ax=axs[0])
    nx.draw_networkx_nodes(G, pos,
                           nodelist=list(best_set),
                           node_color='none',
                           edgecolors='r',
                           linewidths=1.5,
                           node_size=200,
                           ax=axs[0])
    axs[0].set_title("SBM colored by Fiedler vector (best cut highlighted)")

    axs[1].plot(range(1, len(order)), conds, marker='o', markersize=4)
    axs[1].axhline(lambda2/2, linestyle='--', label=r'$\lambda_2/2$')
    axs[1].axhline(np.sqrt(2*lambda2), linestyle='--', label=r'$\sqrt{2\lambda_2}$')
    axs[1].set_xlabel("prefix size in sweep (r)")
    axs[1].set_ylabel("conductance of S_r")
    axs[1].set_title("Sweep cut conductance vs prefix (Cheeger bounds)")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()