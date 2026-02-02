# path_eigenmodes.py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import eigh

def main(n=30):
    G = nx.path_graph(n)
    L = nx.laplacian_matrix(G).toarray()
    w, v = eigh(L)
    low_idx = 1
    high_idx = -1
    low_mode = v[:, low_idx]
    high_mode = v[:, high_idx]

    x = np.arange(n)
    plt.figure(figsize=(8,3.5))
    plt.plot(x, low_mode, marker='o', label=f'Low mode (eig {low_idx}, λ={w[low_idx]:.4f})')
    plt.plot(x, high_mode, marker='s', label=f'High mode (eig {n-1}, λ={w[-1]:.4f})')
    plt.axhline(0, linestyle='--', linewidth=0.6)
    plt.xlabel("vertex index")
    plt.ylabel("eigenvector value")
    plt.title("Low vs High frequency eigenmodes on Path Graph $P_n$")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(30)
