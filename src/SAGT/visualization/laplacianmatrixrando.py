#For project presentation

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 1. Generate random weighted graph
def random_weighted_graph(n_nodes, edge_prob=0.4, weight_range=(0.5, 10.0)):
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < edge_prob:
                weight = round(random.uniform(*weight_range), 2)
                G.add_edge(i, j, weight=weight)
    return G

# Parameters
n = random.randint(5, 10)
p = 0.4
G = random_weighted_graph(n, p)

# 2. Layout with minimal edge overlap
# Try Graphviz first (best layouts)
try:
    pos = nx.nx_agraph.graphviz_layout(G, prog='neato')  # neato avoids overlaps well
    print("Using Graphviz layout (neato)")
except (ImportError, ModuleNotFoundError):
    # Fallback 1: Kamada‑Kawai (good general purpose)
    pos = nx.kamada_kawai_layout(G)
    print("Using Kamada-Kawai layout")
    # If graph is planar, use planar layout (no crossings at all)
    if nx.is_planar(G):
        pos = nx.planar_layout(G)
        print("Graph is planar. Using planar layout (zero edge crossings)")

# 3. Draw and save figure
plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=600)
nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# Draw edge labels with offset to avoid overlapping edges
edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                             font_size=10, label_pos=0.5,
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

plt.title("Random Weighted Graph")
plt.axis('off')
plt.savefig("weighted_graph.png", dpi=200, bbox_inches='tight')
plt.show()

# 4. Degree, adjacency, & Laplacian
nodes = list(G.nodes())
weighted_degree = dict(G.degree(weight='weight'))
print("\nWeighted degree of each node:")
for node, deg in weighted_degree.items():
    print(f"  Node {node}: {deg:.2f}")

n_nodes = G.number_of_nodes()
A = np.zeros((n_nodes, n_nodes))
for u, v, data in G.edges(data=True):
    i, j = nodes.index(u), nodes.index(v)
    A[i, j] = A[j, i] = data['weight']

print("\nWeighted adjacency matrix (A):")
print(np.round(A, 2))

D = np.diag([weighted_degree[node] for node in nodes])
L = D - A
print("\nLaplacian matrix (L = D - A):")
print(np.round(L, 2))