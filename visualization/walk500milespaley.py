import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.colors import Normalize


# Paley graph parameters
p = 17  # prime congruent to 1 mod 4
nodes = list(range(p))

# quadratic residues mod p
residues = set((i*i) % p for i in range(1, p))
if 0 in residues:
    residues.remove(0)

G = nx.Graph()
G.add_nodes_from(nodes)
for i in nodes:
    for j in nodes:
        if i < j:
            diff = (i - j) % p
            if diff in residues:
                G.add_edge(i, j)

pos = nx.circular_layout(G, scale=2.0)
A = nx.to_numpy_array(G, nodelist=nodes)
deg = A.sum(axis=1)
D_inv = np.diag(1.0 / deg)
W = A @ D_inv
tildeW = 0.5 * np.eye(p) + 0.5 * W

# Transition probabilities
P = {}
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    if neighbors:
        P[node] = neighbors

# Initialize walker
current_node = np.random.choice(G.nodes())
visited = np.zeros(len(G.nodes()))

# Color map
cmap = plt.colormaps["plasma"]
norm = Normalize(vmin=0, vmax=1)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Set up figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.axis("off")

# Draw initial state
nodes = nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color="gray", node_size=300)
edges_collection = nx.draw_networkx_edges(G, pos=pos, ax=ax, alpha=0.3, edge_color="gray", width=1.0)
labels = nx.draw_networkx_labels(G, pos=pos, ax=ax, font_color="white")

# Convert edges to list for easy reference
edges = list(G.edges())

cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Visit Probability', rotation=270, labelpad=15)

def update(frame):
    global current_node

    # Randomly choose a neighbor
    if P[current_node]:
        next_node = np.random.choice(P[current_node])
    else:
        # Restart if isolated node (shouldn't happen in connected graph)
        next_node = np.random.choice(list(G.nodes()))

    # Record visit
    visited[next_node] += 1

    # Compute probabilities (normalize visit counts)
    probs = visited / (np.sum(visited) + 1e-8) 
    colors = [cmap(norm(p)) for p in probs]
    colors[current_node] = 'yellow'
    nodes.set_color(colors)

    # Update node sizes (based on probability)
    sizes = 300 + 2000 * probs
    nodes.set_sizes(sizes)

    # Update edge transparency
    edge_colors = []
    for u, v in edges:
        if (u == current_node and v == next_node) or (v == current_node and u == next_node):
            edge_colors.append("yellow")
        else:
            edge_colors.append("gray")
    edges_collection.set_edgecolor(edge_colors)

    # Move walker
    current_node = next_node

    ax.set_title(f"Lazy Random Walk on Paley Graph (p={p}) — Step {frame}", fontsize=12)
    return nodes, edges_collection

anim = animation.FuncAnimation(fig, update, frames=80, interval=120, blit=False)
plt.show()