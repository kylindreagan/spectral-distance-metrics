import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation

# Create a simple connected graph (you can replace this with a Paley or any other)
G = nx.erdos_renyi_graph(10, 0.3, seed=42)
pos = nx.spring_layout(G, seed=42)

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

# Set up figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.axis("off")

# Draw initial state
nodes = nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color="gray", node_size=300)
edges_collection = nx.draw_networkx_edges(G, pos=pos, ax=ax, alpha=0.3, edge_color="gray", width=1.0)
labels = nx.draw_networkx_labels(G, pos=pos, ax=ax, font_color="white")

# Convert edges to list for easy reference
edges = list(G.edges())

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
    probs = visited / np.sum(visited)
    colors = [cmap(p) for p in probs]
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

    ax.set_title(f"Random Walk on Graph— Step {frame}", fontsize=14)
    return nodes, edges_collection

# Animate
anim = FuncAnimation(fig, update, frames=200, interval=400, blit=False, repeat=False)
plt.show()

# Save animation as GIF
anim.save("other_random_walk.gif", writer="pillow", fps=10)
print("GIF saved as other_random_walk.gif")
