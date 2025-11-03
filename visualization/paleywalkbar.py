# paley_walk_bar_mp4.py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Normalize
from matplotlib import cm

# ---- Parameters ----
p = 17                # prime congruent to 1 mod 4
n_steps = 160         # number of frames / steps
fps = 10              # frames per second for mp4
node_base_size = 300
node_scale = 2200.0   # scale added to sizes by probability

# ---- Build Paley graph ----
nodes = list(range(p))
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

# positions for plotting (circular for symmetry)
pos = nx.circular_layout(G, scale=2.0)

# adjacency/degrees/walk matrices (not strictly needed for the empirical walk, but useful)
A = nx.to_numpy_array(G, nodelist=nodes)
deg = A.sum(axis=1)
D_inv = np.diag(1.0 / deg)
W = A @ D_inv
tildeW = 0.5 * np.eye(p) + 0.5 * W

# stationary distribution (analytic)
pi = deg / deg.sum()

# transition lists for random walk
neighbors = {node: list(G.neighbors(node)) for node in G.nodes()}

# ---- Random walk state ----
rng = np.random.default_rng(seed=1234)   # set seed for reproducibility
current = rng.choice(nodes)
visit_counts = np.zeros(p, dtype=float)
visit_counts[current] += 1

# ---- Matplotlib setup ----
plt.rcParams.update({"figure.max_open_warning": 0})
fig, (ax_graph, ax_bar) = plt.subplots(1, 2, figsize=(10, 5))
ax_graph.axis("off")

# colormap: use modern access to colormaps to avoid deprecation warnings
cmap = plt.colormaps["plasma"]
norm = Normalize(vmin=0.0, vmax=1.0)

# draw static graph (we'll update node facecolors / sizes each frame)
node_collection = nx.draw_networkx_nodes(G, pos=pos, ax=ax_graph,
                                         node_color=[0.0]*p,
                                         cmap=cmap,
                                         node_size=node_base_size,
                                         edgecolors='k', linewidths=0.5)
edge_collection = nx.draw_networkx_edges(G, pos=pos, ax=ax_graph, alpha=0.3)
labels = nx.draw_networkx_labels(G, pos=pos, ax=ax_graph, font_color='white', font_size=8)

ax_graph.set_title("Paley Graph: node probabilities (yellow = current)")
# colorbar for probability scale
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([0,1])
cbar = fig.colorbar(sm, ax=ax_graph, fraction=0.046, pad=0.02)
cbar.set_label("Visit prob (empirical)")

# bar plot (right)
bars = ax_bar.bar(nodes, visit_counts / visit_counts.sum(), color=[cmap(norm(0.0))]*p)
ax_bar.set_ylim(0, max(0.1, (visit_counts / visit_counts.sum()).max()*1.5, pi.max()*1.2))
ax_bar.set_xlabel("Node")
ax_bar.set_ylabel("Probability p_t(a)")
ax_bar.set_title("Empirical visit distribution (bars) → stationary π (dashed)")

# plot stationary as dashed line markers
pi_line = ax_bar.scatter(nodes, pi, marker='D', color='k', s=30, label='π (deg-normalized)')
ax_bar.legend()

# ---- Animation update function ----
def update(frame):
    global current, visit_counts

    # lazy step: with prob 1/2 stay, prob 1/2 move uniformly to neighbor
    if rng.random() < 0.5:
        next_node = current
    else:
        neighs = neighbors[current]
        if len(neighs) == 0:
            next_node = rng.choice(nodes)
        else:
            next_node = rng.choice(neighs)

    # record visit
    visit_counts[next_node] += 1

    # empirical probabilities
    probs = visit_counts / (visit_counts.sum() + 1e-12)

    # update node colors & sizes
    facecolors = [cmap(norm(pv)) for pv in probs]
    # highlight the active/current node with a yellow rim (we also set face to yellow)
    facecolors[next_node] = (1.0, 1.0, 0.0, 1.0)   # yellow
    node_sizes = node_base_size + node_scale * probs
    node_collection.set_facecolor(facecolors)
    node_collection.set_sizes(node_sizes)

    # update bar heights and colors
    for i, b in enumerate(bars):
        b.set_height(probs[i])
        b.set_color(cmap(norm(probs[i])))

    # update y limit just in case (so bars don't get clipped early)
    ymax = max(0.12, probs.max()*1.5, pi.max()*1.2)
    ax_bar.set_ylim(0, ymax)

    # update title to show frame/time
    ax_graph.set_title(f"Paley Graph — step {frame+1}/{n_steps} (current={next_node})")

    current = next_node
    return node_collection, bars


anim = animation.FuncAnimation(fig, update, frames=n_steps, interval=100, blit=False)
anim.save("barprob.gif", writer="pillow", fps=10)
print("GIF saved as barprob.gif")
