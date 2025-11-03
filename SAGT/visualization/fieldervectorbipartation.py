#Assitance from https://dyeun.wordpress.ncsu.edu/files/2022/08/Sigmetrics20-FV-1.pdf and https://stackoverflow.com/questions/10924966/https://stackoverflow.com/questions/10924966/computing-the-fiedler-vector-in-pythoncomputing-the-fiedler-vector-in-python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def calculate_laplacian(A):
    degree_matrix = np.diag(A.sum(axis=1))
    L = degree_matrix - A
    return L

def computefiedler(L):
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    lambda2 = eigenvalues[1]
    fiedler_pos = np.where(eigenvalues.real == np.sort(eigenvalues.real)[1])[0][0]
    fiedler_vector = np.transpose(eigenvectors)[fiedler_pos]
    return lambda2, fiedler_vector

def bipartite(fiedler_vector):
    median = np.median(fiedler_vector)
    group1 = np.where(fiedler_vector <= median)[0]
    group2 = np.where(fiedler_vector > median)[0]
    
    return group1, group2

def visualize(n, A, group1, group2, steps=500, k=0.6, repulsion=0.05, lr=0.01):
    """
    n          : number of vertices
    A          : adjacency matrix
    group1/2   : node indices for bipartitioning
    steps      : number of layout iterations
    k          : spring constant (edge attraction)
    repulsion  : repulsion factor between all nodes
    lr         : learning rate (step size for updates)
    """
    np.random.seed(0)
    pos = np.random.randn(n, 2)  # random starting positions in 2D space

    # Ensure binary adjacency matrix
    A = (A > 0).astype(float)

    for _ in range(steps):
        forces = np.zeros_like(pos)

        # --- Repulsion (Coulomb-like) ---
        for i in range(n):
            diff = pos[i] - pos              # vector from others to node i
            dist2 = np.sum(diff**2, axis=1) + 1e-4  # squared distances
            repulse = repulsion * diff / dist2[:, None]
            forces[i] += np.sum(repulse, axis=0)    # sum of all repulsion forces

        # --- Attraction (Spring-like) ---
        for i in range(n):
            neighbors = np.where(A[i] > 0)[0]
            for j in neighbors:
                diff = pos[i] - pos[j]
                forces[i] -= k * diff  # pull connected nodes closer

        # --- Update positions ---
        pos += lr * forces
        pos -= np.mean(pos, axis=0)  # center graph each step

    # --- Plot graph ---
    plt.figure(figsize=(9, 6))
    plt.title("Spectral Bipartitioning (Fiedler Vector-Based)", fontsize=13)

    # Draw edges (light gray lines)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                plt.plot([pos[i, 0], pos[j, 0]],
                         [pos[i, 1], pos[j, 1]],
                         color='gray', alpha=0.4, linewidth=0.8)

    # Draw nodes by partition color
    plt.scatter(pos[group1, 0], pos[group1, 1],
                c='lightblue', s=150, edgecolors='k', label='Group 1 (low Fiedler values)')
    plt.scatter(pos[group2, 0], pos[group2, 1],
                c='orange', s=150, edgecolors='k', label='Group 2 (high Fiedler values)')

    # Label each node by its index
    for i in range(n):
        plt.text(pos[i, 0], pos[i, 1] + 0.15, str(i),
                 ha='center', va='center', fontsize=7)

    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()

def visualize_animated(n, A, group1, group2, steps=200, spring_strength=0.6, repulsion_strength=0.05, learning_rate=0.01):
    np.random.seed(0)
    pos = np.random.randn(n, 2)
    A = (A > 0).astype(float)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Spectral Bipartitioning (Animated Layout)", fontsize=14)
    ax.axis('off')
    ax.set_aspect('equal')

    # --- Initialize edges ---
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                line, = ax.plot([], [], color='gray', alpha=0.3, linewidth=0.8)
                edges.append((i, j, line))

    # --- Initialize nodes and labels ---
    scat1 = ax.scatter([], [], c='lightblue', s=150, edgecolors='k', label='Group 1')
    scat2 = ax.scatter([], [], c='orange', s=150, edgecolors='k', label='Group 2')
    labels = [ax.text(0, 0, str(i), ha='center', va='center', fontsize=7) for i in range(n)]
    ax.legend()

    # --- Pause control ---
    paused = False
    def on_click(event):
        nonlocal paused
        paused = not paused

    fig.canvas.mpl_connect('key_press_event', lambda event: on_click(event) if event.key == ' ' else None)

    def update_positions():
        nonlocal pos
        forces = np.zeros_like(pos)

        for i in range(n):
            diff = pos[i] - pos
            dist2 = np.sum(diff**2, axis=1) + 1e-4
            repulse = repulsion_strength * diff / dist2[:, None]
            forces[i] += np.sum(repulse, axis=0)

        for i in range(n):
            neighbors = np.where(A[i] > 0)[0]
            for j in neighbors:
                diff = pos[i] - pos[j]
                forces[i] -= spring_strength * diff

        pos += learning_rate * forces
        pos -= np.mean(pos, axis=0)
        max_range = np.max(np.linalg.norm(pos, axis=1))
        if max_range > 0:
            pos /= max_range

    def animate(frame):
        if not paused:
            update_positions()

        for (i, j, line) in edges:
            line.set_data([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]])

        scat1.set_offsets(pos[group1])
        scat2.set_offsets(pos[group2])

        for i in range(n):
            labels[i].set_position((pos[i, 0], pos[i, 1] + 0.08))

        margin = 0.3
        x_min, x_max = pos[:, 0].min() - margin, pos[:, 0].max() + margin
        y_min, y_max = pos[:, 1].min() - margin, pos[:, 1].max() + margin
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        return [line for (_, _, line) in edges] + [scat1, scat2] + labels

    ani = animation.FuncAnimation(fig, animate, frames=steps, interval=50, blit=True, repeat=False)
    plt.show()


def main():
    #Test
    A = np.array([
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0]
    ], dtype=float)

    B = np.array([
    # Cluster 1 (0–8)
    [0,1,1,1,0,0,1,0,0,  1,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
    [1,0,1,0,1,0,1,0,0,  1,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
    [1,1,0,1,0,1,1,0,0,  0,1,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
    [1,0,1,0,1,0,1,0,0,  0,0,1,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
    [0,1,0,1,0,1,1,1,0,  0,0,0,1,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
    [0,0,1,0,1,0,1,1,0,  0,0,0,0,1,0,0,0,0,  0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,0,1,0,  0,0,0,0,0,1,0,0,0,  0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,0,1,  0,0,0,0,0,0,1,0,0,  0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,  0,0,0,0,0,0,0,1,0,  0,0,0,0,0,0,0,0,0],

    # Cluster 2 (9–17)
    [1,1,0,0,0,0,0,0,0,  0,1,1,1,0,0,1,0,0,  1,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,  1,0,1,0,1,0,1,0,0,  0,1,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,  1,1,0,1,0,1,1,0,0,  0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,  1,0,1,0,1,0,1,1,0,  0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,  0,1,0,1,0,1,1,0,1,  0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,  0,0,1,0,1,0,1,0,1,  0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,1,0,  1,1,1,1,1,1,0,1,0,  0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,1,  0,0,0,1,0,0,1,0,1,  0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,  0,0,0,0,1,0,0,1,0,  0,0,0,0,0,0,0,0,1],

    # Cluster 3 (18–26)
    [0,0,0,0,0,0,0,0,0,  1,0,0,0,0,0,0,0,0,  0,1,1,1,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,  0,1,0,0,0,0,0,0,0,  1,0,1,0,1,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,  0,0,1,0,0,0,0,0,0,  1,1,0,1,0,1,0,1,0],
    [0,0,0,0,0,0,0,0,0,  0,0,0,1,0,0,0,0,0,  1,0,1,0,1,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,  0,0,0,0,1,0,0,0,0,  0,1,0,1,0,1,0,0,1],
    [0,0,0,0,0,0,0,0,0,  0,0,0,0,0,1,0,0,0,  0,0,1,0,1,0,1,0,1],
    [0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,1,0,0,  0,0,0,1,0,1,0,1,1],
    [0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,1,0,  1,1,0,0,1,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,1,  0,0,0,1,0,1,1,0,0]
    ], dtype=float)

    n = A.shape[0]
    L = calculate_laplacian(A)
    lambda2, fiedler_vector = computefiedler(L)
    print("Algebraic connectivity (λ2):", round(lambda2, 4))
    print("Fiedler vector:", np.round(fiedler_vector, 4))
    group1, group2 = bipartite(fiedler_vector)
    visualize(n, A, group1, group2)

    n = B.shape[0]
    L = calculate_laplacian(B)
    lambda2, fiedler_vector = computefiedler(L)
    print("Algebraic connectivity (λ2):", round(lambda2, 4))
    print("Fiedler vector:", np.round(fiedler_vector, 4))

    group1, group2 = bipartite(fiedler_vector)

    visualize_animated(n, B, group1, group2)

if __name__ == "__main__":
    main()