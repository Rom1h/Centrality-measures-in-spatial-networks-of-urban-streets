import networkx as nx
import matplotlib
import random
import numpy as np
from centralities import betweenness_centrality, Closeness

matplotlib.use('TkAgg')  # For testing on PC
import matplotlib.pyplot as plt


def generate_weighted_multigraph(n, max_edges_per_node=5, min_weight=0, max_weight=200):
    """
    Generate a directed multigraph where nodes can have multiple edges between them.
    Edges store 'length' as the attribute name.
    """
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n))

    for node in range(n):
        num_edges = random.randint(1, max_edges_per_node)
        possible_targets = [i for i in range(n) if i != node]

        for _ in range(num_edges):
            target = random.choice(possible_targets)
            length = random.randint(min_weight, max_weight)
            G.add_edge(node, target, length=length)  # <- changed here

    return G


def visualize_multigraph(G, figsize=(14, 10)):
    """Visualize the multigraph with edge labels showing lengths."""
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue',
                           edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    ax = plt.gca()
    for u, v, key, data in G.edges(keys=True, data=True):
        rad = 0.1 + key * 0.15
        arrowprops = dict(
            arrowstyle='->',
            color='gray',
            connectionstyle=f'arc3,rad={rad}',
            linewidth=1.5,
            alpha=0.6
        )
        ax.annotate('', xy=pos[v], xycoords='data',
                    xytext=pos[u], textcoords='data', arrowprops=arrowprops)

    edge_labels = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        if (u, v) not in edge_labels:
            edge_labels[(u, v)] = []
        edge_labels[(u, v)].append(f"{data['length']}")

    for (u, v), lengths in edge_labels.items():
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2
        plt.text(x, y, ', '.join(lengths), fontsize=8, ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.title(f"Weighted Multigraph (n={G.number_of_nodes()}, edges={G.number_of_edges()})",
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def print_graph_info(G):
    """Print statistics about the generated multigraph."""
    print(f"\n{'=' * 60}")
    print(f"MULTIGRAPH STATISTICS")
    print(f"{'=' * 60}")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"\nNode degrees (out-degree):")
    for node in sorted(G.nodes()):
        out_deg = G.out_degree(node)
        print(f"  Node {node}: {out_deg} outgoing edges")

    print(f"\nEdge details:")
    for u, v, key, data in G.edges(keys=True, data=True):
        print(f"  {u} -> {v} (length: {data['length']})")

    print(f"\nMultiple edges between nodes:")
    edge_counts = {}
    for u, v in G.edges():
        pair = (u, v)
        edge_counts[pair] = edge_counts.get(pair, 0) + 1

    multi_edges = {k: v for k, v in edge_counts.items() if v > 1}
    if multi_edges:
        for (u, v), count in multi_edges.items():
            edge_data = G.get_edge_data(u, v)
            lengths = [d['length'] for d in edge_data.values()]
            print(f"  {u} -> {v}: {count} edges with lengths {lengths}")
    else:
        print("  None")

    print(f"{'=' * 60}\n")


def visualize_centrality(G, centrality, figsize=(14, 10), cmap=plt.cm.viridis, label="Betweenness Centrality"):
    """Visualize a multigraph with node colors based on centrality values."""
    plt.figure(figsize=figsize)
    ax = plt.gca()
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    values = np.array(list(centrality.values()))
    norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    node_colors = [cmap(norm(centrality[n])) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colors,
                           edgecolors='black', linewidths=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    for u, v, key, data in G.edges(keys=True, data=True):
        rad = 0.1 + key * 0.15
        arrowprops = dict(
            arrowstyle='->',
            color='gray',
            connectionstyle=f'arc3,rad={rad}',
            linewidth=1.5,
            alpha=0.6
        )
        ax.annotate('', xy=pos[v], xycoords='data',
                    xytext=pos[u], textcoords='data', arrowprops=arrowprops)

    edge_labels = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        if (u, v) not in edge_labels:
            edge_labels[(u, v)] = []
        edge_labels[(u, v)].append(f"{data['length']}")
    for (u, v), lengths in edge_labels.items():
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2
        ax.text(x, y, ', '.join(lengths), fontsize=8, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.title(f"Weighted Multigraph Colored by {label}", fontsize=14, fontweight='bold')
    plt.axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=label)

    plt.tight_layout()
    plt.show()


def visualize_closeness(G, closeness, figsize=(14, 10), cmap=plt.cm.plasma):
    """Visualize a multigraph with node colors based on closeness centrality."""
    visualize_centrality(G, closeness, figsize=figsize, cmap=cmap, label="Closeness Centrality")


# ---------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------
if __name__ == "__main__":
    random.seed(12)
    N = 40

    print(f"Generating multigraph with {N} nodes...")
    G = generate_weighted_multigraph(n=N, max_edges_per_node=1, min_weight=0, max_weight=200)

    print_graph_info(G)

    # Betweenness centrality
    CB = betweenness_centrality(G, weight="length", normalized=True)
    print("Betweenness centrality values:", CB)
    visualize_centrality(G, CB, label="Betweenness Centrality")

    # Closeness centrality
    CC = Closeness(G)
    print("Closeness centrality values:", CC)
    visualize_closeness(G, CC)
