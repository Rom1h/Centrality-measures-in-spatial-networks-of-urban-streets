import networkx as nx
import matplotlib

from centralities import betweenness_centrality

matplotlib.use('TkAgg') # For testing on pc
import matplotlib.pyplot as plt
import random
import numpy as np


def generate_weighted_multigraph(n, max_edges_per_node=5, min_weight=0, max_weight=200):
    """
    Generate a directed multigraph where nodes can have multiple edges between them.

    Args:
        n: Number of nodes
        max_edges_per_node: Maximum number of outgoing edges per node (default: 5)
        min_weight: Minimum edge weight/distance (default: 0)
        max_weight: Maximum edge weight/distance (default: 200)

    Returns:
        NetworkX MultiDiGraph
    """
    # Create directed multigraph
    G = nx.MultiDiGraph()

    # Add nodes
    G.add_nodes_from(range(n))

    # For each node, create random edges
    for node in range(n):
        # Randomly decide how many edges this node will have (1 to max_edges_per_node)
        num_edges = random.randint(1, max_edges_per_node)

        # Get all possible target nodes (excluding self)
        possible_targets = [i for i in range(n) if i != node]

        # Randomly select target nodes (with replacement, allowing multiple edges to same node)
        for _ in range(num_edges):
            target = random.choice(possible_targets)
            weight = random.randint(min_weight, max_weight)
            G.add_edge(node, target, weight=weight)

    return G


def visualize_multigraph(G, figsize=(14, 10)):
    """
    Visualize the multigraph with edge labels showing weights.
    """
    plt.figure(figsize=figsize)

    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue',
                           edgecolors='black', linewidths=2)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # Draw edges with curved connections
    ax = plt.gca()
    for u, v, key, data in G.edges(keys=True, data=True):
        # Calculate curve for multiple edges between same nodes
        rad = 0.1 + key * 0.15  # Increase curve for each additional edge

        arrowprops = dict(
            arrowstyle='->',
            color='gray',
            connectionstyle=f'arc3,rad={rad}',
            linewidth=1.5,
            alpha=0.6
        )

        ax.annotate('',
                    xy=pos[v], xycoords='data',
                    xytext=pos[u], textcoords='data',
                    arrowprops=arrowprops)

    # Add edge weight labels
    edge_labels = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        if (u, v) not in edge_labels:
            edge_labels[(u, v)] = []
        edge_labels[(u, v)].append(f"{data['weight']}")

    # Draw edge labels (combine multiple weights)
    for (u, v), weights in edge_labels.items():
        # Calculate midpoint with slight offset for multiple edges
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2

        label = ', '.join(weights)
        plt.text(x, y, label, fontsize=8, ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.title(f"Weighted Multigraph (n={G.number_of_nodes()}, edges={G.number_of_edges()})",
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def print_graph_info(G):
    """
    Print statistics about the generated multigraph.
    """
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
        print(f"  {u} -> {v} (weight: {data['weight']})")

    # Check for multiple edges between same node pairs
    print(f"\nMultiple edges between nodes:")
    edge_counts = {}
    for u, v in G.edges():
        pair = (u, v)
        edge_counts[pair] = edge_counts.get(pair, 0) + 1

    multi_edges = {k: v for k, v in edge_counts.items() if v > 1}
    if multi_edges:
        for (u, v), count in multi_edges.items():
            edge_data = G.get_edge_data(u, v)  # dict of key -> attributes
            weights = [d['weight'] for d in edge_data.values()]
            print(f"  {u} -> {v}: {count} edges with weights {weights}")
    else:
        print("  None")

    print(f"{'=' * 60}\n")


def visualize_centrality(G, centrality, figsize=(14, 10), cmap=plt.cm.viridis):
    """
    Visualize a multigraph with node colors based on centrality values.

    Args:
        G: NetworkX MultiDiGraph
        centrality: dict of node -> centrality value
        figsize: figure size
        cmap: matplotlib colormap
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()  # Get current axes

    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Normalize centrality values for coloring
    values = np.array(list(centrality.values()))
    norm = plt.Normalize(vmin=values.min(), vmax=values.max())
    node_colors = [cmap(norm(centrality[n])) for n in G.nodes()]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colors,
                           edgecolors='black', linewidths=2, ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    # Draw edges with curved connections
    for u, v, key, data in G.edges(keys=True, data=True):
        rad = 0.1 + key * 0.15
        arrowprops = dict(
            arrowstyle='->',
            color='gray',
            connectionstyle=f'arc3,rad={rad}',
            linewidth=1.5,
            alpha=0.6
        )
        ax.annotate('',
                    xy=pos[v], xycoords='data',
                    xytext=pos[u], textcoords='data',
                    arrowprops=arrowprops)

    # Optional: add edge weights
    edge_labels = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        if (u, v) not in edge_labels:
            edge_labels[(u, v)] = []
        edge_labels[(u, v)].append(f"{data['weight']}")
    for (u, v), weights in edge_labels.items():
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2
        ax.text(x, y, ','.join(weights), fontsize=8, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Add title and remove axes
    plt.title("Weighted Multigraph Colored by Betweenness Centrality", fontsize=14, fontweight='bold')
    plt.axis('off')

    # Add colorbar correctly
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # needed for ScalarMappable
    plt.colorbar(sm, ax=ax, label="Betweenness Centrality")

    plt.tight_layout()
    plt.show()



# ---------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(12)

    # Generate graph with N nodes
    N = 35  # You can change this value

    print(f"Generating multigraph with {N} nodes...")
    G = generate_weighted_multigraph(
        n=N,
        max_edges_per_node=5,
        min_weight=0,
        max_weight=200
    )

    # Print information
    print_graph_info(G)

    # Compute betweenness centrality using your function
    CB = betweenness_centrality(G, weight="weight", normalized=True)
    print("Betweenness centrality values:", CB)

    # Visualize
    visualize_centrality(G, CB)