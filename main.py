import heapq

import matplotlib
matplotlib.use("Agg")  # Headless-safe backend
from functools import lru_cache
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
import os

# Make output folder
os.makedirs("output", exist_ok=True)

# OSMnx settings
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.cache_folder = "cache"


# Download graph
G = ox.graph_from_place("Paris, France", network_type="drive")

# Convert to GeoDataFrames
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True)

# Optional: save a plot of the graph
fig, ax = ox.plot_graph(G, show=False, close=False)  # headless-friendly
fig.savefig("output/graph.png", dpi=300)
plt.close(fig)
import osmnx as ox


def djikstra_shortest_paths(G, source, weight="length"):
    """
    Djikrstra's algorithm to compute shortest paths from a source node to all other nodes in the graph.
    """

    distances = {source: 0}
    pq = [(0, source)]
    visited = set()

    while pq:
        concurrent_distance, current_node = heapq.heappop(pq)
        if current_node in visited:
            continue
        visited.add(current_node)
        for neighbor in G.neighbors(current_node):
            edge_weight = G[current_node][neighbor][0][weight]
            distance = concurrent_distance + edge_weight

            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances

#prototype , really slow for large graphs
def Closeness(G, node):
    """
    Compute closeness centrality of a single node.
    """
    N = len(G.nodes)
    distances = djikstra_shortest_paths(G, node)
    total_distance = sum(distances.values())
    if total_distance > 0 and N > 1:
        closeness = (N - 1) / total_distance
    else:
        closeness = 0
    return closeness

print("------------------- Closeness Centrality ------------------")
print("nodes in graph:", len(G.nodes))
print("nodes : " , list(G.nodes)[0])
node_0 = list(G.nodes)[0]  # Example: first node in the graph
node_3 = list(G.nodes)[3]
print("node 0:", node_0)
print("Node 0 neighbors , " , list(G.neighbors(node_0)))
print("Closeness of node 0:", Closeness(G, node_0))