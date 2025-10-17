import heapq

import matplotlib
import osmnx.distance

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
"""
Network type optipns can be found in the function's definition
"""
G = ox.graph_from_place("Paris, France", network_type="drive")

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


def euclidean_distance(y1,x1, y2,x2):
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5
def straightness_helper(G_proj, node, node2):
    """
    d(Euclidean) i j / d i j

    IMPORTANT ---> Using djikstra to all paths here is a waste , think bout optimisiting it
    G has to be projected
    """
    # Node coordinates (meters)
    x1, y1 = G_proj.nodes[node]["x"], G_proj.nodes[node]["y"]
    x2, y2 = G_proj.nodes[node2]["x"], G_proj.nodes[node2]["y"]

    # Euclidean distance in meters
    euclidean_m = euclidean_distance(y1,x1,y2,x2)

    # Shortest path distance in meters
    paths = djikstra_shortest_paths(G, node)
    shortest_distance = paths[node2]

    if shortest_distance == 0:
        return 0  # avoid division by zero

    print(f"distance = {shortest_distance}")
    print(f"euclidean = {euclidean_m}")
    straightness = euclidean_m / shortest_distance
    return straightness


# def straightness(G,node):


# print("------------------- Closeness Centrality ------------------")
# print("nodes in graph:", len(G.nodes))
# print("nodes : " , list(G.nodes)[0])
node_0 = list(G.nodes)[0]  # Example: first node in the graph
node_3 = list(G.nodes)[150]
print("node 0:", node_0)
# print("Node 0 neighbors , " , list(G.neighbors(node_0)))
# print("Closeness of node 0:", Closeness(G, node_0))


G_walk_proj = ox.project_graph(G)

node_0 = list(G_walk_proj.nodes)[0]
node_3 = list(G_walk_proj.nodes)[150]

s = straightness_helper(G_walk_proj, node_0, node_3)
print(f"Straightness from node 0 to node 3: {s:.3f}")