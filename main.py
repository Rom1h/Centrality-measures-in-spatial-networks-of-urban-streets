import heapq

import matplotlib
import osmnx.distance

matplotlib.use("Agg")  # Headless-safe backend
from functools import lru_cache
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
import os
import networkx as nx
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

print(f"type : {type(G)}")

def table_info(G):
    # Convert the NetworkX graph G into GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G)

    # --- NODE INFORMATION ---
    print("--- Nodes DataFrame Information ---")
    print("---- Nodes DataFrame columns")
    print(nodes.columns)

    print("-----------Nodes type---------------")
    # The .info() method prints directly and returns None, so we print it separately
    nodes.info()
    print("--       --- First 5 Lines -----------")
    print(nodes.head(5))
    print(f"\nNodes DataFrame Shape: {nodes.shape}")

    # --- EDGE INFORMATION (Added details here) ---
    print("\n" + "="*50) # Separator
    print("--- Edges DataFrame Information ---")
    print("---- Edges DataFrame columns")
    print(edges.columns)
    print(edges["length"])
    print("-----------Edges type---------------")
    # Detailed summary of the Edges DataFrame
    edges.info()
    print("----- First 5 Lines -----------")
    # Prints the first 5 rows of the Edges DataFrame
    print(edges.head(5))
    print(f"\nEdges DataFrame Shape: {edges.shape}")
    print("="*50)
def djikstra_shortest_paths(G, source, weight="length"):
    """
    Dijkstra's algorithm to compute shortest paths from a source node to all other nodes in the graph.
    Returns both the distances dict and the total sum of distances.

    Returns total distance so that we can avoid doing another O(n) in Closeness calculation.
    """
    distances = {source: 0}
    pq = [(0, source)]
    visited = set()
    total_distance = 0  # accumulate distances here

    while pq:
        concurrent_distance, current_node = heapq.heappop(pq)
        if current_node in visited:
            continue
        visited.add(current_node)
        total_distance += concurrent_distance  # accumulate as we finalize a node

        for neighbor in G.neighbors(current_node):
            edge_weight = G[current_node][neighbor][0][weight]
            distance = concurrent_distance + edge_weight

            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances, total_distance


def euclidean_distance(y1,x1, y2,x2):
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5
def straightness(G_proj):
    """
    Uses the djikstra_shortest_paths to compute the straightness centrality for all nodes in the graph.
    Uses Euclidean to calculate the straight distance between two nodes

    Complexity should be O(N * (E + N) log N)
    Where N is the number of nodes and E the number of edges

    """
    n = len(G.nodes)
    straightness = {}
    for node in G_proj.nodes: # O(n)

        paths, _ = djikstra_shortest_paths(G_proj, node) # O(n) at most
        x1 , y1 = G_proj.nodes[node]["x"], G_proj.nodes[node]["y"]

        total = 0
        for j, shortest_distance in paths.items(): # O(n-1) at most , lets call it O(m)
            if node == j or shortest_distance == 0:
                continue
            x2, y2 = G_proj.nodes[j]["x"], G_proj.nodes[j]["y"]
            euclidean_m = euclidean_distance(y1,x1,y2,x2)
            total += euclidean_m / shortest_distance

        straightness[node] = total / (n - 1)
    return straightness

def Closeness(G):
    """
    Compute closeness centrality of a single node.
    """
    n = len(G.nodes)
    if n <= 1:
        return 0.0
    closeness = {}
    for node in G.nodes:
        distances, total_distance = djikstra_shortest_paths(G, node)
        if total_distance > 0:
            closeness[node] = (n-1)/total_distance
        else:
            closeness[node] = 0.0
    return closeness

# G_walk_proj = ox.project_graph(G)
#
# node_0 = list(G_walk_proj.nodes)[0]
# node_2 = list(G_walk_proj.nodes)[150]
#
# paths , total_distance = djikstra_shortest_paths(G_walk_proj, node_0)
# print(f"distance returned = {total_distance} and distance calculated = {sum(paths.values())}")
#
# sub_nodes = list(G_walk_proj.nodes)[:5000]
# G_sub = G_walk_proj.subgraph(sub_nodes).copy()
#
# print("Calculating Closeness Centrality...")
# closeness_centrality = Closeness(G_sub)
# print(f"Closeness Centrality Sample: {list(closeness_centrality.items())[:5]}")
#
# print("Calculating Straightness Centrality...")
# straightness_centrality = straightness(G_sub)
# print(f"Straightness Centrality Sample: {list(straightness_centrality.items())[:5]}")

print("Data info")
table_info(G)