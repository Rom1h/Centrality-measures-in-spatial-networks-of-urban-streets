import heapq

import matplotlib
import osmnx.distance

from centralities import straightness, Closeness , betweenness_centrality
from utils import nodes_Size_compute

# matplotlib.use("Agg")  # Headless-safe backend for Docker
matplotlib.use('TkAgg') # For testing on pc
from functools import lru_cache
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
import os
import networkx as nx
from visualisation import get_square_mile_nodes, plot_p_values

# Make output folder
os.makedirs("output", exist_ok=True)

# OSMnx settings
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.cache_folder = "cache"

# Begin Information Centrality



# End Information Centrality

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


"""
Network type optipns can be found in the function's definition
"""
G = ox.graph_from_place("Paris, France", network_type="drive")
G_proj = ox.project_graph(G)
#
# node_0 = list(G_walk_proj.nodes)[0]
# node_2 = list(G_walk_proj.nodes)[150]
#
# paths , total_distance = djikstra_shortest_paths(G_walk_proj, node_0)
# print(f"distance returned = {total_distance} and distance calculated = {sum(paths.values())}")

center_lat, center_lon = ox.geocode("Notre-Dame, Paris, France")
notre_dame_data = get_square_mile_nodes(G, G_proj, center_lat, center_lon, nb_miles=1)
print("*" * 60)
print(notre_dame_data)
print("*" * 60)

G_sub = G_proj.subgraph(notre_dame_data["nodes"]).copy()

print("data = " , notre_dame_data["nodes"])


print("Calculating Closeness Centrality...")
closeness_centrality = Closeness(G_sub)
print(f"Closeness Centrality Sample: {list(closeness_centrality.items())[:5]}")

print("Calculating Straightness Centrality...")
straightness_centrality = straightness(G_sub)
print(f"Straightness Centrality Sample: {list(straightness_centrality.items())[:5]}")



#bc = betweenness_centrality(G_walk_proj)
#print(bc)
# Exemple: Cibler la zone autour de la Tour Eiffel (coordonnées approximatives)
place_name = "Tour Eiffel, Paris, France"
# Obtenir les coordonnées précises du lieu
point = ox.geocode(place_name)

# Définir le rayon en mètres (ex: 2000 mètres, soit 2 km)
distance = 1000

# Créer un sous-graphe dans un rayon de 2000m autour du point
G_area = ox.graph_from_point(
    point,
    dist=distance,
    network_type="drive_service" # Utiliser le même type que G
)

# Important : projeter ce sous-graphe si vous utilisez la projection dans vos calculs
G_sub_proj = ox.project_graph(G_area)

print(f"Taille du sous-graphe de la zone ({distance}m) : {len(G_sub_proj.nodes)} nœuds")
# --- Le calcul est désormais plus précis géographiquement ---
bc_sub_area = betweenness_centrality(G_sub_proj, normalized=True)
print(bc_sub_area)


"""

RANDOM GENERATION TEST

"""
print("*" * 200)
# --- run and plot for multiple p ---------------------------------------------

cities = [
    {"nodes": 100, "nb_miles": 1},
    {"nodes": 150, "nb_miles": 1},
    {"nodes": 80, "nb_miles": 1},
]

grid_size = 50
N = nodes_Size_compute(cities)

p_values = [0, 0.1, 0.2, 1]
plot_p_values(p_values, N, grid_size)