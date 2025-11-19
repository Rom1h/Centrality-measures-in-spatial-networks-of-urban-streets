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
from visualisation import get_square_mile_nodes, plot_p_values,plot_cumulative_distribution_centrality

# -------------------------------
# SELECT LOCATION IN RICHMOND
# -------------------------------
place_name = "Carytown, Richmond, Virginia, USA"

# Load full drive network for Richmond
G = ox.graph_from_place("Richmond, Virginia, USA", network_type="drive")
G_proj = ox.project_graph(G)

# Geocode center of chosen place
center_lat, center_lon = ox.geocode(place_name)

# Extract 1-square-mile subgraph around the place
sub_data = get_square_mile_nodes(G, G_proj, center_lat, center_lon, nb_miles=1)
print("*" * 60)
print(sub_data)
print("*" * 60)

G_sub = G_proj.subgraph(sub_data["nodes"]).copy()
print("Selected nodes:", sub_data["nodes"])

# ---------------------------------------------------------
# CENTRALITY COMPUTATIONS
# ---------------------------------------------------------

print("Calculating Betweenness Centrality...")
bc_sub_area = betweenness_centrality(G_sub, normalized=True)
print(f"Taille du sous-graphe de la zone: {len(G_sub.nodes)} noeuds")

bw_values = list(bc_sub_area.values())

plot_cumulative_distribution_centrality(
    values=bw_values,
    title="Cumulative distribution of betweenness – Shockoe Bottom, Richmond (1 mile)",
    output_file="betweenness_shockoe_bottom_richmond_v1.png",
    model="gauss",
)
