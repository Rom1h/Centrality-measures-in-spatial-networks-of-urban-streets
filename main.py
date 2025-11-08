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
from visualisation import get_square_mile_nodes
# Make output folder
os.makedirs("output", exist_ok=True)

# OSMnx settings
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.cache_folder = "cache"

# Begin Information Centrality

def sum_of_distance(G):
    djik_dist = {}
    sum_res = 0
    for i in G.nodes: 
        djik_dist = djikstra_shortest_paths(G, i)
        for j in G.nodes:
            if j != i and j in djik_dist :
                res_eucl = euclidean_distance(G.nodes[i]["x"], G.nodes[i]["y"], G.nodes[j]["x"], G.nodes[j]["y"])
                sum_res += res_eucl/djik_dist[j]
    return sum_res

def efficiency(G, N): 
    sum_res = sum_of_distance(G)
    return sum_res/(N *(N - 1))

def information_centrality(G):
    inf_centr = {}
    for i in G.nodes :
        N = len(G.nodes)

        G_prime = G.copy()
        
        G_prime.remove_node(i)

        ef = efficiency(G, N)
        ef_prime = efficiency(G_prime, N)

        inf_centr[i] = (ef-ef_prime)/ef
    
    return inf_centr

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

G_sub = G_proj.subgraph(notre_dame_data["nodes"]).copy()

print("data = " , notre_dame_data["nodes"])


print("Calculating Closeness Centrality...")
closeness_centrality = Closeness(G_sub)
print(f"Closeness Centrality Sample: {list(closeness_centrality.items())[:5]}")

print("Calculating Straightness Centrality...")
straightness_centrality = straightness(G_sub)
print(f"Straightness Centrality Sample: {list(straightness_centrality.items())[:5]}")






from heapq import heappush, heappop

def _edge_weight(G, u, v, weight="length"):
    """
    Renvoie le poids (longueur) de l'arête u->v.
    - Pour Multi(Di)Graph : prend le MIN des parallel edges.
    - Pour (Di)Graph simple : lit directement l'attribut.
    - Si l'attribut n'existe pas : poids = 1.0
    """
    data = G.get_edge_data(u, v, default=None)
    if data is None:
        return float("inf")
    if G.is_multigraph():
        # data : dict {key: {attr...}}
        best = float("inf")
        for _, attrs in data.items():
            w = attrs.get(weight, 1.0)
            if w < best:
                best = w
        return best
    else:
        # data : dict {attr...}
        return data.get(weight, 1.0)

def shortest_dijkstra(G, s, weight="length"):
    """
    Phase 'avant' de Brandes (pondéré) :
    - distances métriques d[s->v]
    - sigma[v] : nb de plus courts chemins (pondérés)
    - P[v] : prédécesseurs sur au moins un plus court chemin
    - S : ordre non décroissant des distances (pile pour back-prop)
    """
    # Initialisation
    S = []
    P = {v: [] for v in G.nodes}
    sigma = dict.fromkeys(G.nodes, 0.0)
    dist = dict.fromkeys(G.nodes, float("inf"))
    sigma[s] = 1.0
    dist[s] = 0.0

    # File de priorité (dijkstra)
    Q = []
    heappush(Q, (0.0, s))

    while Q:
        dv, v = heappop(Q)
        if dv != dist[v]:
            continue  # entrée obsolète
        S.append(v)

        # Pour DiGraph : voisins = successeurs ; pour Graph : voisins classiques
        for w in G.neighbors(v):
            vw = _edge_weight(G, v, w, weight=weight) # distance entre vw
            if vw == float("inf"):
                continue

            alt = dist[v] + vw

            # On a trouvé un chemin plus court ?
            if alt < dist[w]:
                dist[w] = alt
                heappush(Q, (alt, w))
                sigma[w] = sigma[v]  # on remplace le nombre de chemins
                P[w] = [v]           # et on réinitialise les prédécesseurs
            # On a trouvé un chemin de même longueur (un autre plus court chemin)
            elif alt == dist[w]:
                sigma[w] += sigma[v]
                P[w].append(v)

    return S, P, sigma, dist

def general_contrib(S, P, sigma):
    """
    Phase 'arrière' de Brandes (pondéré ou non) :
    propage les dépendances delta du plus loin vers la source.
    """
    delta = {v: 0.0 for v in S}
    S.reverse()
    for w in S:
        for v in P[w]:
            if sigma[w] != 0:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
    return delta

def betweenness_centrality(G, weight="length", normalized=True):
    """
    Brandes pondéré (Dijkstra).
    - weight: nom de l'attribut d'arête (OSMnx: "length")
    - normalized:
        * orienté   : divise par (n-1)(n-2)
        * non orienté: divise par ((n-1)(n-2)/2)
    """

    # Vérif (évite de planter sur DiGraph/MultiDiGraph)
    if not hasattr(G, "nodes") or not hasattr(G, "neighbors"):
        raise TypeError("G doit ressembler à un graphe NetworkX (nodes, neighbors, ...).")

    CB = {v: 0.0 for v in G.nodes}
    n = G.number_of_nodes()
    if n < 3:
        return CB

    for s in G.nodes:
        S, P, sigma, dist = shortest_dijkstra(G, s, weight=weight)
        delta = general_contrib(S, P, sigma)
        for v in delta:
            if v != s:
                CB[v] += delta[v]

    # Normalisation correcte
    if normalized and n > 2:
        if G.is_directed():
            norm = (n - 1) * (n - 2)
        else:
            norm = (n - 1) * (n - 2) / 2.0
        if norm > 0:
            for v in CB:
                CB[v] /= norm

    return CB


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

