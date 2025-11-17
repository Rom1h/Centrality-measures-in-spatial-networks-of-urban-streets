import heapq
import math
from heapq import heappush, heappop
from shapely.geometry import box, Point
import osmnx as ox
import numpy as np
def information_centrality(G):
    inf_centr = {}
    for i in G.nodes:
        N = len(G.nodes)

        G_prime = G.copy()

        G_prime.remove_node(i)

        ef = efficiency(G, N)
        ef_prime = efficiency(G_prime, N)

        inf_centr[i] = (ef - ef_prime) / ef

    return inf_centr

def sum_of_distance(G):
    djik_dist = {}
    sum_res = 0
    for i in G.nodes:
        djik_dist,_ = djikstra_shortest_paths(G, i)
        for j in djik_dist :
            if j != i :
                res_eucl = euclidean_distance(G.nodes[i]["x"], G.nodes[i]["y"], G.nodes[j]["x"], G.nodes[j]["y"])
                sum_res += res_eucl/djik_dist[j]
    return sum_res

def efficiency(G, N):
    sum_res = sum_of_distance(G)
    return sum_res/(N *(N - 1))

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


def get_square_mile_nodes(G, G_proj, center_lat, center_lon, nb_miles=1):
    """Get all nodes within a square area around a center point."""
    mile = 1609.34
    size = nb_miles * mile
    half_side = size / 2

    center_node = ox.distance.nearest_nodes(G, center_lon, center_lat)
    center_x = G_proj.nodes[center_node]['x']
    center_y = G_proj.nodes[center_node]['y']

    square = box(center_x - half_side, center_y - half_side,
                 center_x + half_side, center_y + half_side)

    nodes = [
        n for n, data in G_proj.nodes(data=True)
        if square.contains(Point(data['x'], data['y']))
    ]

    return {
        'nodes': nodes,
        'center_x': center_x,
        'center_y': center_y,
        'square': square,
        'half_side': half_side,
        'center_lat': center_lat,
        'center_lon': center_lon,
        'nb_miles': nb_miles
    }


def node_density(square_data, grid_size):
    """
    Compute node density in meters² per node
    for now nb miles always 1
    """
    nodes = square_data['nodes']
    density = nodes / (grid_size * grid_size)
    return density

def median_calculator(array):
    array = sorted(array)
    n = len(array)
    if n % 2 == 0:
        return (array[(n - 1) // 2] + array[n // 2]) / 2
    else:
        return array[n // 2]

def nodes_Size_compute(cities):
    """
    we define a default grid layout of m * m for one square mile
    We calculate the median_density from our cities

    N = median_density * total_cellsand
    """
    grid_size = 50
    density_values = [node_density(city, grid_size) for city in cities]
    print(density_values)
    median_density = median_calculator(density_values)
    print(median_density)
    N = median_density * (grid_size * grid_size)

    return int(N)


def building_network_with_p(N, grid_size, p):
    """
    Build a grid network of size grid_size x grid_size
    and connect nodes with probability p

    puts nodes in spacing * 1 , spacing *2 ... , spacing * n
    """
    print(f"\nBuilding network: N={N}, grid={grid_size}×{grid_size}, p={p}")

    # Step 1: Place N nodes on initial grid pattern
    nodes_per_side = int(math.sqrt(N))  # e.g., √100 = 10
    spacing = grid_size / nodes_per_side
    nodes = []
    for i in range(nodes_per_side):
        for j in range(nodes_per_side):
            x = int(i * spacing)
            y = int(j * spacing)
            nodes.append([x, y])

    nodes = np.array(nodes)
    moved_count = 0
    for i in range(N):
        if np.random.random() < p:
            # THIS node moves to random position
            nodes[i] = [
                np.random.randint(0, grid_size),
                np.random.randint(0, grid_size)
            ]
            moved_count += 1

    # Step 3: Connect each node to 2 nearest neighbors
    edges = []
    connected = set()  # Track which edges we've added

    # O(N^2) approach
    for i in range(N):
        # Compare with ALL other nodes
        distances = []

        for j in range(N):
            if i != j:
                # Calculate distance
                dist = euclidean_distance(
                    nodes[i][1], nodes[i][0],
                    nodes[j][1], nodes[j][0]
                )
                distances.append((dist, j))

        # Sort by distance
        distances.sort()

        # Connect to 2 closest
        edges_added = 0
        for dist, j in distances:
            # Check if edge already exists (avoid duplicates)
            edge_id = tuple(sorted([i, j]))  # (smaller, larger)

            if edge_id not in connected:
                edges.append((i, j, dist))
                connected.add(edge_id)
                edges_added += 1

                if edges_added >= 2:
                    break

    print(f"  Edges: {len(edges)}, Avg degree: {2 * len(edges) / N:.2f}")

    return nodes, edges




