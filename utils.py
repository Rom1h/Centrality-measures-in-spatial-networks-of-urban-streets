import heapq
import math
from heapq import heappush, heappop

import networkx as nx
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

    distances = {node: float('inf') for node in G.nodes}
    distances[source] = 0

    pq = [(0, source)]
    visited = set()

    while pq:
        dist_u, u = heapq.heappop(pq)

        if u in visited:
            continue

        visited.add(u)

        # Parcourir voisins
        for v in G.neighbors(u):
            edge_w = _edge_weight(G, u, v, weight=weight)
            alt = dist_u + edge_w

            if alt < distances[v]:
                distances[v] = alt
                heapq.heappush(pq, (alt, v))

    # total_distance = somme des distances finies
    total_distance = sum(d for d in distances.values() if d < float('inf'))

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



"""
            ****************BEGINNING OF RANDOM GENERATED GRAPHS FUNCTIONS****************-         
"""

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

def add_edge_lengths_to_graph(G, nodes, edges, grid_size=50, mile_in_meters=1609):
    """
    Add 'length' attribute (in meters) to each edge in the graph G.

    nodes: Nx2 array of node coordinates (grid units)
    edges: list of tuples (i, j, _) from building_network_with_p
    """
    unit_length = mile_in_meters / grid_size  # meters per grid unit

    for i, j, _ in edges:
        dx = (nodes[i][0] - nodes[j][0]) * unit_length
        dy = (nodes[i][1] - nodes[j][1]) * unit_length
        dist_meters = math.sqrt(dx**2 + dy**2)
        # Add distance to the graph as 'length'
        G[i][j]['length'] = dist_meters




def convert_city_data(results):
    """
    Convert detailed square-mile results into the simple format:
    { "nodes": num_nodes, "nb_miles": nb_miles }
    """
    cities_simple = []

    for item in results:
        if not item.get("success", False):
            continue  # skip failed cities

        cities_simple.append({
            "nodes": item["num_nodes"],
            "nb_miles": item["nb_miles"]
        })

    return cities_simple

def get_city_square_data(cities, nb_miles=1):
    """
    Given a list of city names, download each network and return:
    - city_name
    - number of nodes inside the square-mile window
    - nb_miles
    - optionally a failure message
    """
    info_list = []

    for city_name in cities:
        print(f"\nProcessing {city_name}...")
        try:
            # 1. Geocode
            center_lat, center_lon = ox.geocode(city_name)

            # 2. Street network
            G = ox.graph_from_point(
                (center_lat, center_lon),
                dist=2000,
                network_type="drive"
            )

            # 3. Project graph
            G_proj = ox.project_graph(G)

            # 4. Extract square-mile nodes
            square_data = get_square_mile_nodes(
                G, G_proj, center_lat, center_lon, nb_miles
            )
            num_nodes = len(square_data["nodes"])

            info_list.append({
                "city_name": city_name,
                "num_nodes": num_nodes,
                "nb_miles": nb_miles,
                "success": True
            })

        except Exception as e:
            print(f"  ❌ Failed for {city_name}: {e}")
            info_list.append({
                "city_name": city_name,
                "nodes": None,
                "nb_miles": nb_miles,
                "success": False,
                "error": str(e)
            })

    return info_list




def building_network_with_p(N, grid_size, p):
    """
    Build a grid network of size grid_size x grid_size
    and connect each node to its 2 nearest neighbors.

    Ensures exactly N nodes are created by using a rectangular
    grid with dimensions chosen to fit N.
    Randomization avoids duplicate positions.
    """
    print(f"\nBuilding network: N={N}, grid={grid_size}×{grid_size}, p={p}")

    # choose grid dimensions that fit N nodes
    nodes_per_side_x = max(1, int(math.floor(math.sqrt(N))))
    nodes_per_side_y = max(1, int(math.ceil(N / nodes_per_side_x)))

    # spacing along each axis
    spacing_x = grid_size / nodes_per_side_x
    spacing_y = grid_size / nodes_per_side_y

    # Step 1: Place exactly N nodes on the grid
    nodes = []
    for i in range(nodes_per_side_x):
        for j in range(nodes_per_side_y):
            if len(nodes) >= N:
                break
            x = int(i * spacing_x)
            y = int(j * spacing_y)
            nodes.append([x, y])
        if len(nodes) >= N:
            break

    nodes = np.array(nodes)
    assert len(nodes) == N, f"internal error: generated {len(nodes)} nodes but expected {N}"

    # Step 2: Randomly move some nodes with probability p (no duplicates)
    used_positions = set(tuple(pos) for pos in nodes)
    moved_count = 0

    for i in range(len(nodes)):
        if np.random.random() < p:
            while True:
                new_x = np.random.randint(0, grid_size)
                new_y = np.random.randint(0, grid_size)
                if (new_x, new_y) not in used_positions:
                    used_positions.add((new_x, new_y))
                    nodes[i] = [new_x, new_y]
                    moved_count += 1
                    break

    # Step 3: Connect each node to its 2 nearest neighbors (O(N^2))
    edges = []
    connected = set()
    M = len(nodes)

    for i in range(M):
        distances = []
        for j in range(M):
            if i == j:
                continue
            # use your euclidean_distance(y1,x1, y2,x2)
            dist = euclidean_distance(
                nodes[i][1], nodes[i][0],
                nodes[j][1], nodes[j][0]
            )
            distances.append((dist, j))

        distances.sort(key=lambda t: t[0])

        # connect to 2 closest nodes not already connected
        edges_added = 0
        for dist, j in distances:
            edge_id = tuple(sorted((i, j)))
            if edge_id not in connected:
                edges.append((i, j, dist))
                connected.add(edge_id)
                edges_added += 1
                if edges_added >= 2:
                    break

    print(f"  Generated nodes: {len(nodes)}, moved: {moved_count}")
    print(f"  Edges: {len(edges)}, Avg degree: {2 * len(edges) / len(nodes):.2f}")

    return nodes, edges

def create_nx_graph_from_nodes_edges(nodes, edges):
    G = nx.Graph()

    # Add nodes with positions
    for i, (x, y) in enumerate(nodes):
        G.add_node(i, x=float(x), y=float(y))  # store coordinates as attributes

    # Add edges with distance as weight
    for i, j, dist in edges:
        G.add_edge(i, j, weight=dist)

    return G


"""
            ****************END OF RANDOM GENERATED GRAPHS FUNCTIONS****************-         
"""

def cumulative_distribution(values):
    """
    Calcule la distribution cumulée décroissante P(C) à partir d'une liste de valeurs.
    P(C_k) = proportion de nœuds ayant une centralité >= C_k
    """
    vals = np.array(values, dtype=float)
    # On enlève les NaN éventuels
    vals = vals[~np.isnan(vals)]
    vals = vals[vals > 0]


    # Tri des valeurs par ordre croissant
    vals_sorted = np.sort(vals)
    n = len(vals_sorted)

    # P(C) = 1 - F(C) = proba d'avoir une valeur >= C
    y = 1.0 - np.arange(n) / n

    return vals_sorted, y


"""
    ************* FIT FUNCTIONS ************
"""
def expo_model(x, s):
    """Modèle exponentiel pour le fit : P(C) = exp(-C/s)."""
    return np.exp(-x / s)


def fit_exponential(x, y):
    mask = (x > 0) & (y > 0)
    x_fit = x[mask]
    y_fit = y[mask]

    # modèle linéaire : ln P = a*C + b où a = -1/s
    Y = np.log(y_fit)

    # régression linéaire
    a, b = np.polyfit(x_fit, Y, 1)

    s = -1 / a
    return s

def gaussian_model(x, sigma):
    return np.exp(-(x**2) / (2 * sigma**2))

def fit_gaussian(x, y):
    mask = (x > 0) & (y > 0)
    x_fit = x[mask]
    y_fit = y[mask]

    X = x_fit**2
    Y = np.log(y_fit)

    # Régression linéaire Y = a * X + b
    a, b = np.polyfit(X, Y, 1)

    sigma = np.sqrt(-1 / (2 * a))   # car a = -1/(   2σ²)
    return sigma

def powerlaw_model(x, gamma, k=1.0):
    """
    Modèle power law : P(C) = k * C^(-gamma)
    """
    exponent = -(gamma - 1)
    return k * x ** (exponent)


def fit_powerlaw(x, y):
    mask = (x > 0) & (y > 0) & (y < 1)
    x_fit = x[mask]
    y_fit = y[mask]

    # log-log
    X = np.log(x_fit)
    Y = np.log(y_fit)

    # régression : Y = a*X + b
    a, b = np.polyfit(X, Y, 1)

    # a = -(gamma - 1)
    gamma = 1 - a

    # Constante de normalisation k = e^b
    k = np.exp(b)

    return gamma, k


def stretched_exp_model(x, lam, beta, A=1.0):
    return A * np.exp(-(x / lam)**beta)

def fit_stretched_exp_linear(x, y, xmin=None, xmax=None, ymax_excl=0.99):
    # 1. Masquage pour l'ajustement
    mask = (x > 0) & (y > 0) & (y < ymax_excl)

    if xmin is not None:
        mask &= (x >= xmin)
    if xmax is not None:
        mask &= (x <= xmax)

    x_fit = x[mask]
    y_fit = y[mask]

    # 2. Vérification de la validité
    if len(x_fit) < 2:
        # Retourne nan si pas assez de points pour la régression
        return np.nan, np.nan

        # 3. Transformation logarithmique (Double-log)
    X = np.log(x_fit)
    Y = np.log(-np.log(y_fit))

    # 4. Régression
    a, b = np.polyfit(X, Y, 1)

    # 5. Calcul des paramètres
    beta = a
    # Gérer la division par zéro si a est proche de zéro
    if np.abs(a) < 1e-9:
        return np.nan, beta

    lam = np.exp(-b / a)

    return lam, beta

"""
        ************ END OF FIT FUNCTIONS ************
"""