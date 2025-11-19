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


def expo_model(x, s):
    """Modèle exponentiel pour le fit : P(C) = exp(-C/s)."""
    return np.exp(-x / s)


def fit_exponential(x, y):
    # garder seulement les points > 0
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
    return k * x**(-gamma)


def fit_powerlaw(x, y):
    """
    Fit d'une loi de puissance P(C) ~ C^(-gamma)
    via régression linéaire sur log-log.
    Retourne gamma et k.
    """
    # garder seulement les points > 0
    mask = (x > 0) & (y > 0)
    x_fit = x[mask]
    y_fit = y[mask]

    # Changement de variables : X = ln x, Y = ln y
    X = np.log(x_fit)
    Y = np.log(y_fit)

    # Régression linéaire Y = a X + b
    a, b = np.polyfit(X, Y, 1)

    gamma = -a           # a = -gamma
    return gamma

def stretched_exp_model(x, lam, beta, A=1.0):
    return A * np.exp(-(x / lam)**beta)
def fit_stretched_exp_linear(x, y, xmin=None, xmax=None):
    # on garde uniquement les points pertinents
    mask = (x > 0) & (y > 0)
    if xmin is not None:
        mask &= (x >= xmin)
    if xmax is not None:
        mask &= (x <= xmax)

    x_fit = x[mask]
    y_fit = y[mask]

    # Transformation
    X = np.log(x_fit)
    Y = np.log(-np.log(y_fit))

    # Régression linéaire
    a, b = np.polyfit(X, Y, 1)

    beta = a
    lam = np.exp(-b / a)

    return lam, beta

