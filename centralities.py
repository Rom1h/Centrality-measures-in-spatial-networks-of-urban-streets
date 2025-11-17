from utils import djikstra_shortest_paths, euclidean_distance, shortest_dijkstra, general_contrib, efficiency


def straightness(G_proj):
    """
    Uses the djikstra_shortest_paths to compute the straightness centrality for all nodes in the graph.
    Uses Euclidean to calculate the straight distance between two nodes

    Complexity should be O(N * (E + N) log N)
    Where N is the number of nodes and E the number of edges

    """
    n = len(G_proj.nodes)
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

def information_centrality(G):
    inf_centr = {}
    for i in G.nodes:
        N = len(G.nodes)

        G_prime = G.copy()

        G_prime.remove_node(i)

        ef = efficiency(G, N)
        ef_prime = efficiency(G_prime, N)
        if ef != 0 :
            inf_centr[i] = (ef - ef_prime) / ef
        else : 
            inf_centr[i] = 0

    return inf_centr