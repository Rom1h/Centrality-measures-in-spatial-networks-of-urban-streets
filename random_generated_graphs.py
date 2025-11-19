import networkx as nx

from centralities import betweenness_centrality, information_centrality
from utils import building_network_with_p, create_nx_graph_from_nodes_edges, nodes_Size_compute, \
    add_edge_lengths_to_graph, get_city_square_data, convert_city_data
from visualisation import plot_p_values, plot_cumulative_distribution_centrality, \
    plot_cumulative_distribution_centralities_p
import matplotlib

cities = [
    {"nodes": 250, "nb_miles": 1},
    {"nodes": 350, "nb_miles": 1},
    {"nodes": 270, "nb_miles": 1},
]
cities = [
    "New York, USA",
    "Singapore",
    "Marseille, France",
    "London, UK"
]

results = get_city_square_data(cities, nb_miles=1)

cities = convert_city_data(results)

for r in results:
    print(r)
p = 0.0
grid_size = 50
N = nodes_Size_compute(cities)
print(f"N = {N}")
p_values = [0, 0.1, 0.2, 1]
# to visualise the grid with different p values uncomment the line below
# plot_p_values(p_values, N, grid_size)

nodes, edges = building_network_with_p(N, grid_size, p)
G1 = create_nx_graph_from_nodes_edges(nodes, edges)
add_edge_lengths_to_graph(G1,nodes,edges, grid_size)

p = 0.2
nodes, edges = building_network_with_p(N, grid_size, p)
G2 = create_nx_graph_from_nodes_edges(nodes, edges)
add_edge_lengths_to_graph(G2,nodes,edges, grid_size)

# w

i_G1 = list(information_centrality(G1).values())
i_G2 = list(information_centrality(G2).values())

i_datasets = [
    (i_G1, "Randomized P=0", "steelblue"),
    (i_G2, "Randomized P=0.2", "darkorange"),
]

plot_cumulative_distribution_centralities_p(
    i_datasets,
    title="Comparison of Information Centrality Distributions",
    output_file="comparison_information.png.png",
    model="powerlaw",
    method="CI"
)