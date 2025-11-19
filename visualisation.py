import matplotlib

# Only for not docker
# matplotlib.use('TkAgg')
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import box, Point

from utils import get_square_mile_nodes, building_network_with_p, cumulative_distribution, gaussian_model, fit_gaussian,expo_model,fit_exponential,fit_powerlaw,powerlaw_model, fit_stretched_exp_linear, stretched_exp_model




def visualise_graph(G_proj, square_data, zoom=False):
    """
    Visualize the graph with highlighted square mile area.
    Only possible in pc , not in Docker , Docker has no display server Unless you plot it on a website

    Args:
    G: Original graph
    G_proj: Projected graph
    square_data: Dictionary returned from get_square_mile_nodes()
    zoom: Whether to zoom into the square area
    """
    nodes = square_data['nodes']
    center_x = square_data['center_x']
    center_y = square_data['center_y']
    square = square_data['square']
    half_side = square_data['half_side']
    nb_miles = square_data['nb_miles']

    if not nodes:
        print("⚠️ No nodes found in square")
        return

    G_square = G_proj.subgraph(nodes).copy()
    print(f"✅ {len(G_square.nodes)} nodes, {len(G_square.edges)} edges")

    # Print first 10 nodes as sample just to see them
    for node_id, data in list(G_square.nodes(data=True))[:10]:
        print(f"Node {node_id}: x={data['x']:.1f}, y={data['y']:.1f}")

    center_coords = (center_x, center_y)

    # Plot graph 1 , Shows the Full graph
    fig, ax = ox.plot_graph(G_proj, show=False, close=False,
                            node_size=8,
                            node_color='lightblue',
                            edge_color='lightblue',
                            edge_linewidth=1,
                            bgcolor='white')

    # Plot the square boundary
    x_square, y_square = square.exterior.xy
    ax.plot(x_square, y_square, color='red', linewidth=3,
            label=f'{nb_miles} sq mile boundary', zorder=4)

    # Plot the center node
    ax.scatter(*center_coords, color='blue', s=200, label='Center',
               zorder=5, edgecolors='white', linewidth=2)

    # Highlight nodes inside the square in GREEN
    nodes_coords = [(data['x'], data['y']) for n, data in G_square.nodes(data=True)]
    xs, ys = zip(*nodes_coords)

    if zoom:
        # Zoomed: bigger, more visible green nodes
        ax.scatter(xs, ys, color='green', s=40, alpha=0.9, label='Nodes in square',
                   zorder=3, edgecolors='darkgreen', linewidth=0.5)
    else:
        # Full view: smaller, subtle green nodes
        ax.scatter(xs, ys, color='green', s=5, alpha=0.4, label='Nodes in square',
                   zorder=3)

    if zoom:
        # Set zoom limits to the square area
        padding = half_side * 0.1
        ax.set_xlim(center_x - half_side - padding, center_x + half_side + padding)
        ax.set_ylim(center_y - half_side - padding, center_y + half_side + padding)

    ax.legend(fontsize=12)
    title = "Zoomed In" if zoom else "Full View"
    ax.set_title(f"Paris Street Network - {title}", fontsize=14)

    plt.show()


def plot_p_values(p_values, N, grid_size):
    plt.figure(figsize=(18, 18))

    for idx, p in enumerate(p_values):
        nodes, edges = building_network_with_p(N, grid_size, p)

        plt.subplot(2, 2, idx + 1)

        # Draw edges
        for i, j, dist in edges:
            pos_i = nodes[i]
            pos_j = nodes[j]
            plt.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], alpha=0.3)

        # Draw nodes
        plt.scatter(nodes[:, 0], nodes[:, 1], color='red', s=40, zorder=5)

        plt.title(f"p = {p}")
        plt.xlim(0, grid_size)
        plt.ylim(0, grid_size)
        plt.gca().set_aspect('equal', adjustable='box')

    plt.suptitle(
        f"Comparison of p values, N={N}, grid={grid_size}×{grid_size}",
        fontsize=18
    )
    plt.tight_layout()
    plt.show()

"""
G = ox.graph_from_place("Paris, France", network_type="drive")
G_proj = ox.project_graph(G)
center_lat, center_lon = ox.geocode("Notre-Dame, Paris, France")
notre_dame_data = get_square_mile_nodes(G, G_proj, center_lat, center_lon, nb_miles=1)

# Full view
visualise_graph(G_proj,notre_dame_data, zoom=False)

# Zoomed view
visualise_graph(G_proj,notre_dame_data, zoom=True)

"""

def plot_cumulative_distribution_centrality(
    values,
    method = "Centrality ",# betweenness
    title="Cumulative distribution of betweenness",
    output_file="betweenness_cumulative.png",
    model="exp",# "exp" ou "gauss"
):
    x, y = cumulative_distribution(values)

    # --- choix du modèle ---
    if model == "exp":
        param = fit_exponential(x, y)
        y_fit = expo_model(x, param)
        label_fit = f"Exp fit  P(C)≈exp(-C/s), s={param:.4f}"

    elif model == "gauss":
        param = fit_gaussian(x, y)
        y_fit = gaussian_model(x, param)
        label_fit = f"Gauss fit  P(C)≈exp(-C²/(2σ²)), σ={param:.4f}"

    else:
        raise ValueError("model doit être 'exp' ou 'gauss'")

    # --- Plot ---
    plt.figure(figsize=(6, 4))
    plt.scatter(
        x, y,
        s=12,                 # taille des points
        alpha=0.6,
        label="Données (C_B)",
        color="steelblue"
    )

    # Ligne lisse pour la courbe ajustée
    plt.plot(
        x, y_fit,
        "--",
        label=label_fit,
        linewidth=2,
        color="darkorange"
    )

    # Échelle log en Y
    plt.yscale("log")

    plt.xlabel(method +" centrality C_B")
    plt.ylabel("P(C_B)")
    plt.title(title)

    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Figure sauvegardée sous : {output_file}")

def plot_multi_city_cdf(city_data,
                        model="exp",              # "exp", "gauss" ou "powerlaw"
                        title="CDF comparison",
                        output_file=None,
                        method="",
                        ) :
    """
    city_data : dict -> { "nom_ville": liste_centralités }
    model : "exp", "gauss" ou "powerlaw"
    """

    plt.figure(figsize=(7, 7))

    for city_name, values in city_data.items():

        # --- Distribution cumulée P(C ≥ x) ---
        x, y = cumulative_distribution(values)

        # --- Fit selon le modèle choisi ---
        if model == "exp":
            s = fit_exponential(x, y)
            y_fit = expo_model(x, s)
            fit_label = f"{city_name}  exp(s={s:.4f})"
        elif model == "gauss":
            sigma = fit_gaussian(x, y)
            y_fit = gaussian_model(x, sigma)
            fit_label = f"{city_name}  gauss(σ={sigma:.4f})"

        elif model == "powerlaw":
            gamma, k = fit_powerlaw(x, y)
            y_fit = powerlaw_model(x,gamma, k=k)
            fit_label = f"{city_name} power-law(γ={gamma:.2f})"
        elif model == "str":
            lam, beta = fit_stretched_exp_linear(x, y)
            y_fit = stretched_exp_model(x, lam, beta)
            fit_label = f"Stretched exp  P(C)≈exp(-(C/λ)^β), λ={lam:.4f}, β={beta:.3f}"
        else:
            raise ValueError("model doit être 'exp', 'gauss' ou 'powerlaw'")

        # --- Points de données ---
        plt.scatter(x, y, s=10, alpha=0.7, label=city_name)

        # --- Courbe de fit ---
        plt.plot(x, y_fit, "--", linewidth=2, label=fit_label)

    # --- Mise en forme générale ---
    plt.yscale("log")

    plt.xlabel(f"Centrality C{method}")
    plt.ylabel("P(C)")
    plt.title(title)
    plt.grid(True, which="both", linestyle=":", alpha=0.3)
    plt.legend()

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, dpi=300)
        print("Figure sauvegardée sous :", output_file)
    plt.show()
