import matplotlib

# Only for not docker
# matplotlib.use('TkAgg')
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import box, Point


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


G = ox.graph_from_place("Paris, France", network_type="drive")
G_proj = ox.project_graph(G)
center_lat, center_lon = ox.geocode("Notre-Dame, Paris, France")
notre_dame_data = get_square_mile_nodes(G, G_proj, center_lat, center_lon, nb_miles=1)

# Full view
visualise_graph(G_proj,notre_dame_data, zoom=False)

# Zoomed view
visualise_graph(G_proj,notre_dame_data, zoom=True)