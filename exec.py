
import matplotlib
import argparse
import osmnx.distance
import random
from centralities import straightness, Closeness , betweenness_centrality,information_centrality
from utils import nodes_Size_compute

# matplotlib.use("Agg")  # Headless-safe backend for Docker
matplotlib.use('TkAgg') # For testing on pc
from functools import lru_cache
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
import os
import networkx as nx
from visualisation import get_square_mile_nodes, plot_p_values,plot_cumulative_distribution_centrality,plot_multi_city_cdf

def get_city_centrality(place, method="b"):
    """
    Télécharge un sous-graphe autour d'un lieu et calcule sa centralité.

    method : "betweenness" ou "closeness"
    Retourne : nom de ville, liste des centralités
    """
    center_lat, center_lon = ox.geocode(place)
    # 2. Street network
    G = ox.graph_from_point((center_lat, center_lon), dist=2000,
                            network_type='drive')

    # 3. Project graph
    G_proj = ox.project_graph(G)

    # 4. Extract square-mile nodes
    square_data = get_square_mile_nodes(G, G_proj, center_lat, center_lon, 2500)

    # Projection (important pour distances correctes)
    G_sub = G_proj.subgraph(square_data["nodes"]).copy()    # Centralité
    if method == "b":
        centrality = betweenness_centrality(G_sub, normalized=True)
    elif method == "c":
        centrality = Closeness(G_sub)
    elif method == "i":
        centrality = information_centrality(G_sub)
    elif method == "s":

        centrality = straightness(G_sub)

    else:
        raise ValueError("method doit être 'betweenness' ou 'closeness'.")

    return centrality


def main( place: str, method: str, model: str):
    """
    Exécute l'analyse de centralité pour un seul lieu et trace la CCDF ajustée.

    :param city: Nom de la ville (pour la légende).
    :param place: Nom complet du lieu à analyser (e.g., "Manhattan, New York, USA").
    :param method: Type de centralité à calculer ("b", "c", "i", "s").
    :param model: Modèle d'organisation de la ville ("SO" pour Auto-Organisée, "O" pour Organisée).
    """

    # --- 1. Préparation du lieu et de la structure de données ---

    # city_name est le nom court utilisé dans le dictionnaire et le titre

    # Le dictionnaire 'city_data' doit contenir le nom de la ville et les valeurs de centralité
    city_data = {}

    # --- 2. Calcul des centralités ---
    print(f"-> Calcul de la centralité '{method}' pour{place}")

    # On suppose que get_city_centrality retourne la liste des valeurs de centralité
    C = get_city_centrality(place, method=method)

    # Remplissage du dictionnaire pour le tracé (clé=nom, valeur=liste des centralités)
    city_data[place] = list(C.values())

    # --- 3. Détermination du modèle de fit et Tracé ---

    # Définition des variables de sortie
    output_name = "output"
    title = f"Distribution de la Centralité {method.upper()} pour {output_name} ({model})"
    output_file = f"CCDF_{method}_{output_name}.png"

    # Variable qui va contenir le modèle de fit à utiliser
    fit_model = ""

    # Logique de sélection du modèle de fit
    # NOTE: Les méthodes sont mappées sur le type de centralité : I=Information, B=Betweenness, S=Straightness, C=Closeness

    # --- Villes Auto-Organisées (SO) ---
    if model == "SO":
        if method == "i":
            fit_model = "powerlaw"
        elif method == "b":
            fit_model = "exp"
        elif method == "s":
            fit_model = "str"
        elif method == "c":
            fit_model = "exp"
        else:
            raise ValueError(f"Méthode '{method}' non reconnue pour le modèle SO.")

    # --- Villes Organisées (O) ---
    elif model == "O":
        if method == "i":
            fit_model = "exp"
        elif method == "b":
            fit_model = "gauss"
        elif method == "s":
            fit_model = "str"
        elif method == "c":
            # Le dernier cas 'O' && 'S' dans votre code doit probablement être 'O' && 'C'
            fit_model = "exp"
        else:
            raise ValueError(f"Méthode '{method}' non reconnue pour le modèle O.")
    else:
        raise ValueError(f"Type de ville '{model}' non reconnu. Utilisez 'SO' ou 'O'.")

    # --- Exécution du tracé ---
    print(f"-> Ajustement avec le modèle : {fit_model.capitalize()}")

    plot_multi_city_cdf(
        city_data=city_data,
        model=fit_model,  # On passe le modèle de fit calculé (powerlaw, exp, etc.)
        title=title,
        output_file=output_file,
        method=method,
    )

    plt.show()
    print(f"Analyse terminée. Graphique sauvegardé sous {output_file}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyse de la distribution de centralité pour un lieu unique et ajustement d'un modèle.",
        formatter_class=argparse.RawTextHelpFormatter
    )


    parser.add_argument(
        '--place',
        type=str,
        required=True,
        help="Lieu de focus spécifique dans la ville (e.g., 'Times Square')"
    )


    parser.add_argument(
        '--method',
        type=str,
        default='b',
        choices=['b', 'c', 'i', 's'],
        help="Méthode de centralité : [b (betweenness), c (closeness), i (information), s (straightness)]"
    )

    parser.add_argument(
        '--model',
        type=str,
        default='SO',
        choices=['SO', 'O'],
        help="Modèle d'organisation : [SO (Auto-Organisée), O (Organisée)]"
    )

    args = parser.parse_args()

    main(
        place=args.place,
        method=args.method,
        model=args.model
    )