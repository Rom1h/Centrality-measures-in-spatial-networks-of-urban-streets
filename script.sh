#!/bin/bash

# --- Nom du Script Python ---
SCRIPT_NAME="exec.py"

# --- 1. Vérification des Arguments ---
# $# contient le nombre d'arguments passés au script
if [ "$#" -ne 4 ]; then
    echo "Erreur : Ce script nécessite 4 arguments."
    echo "Usage : ./run_analysis.sh <CITY> <PLACE> <METHOD> <MODEL>"
    echo ""
    echo "Exemple :"
    echo "./run_analysis.sh \"Los Angeles, California, USA\" \"Hollywood\" \"b\" \"O\""
    exit 1
fi

# --- 2. Affectation des Variables ---
# $1 est le premier argument, $2 le deuxième, etc.
CITY_NAME="$1"
PLACE_NAME="$2"
METHOD="$3"
MODEL="$4"

# --- 3. Exécution de la Commande ---
echo "--- Analyse de Centralité ---"
echo "Ville : ${CITY_NAME}"
echo "Lieu : ${PLACE_NAME}"
echo "Méthode : ${METHOD} (Centralité de l'intermédiarité)"
echo "Modèle : ${MODEL} (Organisé)"
echo "------------------------------"

echo "Exécution de la commande :"
echo "python ${SCRIPT_NAME} --city \"${CITY_NAME}\" --place \"${PLACE_NAME}\" --method \"${METHOD}\" --model \"${MODEL}\""
echo ""

pip install -r requirements.txt
# Exécution du script Python avec les arguments fournis par l'utilisateur
python3 "${SCRIPT_NAME}" \
    --city "${CITY_NAME}" \
    --place "${PLACE_NAME}" \
    --method "${METHOD}" \
    --model "${MODEL}"

# --- 4. Vérification du Statut de Sortie ---
if [ $? -eq 0 ]; then
    echo ""
    echo "Le script a été exécuté avec succès"
else
    echo ""
    echo "Une erreur s'est produite lors de l'exécution du script."
fi