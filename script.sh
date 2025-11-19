#!/bin/bash

# --- Configuration ---
SCRIPT_NAME="exec.py"
VENV_PATH="venv" # Nom de l'environnement virtuel (créé via 'python3 -m venv venv')

# --- 1. Vérification des Arguments (3 requis : PLACE, METHOD, MODEL) ---
if [ "$#" -ne 3 ]; then
    echo "Erreur : Ce script nécessite 3 arguments (PLACE, METHOD, MODEL)."
    echo "Usage : ./run_analysis.sh <PLACE> <METHOD> <MODEL>"
    echo ""
    echo "Exemple :"
    echo "./run_analysis.sh \"Manhattan, New York, USA\" \"b\" \"O\""
    exit 1
fi

# --- 2. Affectation des Variables ---
# Les arguments sont indexés à partir de 1.
PLACE_NAME="$1"
METHOD="$2"
MODEL="$3"

# --- 3. Préparation de l'Environnement ---

echo "Preparation de l'environnement Python..."

# Activation de l'environnement virtuel
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "   Environnement '$VENV_PATH' active."
else
    echo "ATTENTION : Environnement virtuel '$VENV_PATH' non trouve. Execution avec le python systeme."
fi

# Installation des dépendances (s'assure que tout est à jour)
echo "   Installation des dependances (via requirements.txt)..."
# Le '||' permet d'arrêter le script si pip échoue.
pip install -r requirements.txt || { echo "ERREUR : Echec de l'installation des dependances."; exit 1; }

# --- 4. Exécution de la Commande ---
echo "--- Analyse de Centralite ---"
echo "Lieu : ${PLACE_NAME}"
echo "Methode : ${METHOD}"
echo "Modele : ${MODEL}"
echo "------------------------------"

echo "Execution de la commande :"
# Utilise l'interpréteur 'python' qui est maintenant lié à l'environnement virtuel actif
python "${SCRIPT_NAME}" \
    --place "${PLACE_NAME}" \
    --method "${METHOD}" \
    --model "${MODEL}"

EXIT_CODE=$?

# --- 5. Nettoyage et Vérification du Statut de Sortie ---

# Désactive l'environnement virtuel après l'exécution (si actif)
if [ -d "$VENV_PATH" ]; then
    deactivate
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Le script a ete execute avec succes (Code 0)."
else
    echo ""
    echo "Une erreur s'est produite lors de l'execution du script (Code $EXIT_CODE)."
fi