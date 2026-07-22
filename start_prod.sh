#!/usr/bin/env bash

# Script de démarrage pour l'environnement de PRODUCTION (conteneur Docker).
# Ce script utilise un superviseur de processus simple pour s'assurer que si l'un
# des services (VLLM ou Gradio) tombe, le conteneur entier s'arrête et redémarre.

# --- Fonction de nettoyage ---
cleanup() {
    echo "🛑 [PROD] Signal d'arrêt reçu. Nettoyage des processus..."
    kill -TERM $VLLM_PID $GRADIO_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Script de démarrage pour l'environnement de PRODUCTION (conteneur Docker).

# --- Configuration ---
# Valider que la variable d'environnement MODEL_ID est fournie au conteneur.
if [ -z "$MODEL_ID" ]; then
    echo "❌ ERREUR: La variable d'environnement MODEL_ID est requise mais n'a pas été fournie au conteneur."
    exit 1
fi

# Utilise les variables d'environnement pour la configuration, avec des valeurs par défaut.
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_HOST=${VLLM_HOST:-"localhost"} # 'localhost' est correct car les deux services sont dans le même conteneur.
# Cloud Run fournit la variable PORT. Gradio doit écouter sur ce port.
GRADIO_PORT=${PORT:-8080}

# --- 1. Lancement du serveur VLLM ---
# Lance le serveur en arrière-plan.
# Les logs (stdout/stderr) sont directement gérés par l'environnement du conteneur (ex: Cloud Run).
echo "🚀 [PROD] Lancement du serveur VLLM en arrière-plan..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    &

# Sauvegarde le PID (Process ID) du serveur VLLM pour le monitoring.
VLLM_PID=$!

# --- 2. Attente du serveur VLLM ---
# Boucle jusqu'à ce que l'endpoint de santé du serveur réponde.
echo "⏳ [PROD] Attente du démarrage du serveur VLLM..."
HEALTH_CHECK_URL="http://${VLLM_HOST}:${VLLM_PORT}/health"

until curl -s --fail "$HEALTH_CHECK_URL" > /dev/null; do
    # Si le processus VLLM a planté, le conteneur doit s'arrêter.
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ ERREUR: Le serveur VLLM a cessé de fonctionner. Vérifiez les logs du conteneur."
        exit 1
    fi
    echo "   [PROD] Serveur VLLM non disponible, nouvelle tentative dans 2s..."
    sleep 2
done
echo "✅ [PROD] Serveur VLLM opérationnel."

# --- 3. Lancement de l'interface Gradio ---
# Lance l'UI en arrière-plan et écoute sur le port fourni par Cloud Run.
echo "💻 [PROD] Lancement de l'interface Gradio sur le port ${GRADIO_PORT}..."
export VLLM_API_BASE="http://${VLLM_HOST}:${VLLM_PORT}/v1"
python app/ui.py --server-name 0.0.0.0 --server-port "${GRADIO_PORT}" &
GRADIO_PID=$!

# --- 4. Superviseur de processus ---
# Attend que l'un des deux processus (VLLM ou Gradio) se termine.
# Si l'un s'arrête, le script se termine, ce qui arrête le conteneur.
wait -n $VLLM_PID $GRADIO_PID

# Vérifie quel processus s'est arrêté et logue l'information.
if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "❌ ERREUR: Le serveur VLLM s'est arrêté de manière inattendue."
elif ! kill -0 $GRADIO_PID 2>/dev/null; then
    echo "❌ ERREUR: L'interface Gradio s'est arrêtée de manière inattendue."
fi

# Déclenche le nettoyage pour s'assurer que l'autre processus est bien arrêté.
cleanup