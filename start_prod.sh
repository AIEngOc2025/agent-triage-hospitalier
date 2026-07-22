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

# --- Configuration ---
if [ -z "$MODEL_ID" ]; then
    echo "❌ ERREUR: La variable d'environnement MODEL_ID est requise."
    exit 1
fi

VLLM_PORT=${VLLM_PORT:-8000}
VLLM_HOST="0.0.0.0" # Utiliser 0.0.0.0 pour écouter sur toutes les interfaces
GRADIO_PORT=${PORT:-8080}
LOG_DIR="/var/log/app"
mkdir -p "$LOG_DIR"

# --- 1. Lancement du serveur VLLM ---
echo "🚀 [PROD] Lancement du serveur VLLM..."
# On limite la mémoire VRAM à 90% pour éviter l'OOM Killer
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    > "$LOG_DIR/vllm.log" 2>&1 &

VLLM_PID=$!

# --- 2. Attente du serveur VLLM ---
echo "⏳ [PROD] Attente du démarrage du serveur VLLM..."
HEALTH_CHECK_URL="http://localhost:${VLLM_PORT}/health"

until curl -s --fail "$HEALTH_CHECK_URL" > /dev/null; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ ERREUR: Le serveur VLLM a cessé de fonctionner."
        exit 1
    fi
    sleep 2
done
echo "✅ [PROD] Serveur VLLM opérationnel."

# --- 3. Lancement de l'interface Gradio ---
echo "💻 [PROD] Lancement de l'interface Gradio..."
export VLLM_API_BASE="http://localhost:${VLLM_PORT}/v1"
python app/ui.py --server-name 0.0.0.0 --server-port "${GRADIO_PORT}" \
    > "$LOG_DIR/gradio.log" 2>&1 &
GRADIO_PID=$!

# --- 4. Superviseur de processus ---
wait -n $VLLM_PID $GRADIO_PID

if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "❌ ERREUR: Le serveur VLLM s'est arrêté."
elif ! kill -0 $GRADIO_PID 2>/dev/null; then
    echo "❌ ERREUR: L'interface Gradio s'est arrêtée."
fi

cleanup
