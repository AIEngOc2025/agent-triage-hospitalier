#!/usr/bin/env bash

# Script de démarrage robuste pour la production (Cloud Run).
# Gère le lancement séquentiel des dépendances (VLLM), 
# puis lance l'application principale (FastAPI) en foreground.

# --- Nettoyage ---
cleanup() {
    echo "🛑 [PROD] Signal d'arrêt reçu. Arrêt des services..."
    kill -TERM $VLLM_PID $GRADIO_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# --- Configuration ---
export VLLM_PORT=${VLLM_PORT:-8000}
export PORT=${PORT:-8080}
export VLLM_API_BASE="http://localhost:${VLLM_PORT}/v1"

echo "🚀 [PROD] Démarrage... (PORT: $PORT, VLLM: $VLLM_PORT)"

# 1. Lancement VLLM (Background)
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 &
VLLM_PID=$!

# Attente VLLM
echo "⏳ [PROD] Attente du démarrage de VLLM..."
until curl -s http://localhost:${VLLM_PORT}/health > /dev/null; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ ERREUR: Le serveur VLLM a planté."
        exit 1
    fi
    sleep 5
done
echo "✅ [PROD] VLLM opérationnel."

# 2. Lancement Gradio (Background)
echo "💻 [PROD] Lancement Gradio..."
python app/ui.py --server-name 0.0.0.0 --server-port 7860 &
GRADIO_PID=$!

# 3. Lancement API FastAPI (Foreground - Main Process)
echo "📡 [PROD] Lancement API FastAPI sur port $PORT..."
python -m app.main
# Si FastAPI s'arrête, on nettoie
cleanup
