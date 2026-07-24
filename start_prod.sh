#!/usr/bin/env bash

# --- Fonction de nettoyage ---
cleanup() {
    echo "🛑 [PROD] Signal d'arrêt reçu."
    kill -TERM $VLLM_PID $API_PID $GRADIO_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# --- Configuration ---
export VLLM_PORT=${VLLM_PORT:-8000}
export PORT=${PORT:-8080}

echo "🚀 [PROD] Démarrage... (PORT: $PORT, VLLM: $VLLM_PORT)"

# 1. Lancement VLLM
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 &
VLLM_PID=$!

# Attente VLLM
until curl -s http://localhost:${VLLM_PORT}/health > /dev/null; do
    sleep 2
done

# 2. Lancement API FastAPI (C'est lui qui doit écouter sur $PORT)
echo "📡 [PROD] Lancement API FastAPI sur port $PORT..."
export VLLM_API_BASE="http://localhost:${VLLM_PORT}/v1"
exec python -m app.main &
API_PID=$!
echo "📡 [PROD] API FastAPI lancée avec PID $API_PID"

# 3. Lancement Gradio
python app/ui.py --server-name 0.0.0.0 --server-port 7860 &
GRADIO_PID=$!

# 4. Garder le conteneur vivant
wait -n $API_PID
cleanup
