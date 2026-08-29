#!/usr/bin/env bash
set -e

# Configuration
export VLLM_PORT=${VLLM_PORT:-8000}
export PORT=${PORT:-8080}
# Utilisation du modèle entraîné par défaut
export MODEL_ID=${MODEL_ID:-"/app/models/merged_dpo_final_chsa"}

# Cleanup
cleanup() {
    echo "🛑 Arrêt des services..."
    kill -TERM $VLLM_PID $API_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

echo "🚀 [ORCHESTRATED] Démarrage des services..."

# 1. Lancement VLLM
# Augmentation de la limite mémoire GPU à 0.9 pour de meilleures performances
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 &
    VLLM_PID=$!

until curl -s http://localhost:${VLLM_PORT}/health > /dev/null; do
    sleep 5
done
echo "✅ VLLM opérationnel avec le modèle $MODEL_ID."

# 2. Lancement FastAPI (Main Process)
# Utilisation de exec pour que FastAPI reprenne le PID 1
echo "🚀 Lancement de l'API Gateway..."
export VLLM_API_BASE="http://localhost:${VLLM_PORT}/v1"
exec python3 -m app.main
