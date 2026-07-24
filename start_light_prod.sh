#!/usr/bin/env bash
set -e

# Configuration
export VLLM_PORT=${VLLM_PORT:-8000}
export PORT=${PORT:-8080}
# Utilisation de Qwen2.5-1.5B (léger et performant)
export MODEL_ID=${MODEL_ID:-"Qwen/Qwen2.5-1.5B-Instruct"}

echo "🚀 [CONSOLIDATED] Lancement VLLM et API..."

# 1. Lancement VLLM en arrière-plan
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --gpu-memory-utilization 0.5 &
VLLM_PID=$!

# 2. Attente VLLM opérationnel
echo "⏳ Attente de VLLM..."
until curl -s http://localhost:${VLLM_PORT}/health > /dev/null; do
    sleep 2
done

# 3. Lancement API FastAPI en foreground (Main Process)
echo "📡 Lancement API sur port $PORT..."
export VLLM_API_BASE="http://localhost:${VLLM_PORT}/v1"
exec python -m app.main
