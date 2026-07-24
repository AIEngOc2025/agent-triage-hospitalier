#!/usr/bin/env bash
set -e

# Configuration
# On force le port à 8080 pour que la startup probe de Cloud Run le détecte
export VLLM_PORT=${PORT:-8080}
export MODEL_ID=${MODEL_ID:-"Qwen/Qwen2.5-1.5B-Instruct"}

echo "🚀 [VLLM ONLY] Lancement VLLM sur port $VLLM_PORT..."

# Lancement VLLM en foreground (Main Process)
# Cloud Run attend qu'un processus écoute sur le port PORT
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --gpu-memory-utilization 0.5
