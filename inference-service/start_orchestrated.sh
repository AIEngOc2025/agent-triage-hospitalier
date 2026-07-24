#!/usr/bin/env bash
set -e

# Configuration
export VLLM_PORT=${VLLM_PORT:-8000}
export PORT=${PORT:-8080}
export MODEL_ID=${MODEL_ID:-"Qwen/Qwen3-1.7B-Base"}

# Cleanup
cleanup() {
    echo "🛑 Arrêt des services..."
    kill -TERM $VLLM_PID $API_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

echo "🚀 [ORCHESTRATED] Démarrage des services..."

# 1. Lancement VLLM
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    --gpu-memory-utilization 0.5 &
VLLM_PID=$!

until curl -s http://localhost:${VLLM_PORT}/health > /dev/null; do
    sleep 5
done
echo "✅ VLLM opérationnel."

# 2. Lancement FastAPI
export VLLM_API_BASE="http://localhost:${VLLM_PORT}/v1"
python3 -m app.main &
API_PID=$!

# 3. Lancement Gradio (Main Process)
echo "💻 Lancement Gradio..."
exec python3 app.ui --server-name 0.0.0.0 --server-port 7860
