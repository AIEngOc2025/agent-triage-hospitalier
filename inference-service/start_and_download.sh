#!/usr/bin/env bash
set -e

echo "🚀 Downloading model from GCS..."
python3 /app/inference-service/download_model.py

echo "🚀 Starting vLLM..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "/app/models/merged_dpo_final_chsa" \
    --host 0.0.0.0 \
    --port 8080
