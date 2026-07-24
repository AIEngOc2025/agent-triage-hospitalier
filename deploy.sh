#!/usr/bin/env bash

# Script de déploiement pour l'architecture découplée du Triage Agent

set -e

REGION="europe-west1"
SERVICE_INFERENCE="inference-service"
SERVICE_APP="app-service"

echo "🚀 Déploiement du Service d'Inférence (vLLM)..."
# Déploiement avec les ressources requises pour vLLM
gcloud run deploy $SERVICE_INFERENCE \
  --source inference-service/ \
  --region $REGION \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --memory 32Gi \
  --cpu 4 \
  --min-instances 1 \
  --no-cpu-throttling

# Récupérer l'URL du service d'inférence
INFERENCE_URL=$(gcloud run services describe $SERVICE_INFERENCE --region $REGION --format='value(status.url)')
echo "✅ Service d'Inférence déployé : $INFERENCE_URL"

echo "🚀 Déploiement du Service Applicatif (API + UI)..."
# Déploiement du service applicatif avec l'URL de l'inférence
gcloud run deploy $SERVICE_APP \
  --source app-service/ \
  --region $REGION \
  --memory 2Gi \
  --cpu 1 \
  --set-env-vars INFERENCE_SERVICE_URL=$INFERENCE_URL

APP_URL=$(gcloud run services describe $SERVICE_APP --region $REGION --format='value(status.url)')
echo "✅ Service Applicatif déployé : $APP_URL"
echo "🎉 Déploiement complet !"
