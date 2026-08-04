# Utilisation de l'image officielle vLLM optimisée pour l'API compatible OpenAI
FROM vllm/vllm-openai:latest

# Passage en utilisateur root pour installer les paquets système requis
USER root

# Installation de curl et du toolkit CUDA
RUN apt-get update && apt-get install -y \
    curl \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Installation des dépendances pour le téléchargement GCS
RUN pip install -r inference-service/requirements.txt
RUN chmod +x /app/inference-service/start_and_download.sh

# Définition du port par défaut pour Cloud Run (8080)
ENV PORT=8080
ENV HOST=0.0.0.0

# On utilise le script de téléchargement comme entrypoint
ENTRYPOINT ["/bin/bash", "/app/inference-service/start_and_download.sh"]
