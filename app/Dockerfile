# --- 1. IMAGE DE BASE ---
# Utilisation d'une image officielle NVIDIA avec CUDA et Python
# pour garantir la compatibilité avec vLLM et les GPU.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11

# --- 2. INSTALLATION DES DÉPENDANCES SYSTÈME & PYTHON ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-pip \
    python${PYTHON_VERSION}-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- 3. CONFIGURATION DE L'ENVIRONNEMENT DE TRAVAIL ---
WORKDIR /app

# Copie des requirements et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste de l'application
COPY . .

# --- 4. EXÉCUTION DE L'APPLICATION ---
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]