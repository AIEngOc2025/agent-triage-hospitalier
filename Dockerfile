# --- 1. IMAGE DE BASE ---
# Utilisation d'une image Python slim pour une taille optimisée.
FROM python:3.11-slim

# --- 2. Configuration de l'environnement ---
WORKDIR /app
ENV APP_ENV=production
ENV PYTHONUNBUFFERED=1

# --- 3. INSTALLATION DES DÉPENDANCES ---
# On utilise uv pour une installation plus rapide.
RUN pip install -U uv

# Copie et installation des dépendances en premier pour profiter du cache Docker.
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# --- 4. Copie du code de l'application ---
# Copie uniquement le code de l'application et les modèles nécessaires.
COPY ./app ./app
COPY ./models/merged_dpo_final_chsa ./models/merged_dpo_final_chsa

# --- 5. Exposition du port ---
# Le port 7860 est le port par défaut pour les Spaces Gradio/FastAPI.
EXPOSE 7860

# --- 6. Commande de démarrage ---
# Lancer l'application FastAPI avec uvicorn.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
