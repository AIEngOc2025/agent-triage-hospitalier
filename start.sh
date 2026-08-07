#!/usr/bin/env bash

# --- Fonction de nettoyage ---
# S'assure que le serveur API est arrêté lorsque le script se termine.
cleanup() {
    echo -e "\n🛑 Arrêt du script. Nettoyage des processus..."
    if [ ! -z "$API_PID" ]; then
        # Tuer le groupe de processus pour s'assurer que les enfants (uvicorn) sont arrêtés
        kill -TERM -- "-$API_PID" 2>/dev/null
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

# --- Chargement de la configuration ---
# Si un fichier .env existe, on charge les variables qu'il contient.
if [ -f .env ]; then
    echo "📝 Chargement de la configuration depuis .env"
    set -a
    source .env
    set +a
fi

# --- Configuration & Validation ---
if [ -z "$MODEL_ID" ]; then
    echo "❌ ERREUR: La variable d'environnement MODEL_ID n'est pas définie."
    exit 1
fi

API_PORT=${API_PORT:-8000}
API_HOST=${API_HOST:-"localhost"}
API_BASE_URL="http://${API_HOST}:${API_PORT}"
LOG_FILE="triage_api.log"

# --- 1. Lancement du serveur API FastAPI ---
echo "🚀 Lancement de l'API FastAPI (logs dans ${LOG_FILE})..."
# 'set -m' active le contrôle des tâches, nécessaire pour tuer le groupe de processus.
# On utilise 'uv run' pour garantir l'utilisation de l'environnement virtuel du projet.
(set -m; uv run uvicorn app.main:app --host "${API_HOST}" --port "${API_PORT}" \
      > "$LOG_FILE" 2>&1 &)

API_PID=$!

# --- 2. Attente du serveur API ---
echo "⏳ Attente du démarrage de l'API..."
HEALTH_CHECK_URL="${API_BASE_URL}/health"

until curl -s --fail "$HEALTH_CHECK_URL" > /dev/null 2>&1; do
    # Vérifier si le processus API est toujours en cours d'exécution
    if ! kill -0 $API_PID 2>/dev/null; then
        echo "❌ ERREUR: Le serveur API a cessé de fonctionner."
        echo "   Consultez les logs : cat ${LOG_FILE}"
        cleanup
    fi
    echo "   ...attente..."
    sleep 1
done
echo "✅ API opérationnelle."

# --- 3. Lancement de l'interface Streamlit ---
echo "💻 Lancement de l'interface Streamlit..."
# L'UI pointera vers notre API
export API_BASE_URL="${API_BASE_URL}"
uv run streamlit run app/ui.py

# --- Fin du script ---
cleanup
