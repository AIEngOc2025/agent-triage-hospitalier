#!/usr/bin/env bash

# --- Fonction de nettoyage ---
# S'assure que le serveur VLLM est arrêté lorsque le script se termine.
cleanup() {
    echo -e "\n🛑 Arrêt du script. Nettoyage des processus en arrière-plan..."
    # -PUPID tue tout le groupe de processus, ce qui est plus robuste.
    [ ! -z "$VLLM_PID" ] && kill -TERM -- "-$VLLM_PID" 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# --- Activation de l'environnement virtuel local ---
# Si un environnement virtuel spécifique pour vllm-metal existe, on l'active.
# Cela permet d'utiliser la bonne version de Python et les bons packages en local.
VENV_ACTIVATE_PATH="$HOME/.venv-vllm-metal/bin/activate"
if [ -f "$VENV_ACTIVATE_PATH" ]; then
    echo "🐍 Activation de l'environnement virtuel local : $VENV_ACTIVATE_PATH"
    source "$VENV_ACTIVATE_PATH"
fi

# Fichier pour capturer les logs du serveur VLLM
VLLM_LOG_FILE="vllm.log"

# --- Chargement de la configuration locale ---
# Si un fichier .env existe, on charge les variables qu'il contient.
if [ -f .env ]; then
    echo "📝 Chargement de la configuration depuis le fichier .env"
    # 'set -a' exporte toutes les variables qui sont assignées.
    set -a
    source .env
    set +a
fi

# --- Configuration & Validation ---
if [ -z "$MODEL_ID" ]; then
    echo "❌ ERREUR: La variable d'environnement MODEL_ID n'est pas définie."
    echo "   Veuillez la définir avant de lancer le script, par exemple :"
    echo "   export MODEL_ID=\"models/merged_dpo_final_chsa\""
    exit 1
fi

API_PORT=${API_PORT:-8000}
API_HOST=${API_HOST:-"localhost"}
VLLM_API_BASE="http://${API_HOST}:${API_PORT}" # L'UI pointera vers notre API mockée

# --- 1. Lancement du serveur API Mocké ---
echo "🚀 Lancement du serveur API mocké en arrière-plan (logs dans ${VLLM_LOG_FILE})..."
# 'set -m' active le contrôle des tâches, nécessaire pour tuer le groupe de processus.
(set -m; uvicorn app.main:app --host "${API_HOST}" --port "${API_PORT}" \
      > "$VLLM_LOG_FILE" 2>&1 &)

# Sauvegarder le PID (Process ID) du serveur API
VLLM_PID=$!

# --- 2. Attente du serveur API ---
echo "⏳ Attente du démarrage du serveur API..."
HEALTH_CHECK_URL="${VLLM_API_BASE}/health"

until curl -s --fail "$HEALTH_CHECK_URL" > /dev/null 2>&1; do
    # Vérifier si le processus API est toujours en cours d'exécution
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ ERREUR: Le serveur API a cessé de fonctionner."
        echo "   Veuillez consulter les logs pour plus de détails : cat ${VLLM_LOG_FILE}"
        cleanup
    fi

    echo "   Serveur API non disponible, nouvelle tentative dans 2s..."
    sleep 2
done
echo "✅ Serveur API mocké opérationnel."

# --- 3. Lancement de l'interface Gradio ---
echo "💻 Lancement de l'interface Gradio..."
export VLLM_API_BASE="${VLLM_API_BASE}" # L'URL de base pour l'UI
python app/ui.py

# --- Fin du script ---
# Si l'application Gradio se termine, on nettoie le processus VLLM.
echo "L'interface Gradio s'est terminée."
cleanup