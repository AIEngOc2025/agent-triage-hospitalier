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

VLLM_PORT=${VLLM_PORT:-8000}
VLLM_HOST=${VLLM_HOST:-"localhost"}
VLLM_API_BASE="http://${VLLM_HOST}:${VLLM_PORT}"

# Construire les arguments pour VLLM
VLLM_ARGS=("--trust-remote-code")

# Sur macOS, ajouter le drapeau pour activer le support Metal (MLX)
if [[ "$(uname)" == "Darwin" ]]; then
    echo "🍏 macOS détecté, activation du support Metal pour VLLM."
    VLLM_ARGS+=(
        "--device" "metal"
        "--gpu-memory-utilization" "0.80" # Limite l'usage à 80% de la mémoire unifiée
    )
fi

# --- 1. Lancement du serveur VLLM ---
echo "🚀 Lancement du serveur VLLM en arrière-plan (logs dans ${VLLM_LOG_FILE})..."
# 'set -m' active le contrôle des tâches, nécessaire pour tuer le groupe de processus.
(set -m; python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_ID" \
      --host "$VLLM_HOST" \
      --port "$VLLM_PORT" \
      "${VLLM_ARGS[@]}" \
      > "$VLLM_LOG_FILE" 2>&1 &)

# Sauvegarder le PID (Process ID) du serveur VLLM
VLLM_PID=$!

# --- 2. Attente du serveur VLLM ---
echo "⏳ Attente du démarrage du serveur VLLM..."
HEALTH_CHECK_URL="${VLLM_API_BASE}/health"
MAX_WAIT_SECONDS=120
SECONDS=0

until curl -s --fail "$HEALTH_CHECK_URL" > /dev/null; do
    # Vérifier si le processus VLLM est toujours en cours d'exécution
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ ERREUR: Le serveur VLLM a cessé de fonctionner."
        echo "   Veuillez consulter les logs pour plus de détails : cat ${VLLM_LOG_FILE}"
        cleanup
    fi

    echo "   Serveur VLLM non disponible, nouvelle tentative dans 2s..."
    sleep 2
done
echo "✅ Serveur VLLM opérationnel."

# --- 3. Lancement de l'interface Gradio ---
echo "💻 Lancement de l'interface Gradio..."
export VLLM_API_BASE="${VLLM_API_BASE}/v1"
python app/ui.py

# --- Fin du script ---
# Si l'application Gradio se termine, on nettoie le processus VLLM.
echo "L'interface Gradio s'est terminée."
cleanup