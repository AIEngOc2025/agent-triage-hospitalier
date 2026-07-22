import gradio as gr
import requests
import os
import sys

# --- Path Correction ---
# This adds the project's root directory to the Python path.
# It allows the script to be run directly while resolving imports from the 'app' package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
# L'URL de base de l'API VLLM est fournie par le script de démarrage via une variable d'environnement.
# Si elle n'est pas définie, on utilise une valeur par défaut pour le développement local.
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
CHAT_COMPLETIONS_URL = f"{VLLM_API_BASE}/chat/completions"


def chat_function(message, history):
    """
    @definition : Fonction appelée par Gradio pour gérer la conversation avec l'API VLLM.
    @args/params : message (str), history (list)
    @return : str (la réponse de l'assistant)
    """

    # Préparation du format historique pour l'API OpenAI
    api_history = []
    for human, assistant in history:
        api_history.append({"role": "user", "content": human})
        api_history.append({"role": "assistant", "content": assistant})
    api_history.append({"role": "user", "content": message})

    # Le modèle est défini par le serveur VLLM, pas besoin de le spécifier ici.
    payload = {
        "messages": api_history,
        "stream": False
    }

    try:
        # On appelle l'endpoint de chat completions du serveur VLLM local
        response = requests.post(CHAT_COMPLETIONS_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Erreur de connexion à l'API : {str(e)}"


# Interface Gradio
demo = gr.ChatInterface(
    fn=chat_function,
    title="Agent de Triage CHSA",
    description="Assistant infirmier de triage médical. "
    "Veuillez décrire vos symptômes.",
)

if __name__ == "__main__":
    # --- Argument Parsing ---
    # Permet de passer le port et l'hôte depuis la ligne de commande,
    # ce qui est essentiel pour Cloud Run.
    parser = argparse.ArgumentParser(description="Lancement de l'interface Gradio")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Adresse du serveur")
    parser.add_argument("--server-port", type=int, default=7860, help="Port du serveur")
    args = parser.parse_args()

    # Lancement de l'interface avec les arguments fournis
    demo.launch(server_name=args.server_name, server_port=args.server_port)
