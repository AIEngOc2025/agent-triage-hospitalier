import gradio as gr
import requests

# URL de l'API (à adapter selon le déploiement)
API_URL = "http://localhost:7870/chat"


def chat_function(message, history):
    """
    @definition : Fonction appelée par Gradio pour gérer la conversation avec l'API.
    @args/params : message (str), history (list)
    @return : str (la réponse de l'assistant)
    """

    # Préparation du format historique pour l'API
    api_history = []
    for human, assistant in history:
        api_history.append({"role": "user", "content": human})
        api_history.append({"role": "assistant", "content": assistant})
    api_history.append({"role": "user", "content": message})

    payload = {"patient_id": "gradio_user", "history": api_history, "stream": False}

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["response"]
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
    demo.launch(server_name="0.0.0.0", server_port=7871)
