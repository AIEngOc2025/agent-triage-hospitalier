import gradio as gr
import httpx

def chat_bridge(message, history):
    # On prépare l'historique pour l'API (Format Gradio 5.0 -> API)
    api_history = []
    for msg in history:
        api_history.append({"role": msg["role"], "content": msg["content"]})
    
    # Ajout du message actuel
    api_history.append({"role": "user", "content": message})

    try:
        # On appelle FastAPI sur le port 8001
        response = httpx.post("http://localhost:8001/chat", json={"history": api_history}, timeout=120.0)
        return response.json()["response"]
    except Exception as e:
        return f"Erreur : {e}"

demo = gr.ChatInterface(
    fn=chat_bridge,
    #type="messages",
    title="Agent de Triage CHSA",
    description="Interface directe vLLM Chat",
)

if __name__ == "__main__":
    demo.launch(share=True)