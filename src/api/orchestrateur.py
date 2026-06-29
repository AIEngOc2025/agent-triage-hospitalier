import os
import asyncio
import time
import uuid
import json
import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

# Importations natives vLLM
from vllm import LLM, SamplingParams

# --- 1. CONFIGURATION ---
# On utilise les chemins validés précédemment
MODEL_PATH = os.path.abspath("models/merged_dpo_final_chsa")
LOG_FILE = "logs/audit_medical.jsonl"
os.makedirs("logs", exist_ok=True)

# Variables globales pour le moteur
llm = None
sampling_params = None

# --- 2. LOGIQUE GÉNÉRATIVE (API LLM.CHAT) ---
async def run_triage_inference(messages: List[dict]) -> str:
    """Exécute l'inférence via l'API native llm.chat de vLLM."""
    global llm, sampling_params
    
    if llm is None:
        return "❌ Erreur : Moteur non initialisé."

    # Injection du prompt système pour garantir la concision clinique
    if not messages or messages[0]["role"] != "system":
        messages.insert(0, {
            "role": "system", 
            "content": "Tu es l'infirmier de triage du CHSA. Pose une question courte à la fois pour évaluer l'urgence (Maximale/Modérée/Différée). Sois bilingue."
        })

    # llm.chat est bloquant, on le délègue à un thread pour ne pas freezer FastAPI
    outputs = await asyncio.to_thread(
        llm.chat, 
        messages, 
        sampling_params, 
        use_tqdm=False
    )
    
    return outputs[0].outputs[0].text

# --- 3. LIFESPAN (Activation vLLM) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, sampling_params
    print(f"📥 [vLLM] Chargement du modèle Qwen3 sur puce M1...")
    
    try:
        # Initialisation conforme aux contraintes de ton Mac M1
        llm = LLM(model=MODEL_PATH)
        
        sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=512,
            repetition_penalty=1.15,
            stop=["<|im_end|>"]
        )
        print("✅ [vLLM] Moteur opérationnel.")
    except Exception as e:
        print(f"❌ [CRITICAL] Échec du démarrage : {e}")
    
    yield
    print("🛑 Arrêt de l'orchestrateur.")

# --- 4. API FASTAPI (Passerelle Hospitalière) ---
app = FastAPI(title="CHSA AI Gateway", lifespan=lifespan)

class ChatRequest(BaseModel):
    patient_id: str = "PAT-001"
    history: List[dict]

@app.post("/chat")
async def api_chat(request: ChatRequest):
    start_time = time.time()
    try:
        response = await run_triage_inference(request.history)
        
        # Traçabilité conforme (Livrable 5)
        log_entry = {
            "audit_id": str(uuid.uuid4()),
            "patient_id": request.patient_id,
            "decision": response,
            "latency_sec": round(time.time() - start_time, 3),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
        return {"assistant": response, "audit_ref": log_entry["audit_id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. UI GRADIO (Agent Conversationnel) ---
async def gradio_bridge(message, history):
    # Transformation Gradio -> Format ChatML pour vLLM
    formatted_history = []
    for msg in history:
        formatted_history.append({"role": msg["role"], "content": msg["content"]})
    formatted_history.append({"role": "user", "content": message})
    
    return await run_triage_inference(formatted_history)

demo = gr.ChatInterface(
    fn=gradio_bridge,
    #type="messages",
    title="🏥 CHSA - Assistant de Triage IA",
    description="Système conversationnel adaptatif de triage initial. Optimisé vLLM-Metal.",
    #theme="soft"
)

# Montage de Gradio sur FastAPI (accessible sur http://localhost:8001/)
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    # Lancement sur le port 8001 (Port 8000 réservé au moteur interne si besoin)
    uvicorn.run(app, host="127.0.0.1", port=8001)