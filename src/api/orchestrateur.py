import os
import asyncio
import sys
import time
import uuid
import json
import re
import gradio as gr
import aiofiles
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

# --- 1. CONFIGURATION (Imports simplifiés) ---
from .config import MODEL_PATH, LOG_FILE
try:
    from vllm import LLM, SamplingParams
except ImportError:
    # Permet au code de s'exécuter même si vllm n'est pas installé,
    # par exemple pour des tests unitaires ou du linting.
    LLM, SamplingParams = None, None

# Détection de l'environnement pour l'accélération matérielle
IS_MACOS = sys.platform == "darwin"

# Variables globales pour le moteur
llm = None
sampling_params = None

def clean_llm_response(text: str) -> str:
    """Nettoie la sortie du LLM en supprimant les balises de pensée."""
    # Supprime les blocs <think>...</think> et toute autre balise XML/HTML
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    return clean_text.strip()

# --- 2. LOGIQUE GÉNÉRATIVE (API LLM.CHAT) ---
async def run_triage_inference(messages: List[dict]) -> str:
    """Exécute l'inférence via l'API native llm.chat de vLLM."""
    global llm, sampling_params
    
    if llm is None:
        return "❌ Erreur : Moteur non initialisé."

    # llm.chat est bloquant, on le délègue à un thread pour ne pas freezer FastAPI
    outputs = await asyncio.to_thread(
        llm.chat, 
        messages, 
        sampling_params, 
        use_tqdm=False
    )
    
    raw_response = outputs[0].outputs[0].text
    return clean_llm_response(raw_response)

# --- 3. LIFESPAN (Activation vLLM) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, sampling_params
    
    # Crée le dossier de logs s'il n'existe pas
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    if LLM:
        print(f"📥 [vLLM] Chargement du modèle depuis : {MODEL_PATH}")
        # Pour Hugging Face, le modèle est chargé en mémoire.
        # Pour le local sur Mac, vLLM détecte 'mps' (Metal) automatiquement.
        # On retire les arguments spécifiques pour laisser vLLM auto-configurer.
        llm = LLM(
            model=MODEL_PATH,
            max_model_len=4096,  # Limite la longueur pour éviter les erreurs de mémoire
            trust_remote_code=True  # Nécessaire pour les modèles locaux/custom
        )
        sampling_params = SamplingParams(
            temperature=0.2, 
            max_tokens=512, 
            repetition_penalty=1.15, 
            stop=["<|im_end|>"]
        )
        print("✅ [vLLM] Moteur opérationnel.")
    else:
        print("❌ [CRITICAL] vLLM n'est pas installé. L'inférence ne fonctionnera pas.")
    
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
    
    # Injection du prompt système uniquement si l'historique est vide
    messages = request.history
    if not messages or messages[0].get("role") != "system":
        system_prompt = """Tu es un infirmier de triage pour le Centre Hospitalier Sud-Aveyron (CHSA).

**Instructions strictes :**
1.  **Présentation :** Commence TOUJOURS par te présenter et demander la raison de la venue.
2.  **Une seule question :** Pose UNE SEULE question courte et simple à la fois pour préciser les symptômes.
3.  **Rôle limité :** Ne donne JAMAIS de diagnostic, d'explication, de conseil ou de niveau d'urgence. Ton unique objectif est de poser la question suivante pour recueillir de l'information.
4.  **Bilinguisme :** Réponds en français ou en anglais selon la langue de l'utilisateur.
5.  **Anti-Exemple :** Ne génère JAMAIS de cas cliniques ou de questions à choix multiples. Tu dois converser naturellement."""
        messages.insert(0, {"role": "system", "content": system_prompt})

    try:
        response = await run_triage_inference(messages)
        
        # Traçabilité conforme (Livrable 5)
        log_entry = {
            "audit_id": str(uuid.uuid4()),
            "patient_id": request.patient_id,
            "decision": response,
            "latency_sec": round(time.time() - start_time, 3),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        async with aiofiles.open(LOG_FILE, "a") as f:
            await f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
        return {"assistant": response, "audit_ref": log_entry["audit_id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. UI GRADIO (Agent Conversationnel) ---
async def gradio_bridge(message, history):
    """
    Fait le pont entre l'interface Gradio et le point de terminaison /chat de l'API FastAPI.
    Cela garantit que l'UI utilise la même logique (prompt système, logging) que l'API.
    """
    formatted_history = []
    for user_msg, assistant_msg in history:
        formatted_history.append({"role": "user", "content": user_msg})
        if assistant_msg:
            formatted_history.append({"role": "assistant", "content": assistant_msg})
        
    formatted_history.append({"role": "user", "content": message})

    # Appel interne à l'API FastAPI pour centraliser la logique
    response = await api_chat(ChatRequest(history=formatted_history))
    return response["assistant"]

demo = gr.ChatInterface(
    fn=gradio_bridge,
    title="🏥 CHSA - Assistant de Triage IA",
    description="Système conversationnel adaptatif de triage initial. Optimisé vLLM-Metal."
)

# Montage de Gradio sur FastAPI (accessible sur http://localhost:8001/)
gr.mount_gradio_app(app, demo, path="/")