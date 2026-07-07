import asyncio
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import aiofiles
import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- 1. CONFIGURATION (Imports simplifiés) ---
from .config import (
    IS_PRODUCTION,
    IS_MACOS,
    MODEL_PATH,
    LOG_FILE,
    VLLM_MAX_MODEL_LEN,
    VLLM_TENSOR_PARALLEL_SIZE,
)

try:
    from vllm import LLM, SamplingParams
except ImportError:
    # Permet au code de s'exécuter même si vllm n'est pas installé,
    # par exemple pour des tests unitaires ou du linting.
    LLM, SamplingParams = None, None

# Variables globales pour le moteur
llm = None
sampling_params = None


def clean_llm_response(text: str) -> str:
    """Nettoie la sortie du LLM en supprimant les balises de pensée."""
    # Supprime les blocs <think>...</think> et toute autre balise XML/HTML
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    clean_text = re.sub(r"<[^>]+>", "", clean_text)
    return clean_text.strip()


def patch_model_config_if_needed():
    """
    Vérifie et corrige le fichier config.json du modèle local pour assurer la
    compatibilité avec vLLM sur Metal (MLX), qui requiert 'rope_theta'.
    """
    config_path = Path(MODEL_PATH) / "config.json"
    if not config_path.is_file():
        print(f"⚠️  Avertissement : Fichier de configuration non trouvé à {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    if "rope_theta" not in config:
        print("🔧 [Patch] Ajout du paramètre 'rope_theta' manquant dans config.json...")
        config["rope_theta"] = 1000000.0  # Valeur standard pour Qwen2/Llama3
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("✅ [Patch] Fichier config.json mis à jour.")

def patch_tokenizer_config_if_needed():
    """
    Vérifie et corrige le fichier tokenizer_config.json pour forcer l'utilisation
    du tokenizer 'Fast' (Rust), qui est plus robuste avec vLLM.
    """
    config_path = Path(MODEL_PATH) / "tokenizer_config.json"
    if not config_path.is_file():
        print(f"⚠️  Avertissement : Fichier tokenizer_config.json non trouvé à {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # vLLM préfère la version "Fast" du tokenizer.
    if config.get("tokenizer_class") != "Qwen2TokenizerFast":
        print("🔧 [Patch] Forçage de l'utilisation de Qwen2TokenizerFast dans tokenizer_config.json...")
        config["tokenizer_class"] = "Qwen2TokenizerFast"
        # Assure la présence du fichier tokenizer.json, essentiel pour le tokenizer rapide.
        if not (Path(MODEL_PATH) / "tokenizer.json").is_file():
            print("❌ [CRITICAL] tokenizer.json est manquant ! Le tokenizer rapide ne fonctionnera pas.")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("✅ [Patch] Fichier tokenizer_config.json mis à jour.")

# --- 2. LOGIQUE GÉNÉRATIVE (API LLM.CHAT) ---
async def run_triage_inference(messages: List[dict]) -> str:
    """Exécute l'inférence via l'API native llm.chat de vLLM."""
    global llm, sampling_params

    if llm is None:
        return "❌ Erreur : Moteur non initialisé."

    # llm.chat est bloquant, on le délègue à un thread pour ne pas freezer FastAPI
    outputs = await asyncio.to_thread(
        llm.chat, messages, sampling_params, use_tqdm=False
    )

    raw_response = outputs[0].outputs[0].text
    return clean_llm_response(raw_response)


# --- 3. LIFESPAN (Activation vLLM) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, sampling_params

    # Crée le dossier de logs s'il n'existe pas
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # Patch pour la compatibilité vLLM-Metal (local dev sur Mac)
    if not IS_PRODUCTION and IS_MACOS:
        patch_model_config_if_needed()
        patch_tokenizer_config_if_needed()

    if LLM:
        print(f"📥 [vLLM] Chargement du modèle depuis : {MODEL_PATH}")

        # On initialise le moteur vLLM en lui spécifiant explicitement le chemin
        # du modèle ET du tokenizer. C'est la méthode la plus fiable pour
        # s'assurer que vLLM charge les bons composants, surtout après un patch.
        llm = LLM(
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,  # Ajout crucial pour la stabilité
            max_model_len=VLLM_MAX_MODEL_LEN,
            trust_remote_code=True,  # Nécessaire pour les modèles locaux/custom
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,  # Pour déploiement multi-GPU
            # Sur Mac (mémoire unifiée), on réserve moins de mémoire pour laisser de la
            # place à l'OS et au cache KV, évitant les erreurs "out-of-memory".
            gpu_memory_utilization=0.80
        )
        sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=512,
            repetition_penalty=1.15,
            stop=["<|im_end|>"],
        )
        # "Préchauffage" du modèle pour réduire la latence de la première requête
        print("🔥 [vLLM] Préchauffage du modèle...")
        await asyncio.to_thread(
            llm.generate, "Bonjour", sampling_params, use_tqdm=False
        )
        print("✅ [vLLM] Moteur opérationnel.")
    else:
        print("❌ [CRITICAL] vLLM n'est pas installé. L'inférence ne fonctionnera pas.")

    yield
    print("🛑 Arrêt de l'orchestrateur.")


# --- 4. API FASTAPI (Passerelle Hospitalière) ---
app = FastAPI(title="CHSA AI Gateway", lifespan=lifespan)


@app.get("/health", status_code=200, tags=["Monitoring"])
async def health_check():
    """Vérifie que le service est opérationnel."""
    return {"status": "ok"}


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
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
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
    description="Système conversationnel adaptatif de triage initial. Optimisé vLLM-Metal.",
)

# Montage de Gradio sur FastAPI (accessible sur http://localhost:8001/)
gr.mount_gradio_app(app, demo, path="/")
