import os
import asyncio
import time
import uuid
import re
import html
import json
import httpx
import aiofiles
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

from ..config import (
    VLLM_BINARY,
    MODEL_PATH,
    LOG_FILE,
    VLLM_LOG_FILE,
    VLLM_PORT,
    API_PORT,
    VLLM_HOST,
    VLLM_SERVER_ARGS,
)

# Variables environnementales
os.environ.update(
    {
        "VLLM_TARGET_DEVICE": "mps",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "HF_HUB_OFFLINE": "1",
    }
)

vllm_process = None


@asynccontextmanager
async def noop_lifespan(app: FastAPI):
    yield


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vllm_process
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("📥 [vLLM] Lancement forcé sur GPU Apple Silicon...")
    # Redirige la sortie du serveur vLLM vers un fichier de log dédié
    vllm_log_file = open(VLLM_LOG_FILE, "w")
    vllm_process = await asyncio.create_subprocess_exec(
        VLLM_BINARY,
        "serve",
        *VLLM_SERVER_ARGS,
        stdout=vllm_log_file,
        stderr=vllm_log_file,
    )

    # Attente du moteur
    print("⏳ Attente de l'initialisation du moteur (timeout: 90s)...")
    start_time = time.time()
    timeout = 90  # secondes
    while time.time() - start_time < timeout:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"http://{VLLM_HOST}:{VLLM_PORT}/health")
                if resp.status_code == 200:
                    print("✅ [vLLM] Moteur prêt.")
                    break
        except (httpx.ConnectError, httpx.TimeoutException):
            await asyncio.sleep(2)
    else:
        raise RuntimeError("Le moteur vLLM n'a pas pu démarrer dans le temps imparti.")

    try:
        yield
    finally:
        if vllm_process and vllm_process.returncode is None:
            print("🛑 Arrêt du service.")
            try:
                vllm_process.terminate()
                await vllm_process.wait()
            except ProcessLookupError:
                pass
        vllm_log_file.close()


app = FastAPI(title="CHSA AI Agent Gateway", lifespan=lifespan)

if os.getenv("APP_TEST_MODE") == "1":
    app.router.lifespan_context = noop_lifespan


# --- LOGIQUE API ---
async def chat_relay(history: list, patient_id: str):
    payload = {
        "model": MODEL_PATH,  # Utilisation directe de la variable de configuration pour plus de robustesse
        # Injection du prompt système pour garantir la concision clinique
        "messages": [
            {
                "role": "system",
                "content": """Tu es un infirmier de triage pour le Centre Hospitalier Sud-Aveyron (CHSA).

**Instructions strictes :**
1.  **Présentation :** Commence TOUJOURS par te présenter et demander la raison de la venue.
2.  **Une seule question :** Pose UNE SEULE question courte et simple à la fois pour préciser les symptômes.""",
            }
        ]
        + history,
        "temperature": 0.3,
        "repetition_penalty": 1.15,
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"http://{VLLM_HOST}:{VLLM_PORT}/v1/chat/completions",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP (4xx, 5xx)
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            print(f"❌ Erreur de communication avec le moteur vLLM: {e}")
            raise HTTPException(
                status_code=503, detail="Le service d'inférence est indisponible."
            )

        result = response.json()
        # 1. Décodage initial des entités HTML (ex: &lt;)
        decoded_response = html.unescape(result["choices"][0]["message"]["content"])

        # 2. Nettoyage robuste en plusieurs étapes
        # Étape A: Supprimer les blocs de pensée <think>...</think>
        no_think_response = re.sub(
            r"<think>.*?</think>", "", decoded_response, flags=re.DOTALL
        )
        # Étape B: Supprimer toute autre balise XML/HTML restante pour ne garder que le texte brut
        clean_text = re.sub(r"<[^>]+>", "", no_think_response)

        ai_response = clean_text.strip()

        # Traçabilité (Audit Médical - Page 2 PDF)
        log_entry = {
            "audit_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "out": ai_response,
            "ts": time.time(),
        }
        async with aiofiles.open(LOG_FILE, "a") as f:
            await f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        return ai_response


# --- POINT DE TERMINAISON (ENDPOINT) API ---
class ChatRequest(BaseModel):
    history: List[dict]
    patient_id: str = "PAT-API-001"


@app.post("/chat")
async def api_chat(request: ChatRequest):
    response = await chat_relay(request.history, request.patient_id)
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=API_PORT, workers=1)
