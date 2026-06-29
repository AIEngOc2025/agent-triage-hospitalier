import os, sys, asyncio, time, uuid, json, httpx, gradio as gr, uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from pathlib import Path

# --- CONFIGURATION ---
VENV_PYTHON = "/Users/mpaga/.venv-vllm-metal/bin/python"
VLLM_BINARY = "/Users/mpaga/.venv-vllm-metal/bin/vllm"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(BASE_DIR / "models" / "merged_dpo_final_chsa")
LOG_FILE = str(BASE_DIR / "logs" / "audit_medical.jsonl")

VLLM_PORT = 8003
API_PORT = 8004

# Variables environnementales
os.environ.update({
    "VLLM_TARGET_DEVICE": "mps",
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "HF_HUB_OFFLINE": "1"
})

if sys.executable != VENV_PYTHON:
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

vllm_process = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vllm_process
    Path(LOG_FILE).parent.mkdir(exist_ok=True)
    
    print(f"📥 [vLLM] Lancement forcé sur GPU Apple Silicon...")
    
    vllm_process = await asyncio.create_subprocess_exec(
        VLLM_BINARY, "serve", MODEL_PATH,
        "--port", str(VLLM_PORT),
        "--host", "127.0.0.1",
        stdout=None, stderr=None 
    )
    
    # Attente du moteur
    print("⏳ Attente de l'initialisation du moteur (environ 60s)...")
    for i in range(10):
        await asyncio.sleep(3)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"http://127.0.0.1:{VLLM_PORT}/health")
                if resp.status_code == 200:
                    print("✅ [vLLM] Moteur prêt.")
                    break
        except: pass
            
    yield
    if vllm_process:
        print("🛑 Arrêt du service.")
        try:
            vllm_process.terminate()
            await vllm_process.wait()
        except: pass

app = FastAPI(title="CHSA AI Agent Gateway", lifespan=lifespan)

# --- LOGIQUE API ---
async def chat_relay(history: list, patient_id: str):
    payload = {
        "model": MODEL_PATH,
        "messages": history,
        "temperature": 0.3,
        "repetition_penalty": 1.15
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions", json=payload, timeout=120.0)
        result = response.json()
        ai_response = result["choices"][0]["message"]["content"]
        
        # Traçabilité (Audit Médical - Page 2 PDF)
        log = {"audit_id": str(uuid.uuid4()), "patient_id": patient_id, "out": ai_response, "ts": time.time()}
        with open(LOG_FILE, "a") as f: f.write(json.dumps(log, ensure_ascii=False) + "\n")
        return ai_response

# --- INTERFACE GRADIO ---
async def gradio_chat(message, history):
    api_history = [{"role": "system", "content": "Tu es l'infirmier du CHSA. Sois concis."}]
    for msg in history:
        content = msg["content"]
        if isinstance(content, list): content = content[0]["text"]
        api_history.append({"role": msg["role"], "content": content})
    api_history.append({"role": "user", "content": message})
    
    try:
        return await chat_relay(api_history, "PAT-001")
    except Exception as e:
        return f"⏳ Moteur en chargement... ({e})"

demo = gr.ChatInterface(fn=gradio_chat,  
        title="🏥 Agent Triage CHSA"
)
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=API_PORT, workers=1)