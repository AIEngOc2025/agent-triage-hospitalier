import traceback
import uuid
from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncGenerator, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from httpx import HTTPStatusError
from pydantic import BaseModel, Field

from app.api_utils import clean_response, create_log_entry, log_audit
from app.core.settings import settings
from app.remote.client import RemoteInferenceClient
from app.schemas import TriageResponse
from app.system_prompts import SYSTEM_PROMPT_FR

# --- ENGINE ABSTRACTION ---
class ModelEngine:
    def __init__(self):
        self.client = None
        self.engine_type = "RemoteInference"

    def initialize(self):
        # Initialisation du client unique (Pool de connexion maintenu)
        self.client = RemoteInferenceClient()

    async def generate_stream(self, messages: List[dict]) -> AsyncGenerator[str, None]:
        async for chunk in self.client.generate_stream(messages):
            yield chunk

    async def generate(self, messages: List[dict]) -> str:
        # On délègue le nettoyage au client si possible, ou on le garde ici
        response = await self.client.generate(messages)
        return clean_response(response)

    async def generate_structured(self, messages: List[dict]) -> TriageResponse:
        return await self.client.generate_structured(messages)

engine = ModelEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    engine.initialize()
    yield
    if engine.client is not None:
        await engine.client.close()

app = FastAPI(title="CHSA AI Gateway", lifespan=lifespan)

# --- SCHEMAS ---
class ChatRequest(BaseModel):
    patient_id: str = Field(..., pattern=r"^(PAT-\d{3,}|conv-user)$")
    history: List[dict] = Field(..., min_length=1, max_length=50)
    stream: bool = False

class TriageRequest(BaseModel):
    patient_id: str = Field(..., pattern=r"^(PAT-\d{3,}|conv-user)$")
    history: List[dict] = Field(..., min_length=1, max_length=50)

# --- UTILS ---
def _ensure_system_prompt(messages: List[dict]) -> List[dict]:
    """Plus rapide : évite la mutation et réduit les vérifications."""
    if not messages or messages[0].get("role") != "system":
        return [{"role": "system", "content": SYSTEM_PROMPT_FR}] + messages
    return messages

# --- ROUTES ---

@app.post("/chat")
async def api_chat(request: ChatRequest, background_tasks: BackgroundTasks):
    start_time = perf_counter()
    messages = _ensure_system_prompt(request.history)

    if request.stream:
        async def event_generator():
            full_content = []
            try:
                async for chunk in engine.generate_stream(messages):
                    full_content.append(chunk)
                    yield f"data: {chunk}\n\n"
                
                # Audit lancé APRÈS que le stream soit fini, sans bloquer la connexion
                latency = perf_counter() - start_time
                log_entry = create_log_entry(
                    request.patient_id, 
                    messages[-1]["content"], 
                    "".join(full_content), 
                    latency, True
                )
                background_tasks.add_task(log_audit, log_entry)
            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    try:
        response = await engine.generate(messages)
        latency = perf_counter() - start_time
        
        # On prépare le log
        log_entry = create_log_entry(
            request.patient_id, messages[-1]["content"], response, latency, False
        )
        # On répond TOUT DE SUITE, l'audit se fera juste après
        background_tasks.add_task(log_audit, log_entry)
        
        return {"response": response, "audit_ref": log_entry["audit_id"]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/triage", response_model=TriageResponse)
async def api_triage(request: TriageRequest, background_tasks: BackgroundTasks):
    """Optimisé pour réduire le temps de sérialisation JSON."""
    start_time = perf_counter()
    messages = _ensure_system_prompt(request.history)

    try:
        result = await engine.generate_structured(messages)
        latency = perf_counter() - start_time

        # Audit en arrière-plan
        log_entry = create_log_entry(
            request.patient_id, 
            messages[-1]["content"], 
            result.model_dump_json(), 
            latency, False
        )
        background_tasks.add_task(log_audit, log_entry)
        
        # Ajout manuel de l'audit_ref au modèle sans re-sérialisation complète
        response_data = result.model_dump()
        response_data["audit_ref"] = log_entry["audit_id"]
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))