import traceback
import uuid
from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncGenerator, List

from fastapi import FastAPI, HTTPException
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
    """Client polymorphe pour l'inférence distante (vLLM OpenAI-compatible).

    Supporte deux modes :
    - `conversationnel` (par défaut) : pour le chat libre, via httpx brut
    - `structured` : pour le triage structuré, via instructor + Pydantic
    """

    def __init__(self):
        self.client = None
        self.engine_type = "RemoteInference"

    def initialize(self):
        print("🌐 [REMOTE] Initializing remote inference client...")
        self.client = RemoteInferenceClient()

    async def generate_stream(
        self, messages: List[dict], request_id: str
    ) -> AsyncGenerator[str, None]:
        async for chunk in self.client.generate_stream(messages):
            yield chunk

    async def generate(self, messages: List[dict]) -> str:
        response = await self.client.generate(messages)
        return clean_response(response)

    async def generate_structured(self, messages: List[dict]) -> TriageResponse:
        """Génère une réponse typée TriageResponse via instructor.

        Le system prompt doit être en tête de `messages` pour garantir
        la conformité au contrat JSON.
        """
        return await self.client.generate_structured(messages)


engine = ModelEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    engine.initialize()
    yield
    if engine.client is not None:
        await engine.client.close()
    print("🛑 Shutdown: remote inference client released")


app = FastAPI(title="CHSA AI Gateway", lifespan=lifespan)


@app.get("/health", status_code=200, tags=["Monitoring"])
async def health_check():
    return {"status": "ok", "engine": engine.engine_type}


class ChatRequest(BaseModel):
    patient_id: str = Field(
        ..., pattern=r"^PAT-\d{3,}$", description="Patient identifier (format: PAT-XXX)"
    )
    history: List[dict] = Field(..., min_length=1, max_length=50)
    stream: bool = False


class TriageRequest(BaseModel):
    patient_id: str = Field(
        ..., pattern=r"^PAT-\d{3,}$", description="Patient identifier (format: PAT-XXX)"
    )
    history: List[dict] = Field(..., min_length=1, max_length=50)


def _extract_user_input(messages: List[dict]) -> str:
    """Extrait le dernier message user de l'historique (helper partagé)."""
    return next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        "",
    )


def _ensure_system_prompt(messages: List[dict]) -> List[dict]:
    """Garantit la présence du system prompt en tête (sans muter l'input)."""
    if messages[0].get("role") != "system":
        return [{"role": "system", "content": SYSTEM_PROMPT_FR}] + list(messages)
    return list(messages)


@app.post("/chat")
async def api_chat(request: ChatRequest):
    """Endpoint conversationnel : génère une réponse texte libre.

    Supporte streaming (SSE) et non-streaming. La réponse est nettoyée
    (`clean_response`) pour supprimer les balises internes (`think`, etc.).
    """
    start_time = perf_counter()
    messages = _ensure_system_prompt(request.history)

    if request.stream:

        async def event_generator():
            try:
                full_response = []
                async for chunk in engine.generate_stream(messages, str(uuid.uuid4())):
                    full_response.append(chunk)
                    yield f"data: {chunk}\n\n"

                latency = perf_counter() - start_time
                log_entry = create_log_entry(
                    request.patient_id,
                    _extract_user_input(messages),
                    "".join(full_response),
                    latency,
                    True,
                )
                await log_audit(log_entry)
            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    try:
        response = await engine.generate(messages)
        latency = perf_counter() - start_time
        log_entry = create_log_entry(
            request.patient_id,
            _extract_user_input(messages),
            response,
            latency,
            False,
        )
        await log_audit(log_entry)
        return {"response": response, "audit_ref": log_entry["audit_id"]}
    except HTTPStatusError as e:
        print(f"❌ HTTP Error: {e.response.text}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        print(f"❌ Internal Server Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/triage", response_model=None)
async def api_triage(request: TriageRequest):
    """Endpoint structuré : génère une `TriageResponse` typée.

    Utilise `instructor.from_openai` (mode MD_JSON) côté client pour
    valider la sortie contre le schema Pydantic. Le résultat est
    sérialisé en JSON via `.model_dump()`.

    Le format de réponse est :
        {
            "message": "...",
            "triage_result": {"niveau": "maximale|modérée|différée" | null,
                              "orientation": "..." | null},
            "audit_ref": "..."
        }
    """
    start_time = perf_counter()
    messages = _ensure_system_prompt(request.history)

    try:
        result = await engine.generate_structured(messages)
        latency = perf_counter() - start_time

        # Sérialisation Pydantic -> dict
        log_entry = create_log_entry(
            request.patient_id,
            _extract_user_input(messages),
            result.model_dump_json(),
            latency,
            False,
        )
        await log_audit(log_entry)
        return {
            **result.model_dump(),
            "audit_ref": log_entry["audit_id"],
        }
    except Exception as e:
        print(f"❌ Structured triage failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
