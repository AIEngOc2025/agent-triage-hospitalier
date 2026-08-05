import os
import uuid
from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncGenerator, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api_utils import clean_response, create_log_entry, log_audit
from app.core.settings import settings
from app.remote.client import RemoteInferenceClient
from app.system_prompts import SYSTEM_PROMPT_FR

# --- 2. ENGINE ABSTRACTION ---


class ModelEngine:
    """
    Client for remote LLM inference.
    """

    def __init__(self):
        self.client = None
        self.engine_type = "RemoteInference"

    def initialize(self):
        print(f"DEBUG: APP_ENV={settings.APP_ENV}")
        print(f"DEBUG: Initializing with model_name={settings.MODEL_PATH}")
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


# Global engine instancegc
engine = ModelEngine()


# --- 3. LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    @definition: Manages the application's lifespan events (startup and shutdown).
    @args/params:
        - app (FastAPI): The FastAPI application instance.
    @return: None (générateur asynchrone).
    """
    settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    engine.initialize()
    yield
    print("🛑 Arrêt de la tentative du chargement de l'orchestrateur")


# --- 4. API FASTAPI ---
app = FastAPI(title="CHSA AI Gateway", lifespan=lifespan)

if __name__ == "__main__":
    import uvicorn

    # Cloud Run defaults to 8080, ensure we respect it.
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)


@app.get("/health", status_code=200, tags=["Monitoring"])
async def health_check():
    """
    @definition: Provides a health check endpoint to verify service and model
    engine status.
    @return: A dictionary with the service status and engine type.
    """
    print(f"🩺 Health check called. Engine: {engine.engine_type}")
    return {"status": "ok", "engine": engine.engine_type}


class ChatRequest(BaseModel):
    patient_id: str = "PAT-001"
    history: List[dict]
    stream: bool = False


@app.post("/chat")
async def api_chat(request: ChatRequest):
    """
    @definition: Main endpoint for handling chat requests, supporting both
    streaming and non-streaming responses.
    @args/params:
        - request (ChatRequest): The incoming request containing patient ID and
        history.
    @return: A StreamingResponse or a JSON object with the model's response.
    """
    start_time = perf_counter()
    messages = request.history

    # --- Intercepteur de salutations pour démonstration (Bypass LLM) ---
    user_input = (
        next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        .strip()
        .lower()
    )

    if user_input in ["bonjour", "salut", "hello", "hi"]:
        response = (
            "Bonjour. Veuillez décrire vos symptômes ou votre situation médicale."
        )
        return {"response": response, "audit_ref": "demo-interceptor"}

    # Ensure a system prompt is always present
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT_FR})

    if request.stream:

        async def event_generator():
            try:
                full_response = []
                async for chunk in engine.generate_stream(messages, str(uuid.uuid4())):
                    full_response.append(chunk)
                    yield f"data: {chunk}\n\n"

                # Log completion of streaming request
                latency = perf_counter() - start_time
                user_input = next(
                    (m["content"] for m in reversed(messages) if m["role"] == "user"),
                    "",
                )
                log_entry = create_log_entry(
                    request.patient_id,
                    user_input,
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
        user_input = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        log_entry = create_log_entry(
            request.patient_id, user_input, response, latency, False
        )
        await log_audit(log_entry)
        return {"response": response, "audit_ref": log_entry["audit_id"]}
    except Exception as e:
        import traceback

        from httpx import HTTPStatusError

        if isinstance(e, HTTPStatusError):
            print(f"❌ HTTP Error: {e.response.text}")
        print(f"❌ Internal Server Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
