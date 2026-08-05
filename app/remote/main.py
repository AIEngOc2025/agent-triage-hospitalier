import uuid
from contextlib import asynccontextmanager
from time import perf_counter
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api_utils import clean_response, create_log_entry, log_audit
from app.remote.client import RemoteInferenceClient
from app.settings import settings
from app.system_prompts import SYSTEM_PROMPT_FR


# --- API Gateway Engine Wrapper ---
class RemoteEngine:
    """
    Gateway wrapper that only knows how to talk to a remote inference service.
    """

    def __init__(self):
        self.client = None
        self.engine_type = "RemoteInference"

    def initialize(self):
        print(f"DEBUG: APP_ENV={settings.APP_ENV}")
        print("🌐 [REMOTE] Initializing remote inference client...")
        self.client = RemoteInferenceClient()

    async def generate_stream(self, messages: List[dict], request_id: str):
        async for chunk in self.client.generate_stream(messages):
            yield chunk

    async def generate(self, messages: List[dict]) -> str:
        response = await self.client.generate(messages)
        return clean_response(response)


# --- LIFESPAN ---
engine = RemoteEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    engine.initialize()
    yield
    print("🛑 Arrêt de l'API Gateway")


# --- APP ---
app = FastAPI(title="CHSA AI Gateway", lifespan=lifespan)


class ChatRequest(BaseModel):
    patient_id: str = "PAT-001"
    history: List[dict]
    stream: bool = False


@app.post("/chat")
async def api_chat(request: ChatRequest):
    start_time = perf_counter()
    messages = request.history
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT_FR})

    if request.stream:

        async def event_generator():
            try:
                full_response = []
                async for chunk in engine.generate_stream(messages, str(uuid.uuid4())):
                    full_response.append(chunk)
                    yield f"data: {chunk}\n\n"
                latency = perf_counter() - start_time
                user_input = messages[-1]["content"] if messages else ""
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
        user_input = messages[-1]["content"] if messages else ""
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
