import asyncio
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from time import perf_counter, strftime
from typing import AsyncGenerator, Dict, List

import httpx
import spacy
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.settings import settings
from app.system_prompts import SYSTEM_PROMPT_FR

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine

# --- 2. ENGINE ABSTRACTION ---

class ModelEngine:
    def __init__(self):
        self.engine_type = None
        self.client = httpx.AsyncClient(timeout=120.0)

    def initialize(self):
        self.inference_url = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8000")
        self.engine_type = "RemoteVLLM"

    async def generate_stream(
        self, messages: List[dict], request_id: str
    ) -> AsyncGenerator[str, None]:
        async with self.client.stream(
            "POST",
            f"{self.inference_url}/v1/chat/completions",
            json={"messages": messages, "stream": True},
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    yield line

    async def generate(self, messages: List[dict]) -> str:
        response = await self.client.post(
            f"{self.inference_url}/v1/chat/completions",
            json={"messages": messages, "stream": False},
        )
        response.raise_for_status()
        return self.clean_response(
            response.json()["choices"][0]["message"]["content"]
        )

    def clean_response(self, text: str) -> str:
        """
        @definition: Removes specific tags (like <think>) and any other
        HTML-like tags from the model's output.
        @args/params:
            - text (str): The raw text from the model.
        @return: The cleaned text string.
        """
        clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        clean_text = re.sub(r"<[^>]+>", "", clean_text)
        return clean_text.strip()


# --- UTILITIES ---

analyzer = None
anonymizer = AnonymizerEngine()

def anonymize_text(text: str) -> str:
    global analyzer
    if analyzer is None:
        nlp = spacy.load("fr_core_news_sm")
        nlp_engine = SpacyNlpEngine(models={"fr": nlp})
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    results = analyzer.analyze(text=text, language="fr")
    anonymized_result = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized_result.text

async def log_audit(entry: dict):
    try:
        entry["decision"] = anonymize_text(entry["decision"])
        def write_log():
            with open(settings.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        await asyncio.to_thread(write_log)
    except Exception as e:
        print(f"❌ Audit logging failed: {e}")

def create_log_entry(patient_id: str, decision: str, latency: float, stream: bool) -> Dict:
    return {
        "audit_id": str(uuid.uuid4()),
        "patient_id": patient_id,
        "decision": decision,
        "latency_sec": round(latency, 3),
        "timestamp": strftime("%Y-%m-%d %H:%M:%S"),
        "stream": stream,
    }

engine = ModelEngine()

# --- 3. LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    engine.initialize()
    yield

# --- 4. API FASTAPI ---
app = FastAPI(title="CHSA AI Gateway", lifespan=lifespan)

# Route de santé
@app.get("/health", status_code=200, tags=["Monitoring"])
async def health_check():
    return {"status": "ok", "engine": engine.engine_type}

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
                log_entry = create_log_entry(request.patient_id, "".join(full_response), latency, True)
                await log_audit(log_entry)
            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    try:
        response = await engine.generate(messages)
        latency = perf_counter() - start_time
        log_entry = create_log_entry(request.patient_id, response, latency, False)
        await log_audit(log_entry)
        return {"response": response, "audit_ref": log_entry["audit_id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
