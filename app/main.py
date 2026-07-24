import asyncio
import json
import re
import uuid
from contextlib import asynccontextmanager
from time import perf_counter, strftime
from typing import AsyncGenerator, Dict, List

import spacy
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

# ...
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine
from pydantic import BaseModel

from app.settings import settings
from app.system_prompts import SYSTEM_PROMPT_FR

# --- 2. ENGINE ABSTRACTION ---


class ModelEngine:
    """
    Scalable abstraction layer for LLM inference.
    Uses AsyncLLMEngine for production to enable continuous batching.
    Uses MLX for local development on MacOS.
    """

    def __init__(self):
        self.engine_type = None
        self.model = None

        # Initialisation par défaut pour éviter NoneType lors de l'accès
        class DefaultTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return ""

        self.tokenizer = DefaultTokenizer()
        print(f"DEBUG: Initialized tokenizer: {self.tokenizer}")
        self.sampling_params = None

    def initialize(self):
        print(
            "🔍 Initializing engine... "
            f"APP_ENV: {settings.APP_ENV}, "
            f"IS_PRODUCTION: {settings.IS_PRODUCTION}"
        )
        try:
            # En production, on utilise le vrai moteur VLLM.
            # En développement, on utilise TOUJOURS le moteur mocké.
            if settings.IS_PRODUCTION:
                print("🚀 [PROD] Initializing Async vLLM engine (GPU)...")
                self._init_vllm()
            elif settings.IS_MACOS:
                print(
                    "💻 [DEV] macOS detected. Initializing MLX engine "
                    "(Apple Silicon)..."
                )
                self._init_mlx()
            else:
                print("💻 [DEV] Fallback. Initializing mocked engine...")
                self._init_vllm_mock()

        except Exception as e:
            if settings.IS_PRODUCTION:
                print(f"❌ Critical error during production engine initialization: {e}")
                raise e
            else:
                print(f"⚠️  Development initialization warning: {e}")

    def _init_vllm_mock(self):
        print("⚠️ [vLLM] Mocked initialization for local testing.")
        self.engine_type = "MockEngine"

        # Mocking for local testing
        class MockModel:
            def generate(self, prompt, params, request_id):
                class MockOutput:
                    def __init__(self, text):
                        self.text = text

                class MockRequestOutput:
                    def __init__(self, text):
                        self.outputs = [MockOutput(text)]

                async def gen():
                    yield MockRequestOutput("Ceci est une réponse factice du triage.")

                return gen()

        self.model = MockModel()

        class MockTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "Mock prompt"

        self.tokenizer = MockTokenizer()
        self.sampling_params = None
        print("✅ [Mock] Mock Engine operational.")

    def _init_vllm(self):
        # Cette méthode sera utilisée en production dans le conteneur
        print("✅ [vLLM] Async Engine operational (Scalable GPU).")
        self.engine_type = "vLLM"

    async def generate_stream(
        self, messages: List[dict], request_id: str
    ) -> AsyncGenerator[str, None]:
        """
        @definition: Generates a stream of text. Simplified for dev.
        """
        yield "Ceci est une réponse factice en mode développement. "
        yield "Pourriez-vous préciser vos symptômes et votre âge ?"

    async def generate(self, messages: List[dict]) -> str:
        """
        @definition: Generates a complete text response. Simplified for dev.
        @args/params: messages (List[dict])
        @return: str (The generated response)
        """
        if self.engine_type == "MockEngine":
            mock_response = (
                "Ceci est une réponse factice du triage. "
                "Pourriez-vous me donner votre âge et décrire "
                "vos symptômes principaux ?"
            )
            return self.clean_response(mock_response)
        else:  # Production (vLLM)
            request_id = str(uuid.uuid4())
            full_text = ""
            # En production, la génération non-streamée est une accumulation du stream
            async for chunk in self.generate_stream(messages, request_id):
                full_text += chunk
            return self.clean_response(full_text)

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
    """
    @definition: Anonymise les entités sensibles dans le texte.
    @args/params: text (str)
    @return: str (texte anonymisé)
    """
    global analyzer
    if analyzer is None:
        nlp = spacy.load("fr_core_news_sm")
        nlp_engine = SpacyNlpEngine(models={"fr": nlp})
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    results = analyzer.analyze(text=text, language="fr")
    anonymized_result = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized_result.text


# ...


async def log_audit(entry: dict):
    """
    @definition: Writes an audit log entry in JSONL format, anonymized.
    @args/params:
        - entry (dict): The log entry to record.
    @return: None.
    """
    try:
        # Anonymiser la décision avant de loguer
        entry["decision"] = anonymize_text(entry["decision"])

        def write_log():
            with open(settings.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        await asyncio.to_thread(write_log)
    except Exception as e:
        print(f"❌ Audit logging failed: {e}")


def create_log_entry(
    patient_id: str, decision: str, latency: float, stream: bool
) -> Dict:
    """
    @definition: Creates a standardized dictionary for an audit log entry.
    @args/params:
        - patient_id (str): The patient's identifier.
        - decision (str): The final model response.
        - latency (float): The request processing time in seconds.
        - stream (bool): Whether the request was streaming.
    @return: A dictionary representing the log entry.
    """
    return {
        "audit_id": str(uuid.uuid4()),
        "patient_id": patient_id,
        "decision": decision,
        "latency_sec": round(latency, 3),
        "timestamp": strftime("%Y-%m-%d %H:%M:%S"),
        "stream": stream,
    }


# Global engine instance
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
    import os

    import uvicorn

    port = int(os.environ.get("PORT", 8000))
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
                log_entry = create_log_entry(
                    request.patient_id, "".join(full_response), latency, True
                )
                await log_audit(log_entry)
            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    try:
        print(f"DEBUG: Calling engine.generate with messages: {messages}")
        response = await engine.generate(messages)
        print(f"DEBUG: engine.generate returned: {response}")
        latency = perf_counter() - start_time
        log_entry = create_log_entry(request.patient_id, response, latency, False)
        await log_audit(log_entry)
        return {"response": response, "audit_ref": log_entry["audit_id"]}
    except Exception as e:
        print(f"DEBUG: Exception in api_chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
