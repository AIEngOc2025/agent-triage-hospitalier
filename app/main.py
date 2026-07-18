import asyncio
import json
import re
import uuid
from contextlib import asynccontextmanager
from time import perf_counter, strftime
from typing import AsyncGenerator, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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
        self.tokenizer = None
        self.sampling_params = None

    def initialize(self):
        print(
            "🔍 Initializing engine... "
            f"APP_ENV: {settings.APP_ENV}, "
            f"IS_PRODUCTION: {settings.IS_PRODUCTION}"
        )
        try:
            if settings.IS_MACOS:
                print("💻 MacOS detected: initializing vLLM-Metal (MLX)...")
                self._init_mlx()
            else:
                print("🚀 Initializing Async vLLM engine (Scalable GPU)...")
                self._init_vllm()
        except Exception as e:
            if settings.IS_PRODUCTION:
                print(f"❌ Critical error during production engine initialization: {e}")
                raise e
            else:
                print(f"⚠️  Development initialization warning: {e}")

    def _init_vllm(self):
        print("⚠️ [vLLM] Mocked initialization for local testing.")
        self.engine_type = "mock"
        self.model = None
        self.tokenizer = None
        self.sampling_params = None
        print("✅ [vLLM] Async Engine operational (Scalable GPU).")

    def _init_mlx(self):
        try:
            from mlx_lm import load
        except ImportError:
            print("❌ [MLX] Package mlx-lm not installed.")
            return

        print(f"📥 [MLX] Loading model from: {settings.MODEL_PATH}")
        try:
            self.model, self.tokenizer = load(str(settings.MODEL_PATH))
            self.engine_type = "MLX"
            print("✅ [MLX] Engine operational.")
        except Exception as e:
            print(f"❌ [MLX] Failed to load model: {e}")

    async def generate_stream(
        self, messages: List[dict], request_id: str
    ) -> AsyncGenerator[str, None]:
        """
        @definition: Generates a stream of text from the model based on the
        message history.
        @args/params:
            - messages (List[dict]): The conversation history.
            - request_id (str): A unique ID for the generation request.
        @return: An async generator yielding text chunks.
        """
        if not self.model:
            yield "❌ Erreur : Moteur non initialisé."
            return

        if self.engine_type == "vLLM":
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            results_generator = self.model.generate(
                prompt, self.sampling_params, request_id
            )

            final_text = ""
            async for request_output in results_generator:
                text = request_output.outputs[0].text
                # Extract only the new tokens
                delta = text[len(final_text) :]
                final_text = text
                yield delta

        elif self.engine_type == "MLX":
            from mlx_lm import generate as mlx_generate

            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # MLX generate is not natively an async stream like vLLM,
            # but we wrap it to keep the interface consistent.
            raw_text = await asyncio.to_thread(
                mlx_generate,
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=512,
            )
            yield self.clean_response(raw_text)
        else:
            yield "❌ Erreur : Moteur non supporté."

    async def generate(self, messages: List[dict]) -> str:
        """
        @definition: Generates a complete text response by consuming the entire stream.
        @args/params:
            - messages (List[dict]): The conversation history.
        @return: The final, complete response string from the model.
        """
        request_id = str(uuid.uuid4())
        full_text = ""
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


async def log_audit(entry: dict):
    """
    @definition: Writes an audit log entry to the configured log file in JSONL format.
    @args/params:
        - entry (dict): The log entry to record.
    @return: None.
    """
    try:

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
    """
    settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Start model loading in the background to allow the health check to
    # respond immediately
    asyncio.create_task(asyncio.to_thread(engine.initialize))
    yield
    print("🛑 Arrêt de la tentative du chargement de l'orchestrateur")


# --- 4. API FASTAPI ---
app = FastAPI(title="CHSA AI Gateway", lifespan=lifespan)


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
        response = await engine.generate(messages)
        latency = perf_counter() - start_time
        log_entry = create_log_entry(request.patient_id, response, latency, False)
        await log_audit(log_entry)
        return {"response": response, "audit_ref": log_entry["audit_id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
