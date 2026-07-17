import asyncio
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.settings import settings

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
            f"🔍 Initializing engine... APP_ENV: {settings.APP_ENV}, IS_PRODUCTION: {settings.IS_PRODUCTION}"
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
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
        except ImportError:
            raise ImportError("vLLM AsyncLLMEngine package not installed.")

        print(f"📥 [vLLM] Loading model from: {settings.MODEL_PATH}")
        self.engine_type = "vLLM"

        engine_args = AsyncEngineArgs(
            model=str(settings.MODEL_PATH),
            tokenizer=str(settings.MODEL_PATH),
            max_model_len=settings.VLLM_MAX_MODEL_LEN,
            trust_remote_code=True,
            tensor_parallel_size=settings.VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=0.80,
        )
        self.model = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = self.model.get_tokenizer()

        self.sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=512,
            repetition_penalty=1.15,
            stop=["<|im_end|>"],
        )
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
        request_id = str(uuid.uuid4())
        full_text = ""
        async for chunk in self.generate_stream(messages, request_id):
            full_text += chunk

        return self.clean_response(full_text)

    def clean_response(self, text: str) -> str:
        clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        clean_text = re.sub(r"<[^>]+>", "", clean_text)
        return clean_text.strip()


async def log_audit(entry: dict):
    """
    Writes an audit log entry to the configured log file in JSONL format.

    @args/params : entry (dict) - The log entry to record.
    @return : None
    """
    try:

        def write_log():
            with open(settings.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        await asyncio.to_thread(write_log)
    except Exception as e:
        print(f"❌ Audit logging failed: {e}")


# Global engine instance
engine = ModelEngine()


# --- 3. LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Start model loading in the background to allow the health check to respond immediately
    asyncio.create_task(asyncio.to_thread(engine.initialize))
    yield
    print("🛑 Arrêt de la tentative du chargement de l'orchestrateur")


# --- 4. API FASTAPI ---
app = FastAPI(title="CHSA AI Gateway", lifespan=lifespan)


@app.get("/health", status_code=200, tags=["Monitoring"])
async def health_check():
    print(f"🩺 Health check called. Engine: {engine.engine_type}")
    return {"status": "ok", "engine": engine.engine_type}


class ChatRequest(BaseModel):
    patient_id: str = "PAT-001"
    history: List[dict]
    stream: bool = False


@app.post("/chat")
async def api_chat(request: ChatRequest):
    start_time = time.time()
    messages = request.history
    if not messages or messages[0].get("role") != "system":
        system_prompt = """Tu es un infirmier de triage pour le Centre Hospitalier Sud-Aveyron (CHSA).

**Instructions strictes :**
1.  **Présentation :** Commence TOUJOURS par te présenter et demander la raison de la venue.
2.  **Une seule question :** Pose UNE SEULE question courte et simple à la fois pour préciser les symptômes.
3.  **Rôle limité :** Ne donne JAMAIS de diagnostic, d'explication, de conseil ou de niveau d'urgence. Ton unique objectif est de poser la question suivante pour recueillir de l'information.
4.  **Bilinguisme :** Réponds en français ou en anglais selon la langue de l'utilisateur.
5.  **Anti-Répétition :** Ne répète JAMAIS les mêmes phrases. Sois extrêmement concis. Une seule phrase courte suffit.
6.  **Anti-Exemple :** Ne génère JAMAIS de cas cliniques ou de questions à choix multiples. Tu dois converser naturellement."""
        messages.insert(0, {"role": "system", "content": system_prompt})

    if request.stream:

        async def event_generator():
            try:
                full_response = []
                async for chunk in engine.generate_stream(messages, str(uuid.uuid4())):
                    full_response.append(chunk)
                    yield f"data: {chunk}\n\n"

                # Log completion of streaming request
                log_entry = {
                    "audit_id": str(uuid.uuid4()),
                    "patient_id": request.patient_id,
                    "decision": "".join(full_response),
                    "latency_sec": round(time.time() - start_time, 3),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "stream": True,
                }
                await log_audit(log_entry)
            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    try:
        response = await engine.generate(messages)
        log_entry = {
            "audit_id": str(uuid.uuid4()),
            "patient_id": request.patient_id,
            "decision": response,
            "latency_sec": round(time.time() - start_time, 3),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stream": False,
        }
        await log_audit(log_entry)
        return {"response": response, "audit_ref": log_entry["audit_id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
