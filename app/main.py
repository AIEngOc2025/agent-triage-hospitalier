import asyncio
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .settings import settings

# --- PATCH: Robust fix for transformers/mlx-lm compatibility bug ---
try:
    from transformers.models.auto import TOKENIZER_MAPPING

    original_register = TOKENIZER_MAPPING.register

    def patched_register(*args, **kwargs):
        try:
            return original_register(*args, **kwargs)
        except AttributeError as e:
            if "'str' object has no attribute '__module__'" in str(e):
                return None
            raise e

    TOKENIZER_MAPPING.register = patched_register
    print("✅ [Patch] Transformers Tokenizer mapping patched with try-except for MLX.")
except Exception as e:
    print(f"⚠️  Patch failed: {e}")

# Optional imports for engines
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM, SamplingParams = None, None

try:
    from mlx_lm import load, generate
except ImportError:
    load, generate = None, None

# --- 2. ENGINE ABSTRACTION ---


class ModelEngine:
    """
    Abstraction layer for LLM inference.
    Uses vLLM in production (GPU) and MLX for local development (Apple Silicon).
    """

    def __init__(self):
        self.engine_type = None
        self.model = None
        self.tokenizer = None
        self.sampling_params = None

    def initialize(self):
        print(f"🔍 Initializing engine... APP_ENV: {settings.APP_ENV}, IS_PRODUCTION: {settings.IS_PRODUCTION}, IS_MACOS: {settings.IS_MACOS}")
        if settings.IS_PRODUCTION:
            self._init_vllm()
        elif settings.IS_MACOS:
            self._init_mlx()
        else:
            print("❌ Error: No compatible engine found for this environment.")

    def _init_vllm(self):
        if LLM is None:
            print("❌ [vLLM] Package not installed. Cannot start production engine.")
            return

        print(f"📥 [vLLM] Loading model from: {settings.MODEL_PATH}")
        self.engine_type = "vLLM"
        self.model = LLM(
            model=str(settings.MODEL_PATH),
            tokenizer=str(settings.MODEL_PATH),
            max_model_len=settings.VLLM_MAX_MODEL_LEN,
            trust_remote_code=True,
            tensor_parallel_size=settings.VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=0.80,
        )
        self.sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=512,
            repetition_penalty=1.15,
            stop=["<|im_end|>"],
        )
        print("✅ [vLLM] Engine operational.")

    def _init_mlx(self):
        if load is None:
            print("❌ [MLX] Package mlx-lm not installed. Local inference unavailable.")
            return

        print(f"📥 [MLX] Loading model from: {settings.MODEL_PATH}")
        try:
            self.model, self.tokenizer = load(str(settings.MODEL_PATH))
            self.engine_type = "MLX"
            print("✅ [MLX] Engine operational.")
        except Exception as e:
            print(f"❌ [MLX] Failed to load model: {e}")

    async def generate(self, messages: List[dict]) -> str:
        if not self.model:
            return "❌ Erreur : Moteur non initialisé."

        if self.engine_type == "vLLM":
            # On unifie la logique : on applique le template manuellement comme pour MLX
            # pour garantir un comportement identique entre dev et prod.
            prompt = self.model.get_tokenizer().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            outputs = await asyncio.to_thread(
                self.model.generate, prompt, self.sampling_params, use_tqdm=False
            )
            raw_text = outputs[0].outputs[0].text
        elif self.engine_type == "MLX":
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # La génération MLX reste inchangée
            raw_text = await asyncio.to_thread(
                generate,
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=512,
                temp=0.2,
            )
        else:
            return "❌ Erreur : Moteur non supporté."

        return self.clean_response(raw_text)

    def clean_response(self, text: str) -> str:
        clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        clean_text = re.sub(r"<[^>]+>", "", clean_text)
        return clean_text.strip()


# Global engine instance
engine = ModelEngine()


# --- 3. LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Start model loading in the background to allow the health check to respond immediately
    asyncio.create_task(asyncio.to_thread(engine.initialize))
    yield
    print("🛑 Arrêt de l'orchestrateur.")


# --- 4. API FASTAPI ---
app = FastAPI(title="CHSA AI Gateway", lifespan=lifespan)


@app.get("/health", status_code=200, tags=["Monitoring"])
async def health_check():
    return {"status": "ok", "engine": engine.engine_type}


class ChatRequest(BaseModel):
    patient_id: str = "PAT-001"
    history: List[dict]


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
5.  **Anti-Exemple :** Ne génère JAMAIS de cas cliniques ou de questions à choix multiples. Tu dois converser naturellement."""
        messages.insert(0, {"role": "system", "content": system_prompt})

    try:
        response = await engine.generate(messages)
        log_entry = {
            "audit_id": str(uuid.uuid4()),
            "patient_id": request.patient_id,
            "decision": response,
            "latency_sec": round(time.time() - start_time, 3),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return {"response": response, "audit_ref": log_entry["audit_id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
