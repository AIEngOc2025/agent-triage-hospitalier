import asyncio
import logging
import os
import traceback
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent_orchestrator import TriageAgentOrchestrator
from app.api_utils import create_log_entry, log_audit
from app.core.settings import settings
from app.local.engine import LocalEngine
from app.middleware_timing import TimingMiddleware
from app.nlp_triage import triage_classifier
from app.remote.engine import RemoteEngine
from app.remote.retry_utils import call_with_retry
from app.system_prompts import SYSTEM_PROMPT_FR, SYSTEM_PROMPT_JSON_FR
from app.triage_veto import decide_veto

logger = logging.getLogger(__name__)

# --- AGENT PERSISTENCE ---
agent_sessions: Dict[str, TriageAgentOrchestrator] = {}

# --- Warmup config (cold start resilience) ---
WARMUP_TIMEOUT_SEC: float = 30.0  # pire cas : cold start vLLM ~5-15 s


# --- ENGINE ABSTRACTION ---
def get_engine():
    if settings.ENGINE_MODE == "local":
        return LocalEngine(settings)
    return RemoteEngine()


engine = get_engine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    engine.initialize()
    # Warmup best-effort : amorce vLLM pour éviter le cold start sur la 1ère
    # requête réelle. L'API démarre même si le warmup échoue.
    try:
        await asyncio.wait_for(
            engine.generate(
                [
                    {"role": "system", "content": "Warmup."},
                    {"role": "user", "content": "ok"},
                ]
            ),
            timeout=WARMUP_TIMEOUT_SEC,
        )
        logger.info("✅ Inference warmup OK")
    except Exception as exc:
        logger.warning(
            "⚠️ Warmup best-effort échoué (%s) — l'API démarre quand même",
            exc,
        )
    yield
    await engine.close()
    print("🛑 Shutdown: engine released")


app = FastAPI(title="CHSA AI Gateway", lifespan=lifespan)
app.add_middleware(TimingMiddleware)


@app.get("/health", status_code=200, tags=["Monitoring"])
async def health_check():
    return {"status": "ok", "engine": engine.engine_type}


class ChatRequest(BaseModel):
    patient_id: str = Field(
        ...,
        pattern=r"^(PAT-\d{3,}|conv-user)$",
        description="Patient identifier (format: PAT-XXX ou conv-user)",
    )
    history: List[dict] = Field(..., min_length=1, max_length=50)
    stream: bool = False


class TriageRequest(BaseModel):
    patient_id: str = Field(
        ...,
        pattern=r"^(PAT-\d{3,}|conv-user)$",
        description="Patient identifier (format: PAT-XXX ou conv-user)",
    )
    history: List[dict] = Field(..., min_length=1, max_length=50)


def _extract_user_input(messages: List[dict]) -> str:
    """Extrait le dernier message user de l'historique (helper partagé)."""
    return next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        "",
    )


def _ensure_system_prompt(
    messages: List[dict], prompt_content: str = SYSTEM_PROMPT_FR
) -> List[dict]:
    """Garantit la présence du system prompt en tête (sans muter l'input)."""
    if messages[0].get("role") != "system":
        return [{"role": "system", "content": prompt_content}] + list(messages)
    return list(messages)


@app.post("/chat")
async def api_chat(request: ChatRequest):
    """Endpoint conversationnel : utilise l'orchestrateur agentique."""
    start_time = perf_counter()
    user_input = _extract_user_input(request.history)

    # 1. Gestion de session agent
    if request.patient_id not in agent_sessions:
        agent_sessions[request.patient_id] = TriageAgentOrchestrator()
    orchestrator = agent_sessions[request.patient_id]

    # 2. Exécution agentique
    try:
        agent_result = orchestrator.run(user_input)

        # 3. Formatage réponse
        response_text = (
            agent_result.get("final_decision")
            or agent_result.get("question")
            or "Pas de réponse générée."
        )
        reasoning = agent_result.get("reasoning")
        state = agent_result.get("state")

        # 4. Logs
        latency = perf_counter() - start_time
        log_entry = create_log_entry(
            request.patient_id,
            user_input,
            response_text,
            latency,
            request.stream,
        )
        await log_audit(log_entry)
        return {
            "response": response_text,
            "reasoning": reasoning,
            "state": state,
            "audit_ref": log_entry["audit_id"],
        }

    except Exception as e:
        logger.error(f"❌ Erreur agentique : {traceback.format_exc()}")
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
    messages = _ensure_system_prompt(request.history, SYSTEM_PROMPT_JSON_FR)

    try:
        result = await call_with_retry(lambda: engine.generate_structured(messages))
        latency = perf_counter() - start_time

        # --- Veto bidirectionnel NLP ↔ LLM ---
        user_input = _extract_user_input(messages)
        nlp_pred = triage_classifier.predict(user_input)
        triage_result = result.triage_result
        llm_niveau = (
            triage_result.niveau.value
            if triage_result and triage_result.niveau
            else None
        )
        llm_orientation = triage_result.orientation if triage_result else result.message

        veto = decide_veto(
            llm_niveau=llm_niveau,
            llm_orientation=llm_orientation,
            nlp_niveau=nlp_pred.get("niveau"),
            nlp_confiance=float(nlp_pred.get("confiance", 0.0)),
        )

        # Si le veto a modifié le niveau, on reconstruit la réponse
        response_payload = result.model_dump()
        if veto.source != "llm" and veto.final_niveau != llm_niveau:
            response_payload["triage_result"] = {
                "niveau": veto.final_niveau,
                "orientation": veto.orientation,
            }
        response_payload["triage_source"] = veto.source
        response_payload["nlp_veto_meta"] = {
            "nlp_niveau": veto.nlp_niveau,
            "nlp_confiance": veto.nlp_confiance,
            "nlp_mode": nlp_pred.get("mode"),
            "rationale": veto.rationale,
        }

        # Sérialisation Pydantic -> dict
        log_entry = create_log_entry(
            request.patient_id,
            _extract_user_input(messages),
            response_payload,
            latency,
            False,
        )
        await log_audit(log_entry)
        return {
            **response_payload,
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
