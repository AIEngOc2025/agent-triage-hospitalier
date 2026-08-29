import asyncio
import logging
import os
import traceback
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent_orchestrator import TriageAgentOrchestrator
from app.api_utils import create_log_entry, log_audit
from app.core.settings import settings
from app.engine_factory import engine
from app.middleware_timing import TimingMiddleware
from app.nlp_triage import triage_classifier
from app.remote.retry_utils import call_with_retry
from app.system_prompts import SYSTEM_PROMPT_FR, SYSTEM_PROMPT_JSON_FR
from app.triage_veto import decide_veto
from instructor.v2.core.errors import IncompleteOutputException

# ... existing imports ...

logger = logging.getLogger(__name__)

# --- AGENT PERSISTENCE ---
agent_sessions: Dict[str, TriageAgentOrchestrator] = {}

# --- Warmup config (cold start resilience) ---
WARMUP_TIMEOUT_SEC: float = 30.0  # pire cas : cold start vLLM ~5-15 s


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    @definition : Gestionnaire de cycle de vie de l'application FastAPI (init & shutdown).
    @args/params : app (FastAPI) - Instance de l'application.
    @return : AsyncGenerator - Contexte actif pendant l'exécution de l'API.
    """
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
    logger.info("🛑 Shutdown: engine released")


app = FastAPI(title="CHSA AI Gateway", lifespan=lifespan)
app.add_middleware(TimingMiddleware)


@app.get("/health", status_code=200, tags=["Monitoring"])
async def health_check():
    """
    @definition : Endpoint de contrôle de santé du service API et du moteur d'inférence.
    @args/params : Aucun.
    @return : Dict - Statut du service et type du moteur d'inférence actif.
    """
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
    """
    @definition : Extrait le dernier message de l'utilisateur depuis l'historique.
    @args/params : messages (List[dict]) - Historique des messages de la conversation.
    @return : str - Contenu textuel du dernier message envoyé par l'utilisateur.
    """
    return next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "",
    )


def _ensure_system_prompt(
    messages: List[dict], prompt_content: str = SYSTEM_PROMPT_FR
) -> List[dict]:
    """
    @definition : Garantit la présence du prompt système en tête d'historique.
    @args/params :
        messages (List[dict]) - Liste des messages de la session.
        prompt_content (str) - Contenu du prompt système à injecter si absent.
    @return : List[dict] - Liste des messages avec prompt système garanti en tête.
    """
    if not messages or messages[0].get("role") != "system":
        return [{"role": "system", "content": prompt_content}] + list(messages)
    return list(messages)


@app.post("/chat")
async def api_chat(request: ChatRequest):
    """
    @definition : Endpoint conversationnel exploitant l'orchestrateur de triage agentique.
    @args/params : request (ChatRequest) - Données de la requête (patient_id, history, stream).
    @return : Dict ou StreamingResponse - Réponse générée avec métadonnées d'audit et de triage.
    """
    start_time = perf_counter()
    user_input = _extract_user_input(request.history)

    # 1. Gestion de session agentique
    if request.patient_id not in agent_sessions:
        agent_sessions[request.patient_id] = TriageAgentOrchestrator()
    orchestrator = agent_sessions[request.patient_id]

    # --- Mode Streaming (SSE) ---
    if request.stream:

        async def event_generator():
            async for chunk in orchestrator.run_stream(
                user_input, history=request.history
            ):
                yield chunk

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # --- Mode Non-streaming (Exécution agentique complète) ---
    try:
        agent_result = await orchestrator.run(user_input, history=request.history)

        # Formatage de la réponse
        response_text = (
            agent_result.get("final_decision")
            or "Vos informations ont été enregistrées par l'équipe de triage."
        )
        reasoning = agent_result.get("reasoning")
        state = agent_result.get("state")
        triage_level = agent_result.get("triage_level", "modérée")
        is_emergency = agent_result.get("is_emergency", False)

        # Journalisation d'audit RGPD
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
            "triage_level": triage_level,
            "is_emergency": is_emergency,
            "audit_ref": log_entry["audit_id"],
        }

    except IncompleteOutputException as e:
        logger.error("❌ Erreur agentique : sortie incomplète (limite max_tokens)")
        raise HTTPException(
            status_code=422,
            detail="La réponse générée est incomplète (limite max_tokens atteinte).",
        ) from e
    except Exception as e:
        logger.error("❌ Erreur agentique : %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/triage", response_model=None)
async def api_triage(request: TriageRequest):
    """
    @definition : Endpoint structuré générant une réponse de triage typée Pydantic.
    @args/params : request (TriageRequest) - Requête contenant patient_id et historique.
    @return : Dict - Résultat structuré validé avec garde-fou NLP et veto de sécurité.
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

        # Sérialisation Pydantic -> dict & Log d'audit
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
    except IncompleteOutputException as e:
        logger.error(
            "❌ Structured triage failed: Output incomplete (max_tokens limit)"
        )
        raise HTTPException(
            status_code=422,
            detail="La réponse générée est incomplète (limite max_tokens atteinte).",
        ) from e
    except Exception as e:
        logger.error("❌ Structured triage failed: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
