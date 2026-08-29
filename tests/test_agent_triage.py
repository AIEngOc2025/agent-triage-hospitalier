from unittest.mock import AsyncMock, patch
import pytest

from app.agent_orchestrator import TriageAgentOrchestrator, TriageState
from app.agent_tools import (
    anonymize_clinical_data,
    classify_triage_urgency,
    is_vital_emergency_suspected,
)


def test_anonymization_tool():
    """Vérifie que l'outil d'anonymisation supprime les entités sensibles."""
    raw_text = "Patient John Doe, 45 ans, résidant à Paris."
    anonymized = anonymize_clinical_data(raw_text)
    assert anonymized != raw_text
    assert "<PATIENT>" in anonymized or "<ADRESSE>" in anonymized


def test_vital_emergency_tool():
    """Vérifie la détection de drapeaux rouges d'urgence vitale."""
    res_emergency = is_vital_emergency_suspected(
        "Forte douleur thoracique irradiant bras gauche"
    )
    assert res_emergency["vital_emergency"] is True
    assert "douleur thoracique" in res_emergency["matched_keywords"]

    res_normal = is_vital_emergency_suspected("Renouvellement d'ordonnance")
    assert res_normal["vital_emergency"] is False


def test_classify_urgency_tool():
    """Vérifie le format de réponse du classifieur NLP."""
    res = classify_triage_urgency("Douleur au genou depuis 3 semaines")
    assert "niveau" in res
    assert "confiance" in res
    assert res["niveau"] in ["maximale", "modérée", "différée"]


@pytest.mark.asyncio
async def test_orchestrator_flow():
    """Vérifie le flux de l'orchestrateur de START à FINALIZATION."""
    orchestrator = TriageAgentOrchestrator()
    assert orchestrator.state == TriageState.START

    with patch(
        "app.agent_orchestrator.engine.generate", new_callable=AsyncMock
    ) as mock_gen:
        mock_gen.return_value = "Orientation : Box d'urgence modérée."
        result = await orchestrator.run(
            "J'ai de la fièvre et mal à la gorge depuis hier."
        )

    assert orchestrator.state == TriageState.FINALIZATION
    assert result["state"] == "FINALIZATION"
    assert "final_decision" in result
    assert "reasoning" in result
    assert "triage_level" in result


@pytest.mark.asyncio
async def test_orchestrator_vital_emergency_stream():
    """Vérifie l'interception immédiate des urgences vitales en streaming."""
    orchestrator = TriageAgentOrchestrator()
    chunks = []
    async for chunk in orchestrator.run_stream(
        "Infarctus en cours et douleur thoracique"
    ):
        chunks.append(chunk)

    full_text = "".join(chunks)
    assert "URGENCE VITALE DÉTECTÉE" in full_text
    assert orchestrator.state == TriageState.FINALIZATION


def test_orchestrator_veto_handling():
    """Vérifie la transition après enregistrement d'un veto soignant."""
    orchestrator = TriageAgentOrchestrator()
    orchestrator.state = TriageState.VETO_WAIT
    res = orchestrator.handle_veto(approved=True, comment="Validation soignante")
    assert orchestrator.state == TriageState.FINALIZATION
    assert "Finalization" in res
