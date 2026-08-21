from unittest.mock import patch

import pytest

from app.api_utils import TriageLogic


# Mock des dépendances pour le test
@patch("app.remote.client.RemoteInferenceClient.generate_structured")
@pytest.mark.asyncio
async def test_triage_logic_returns_correct_response(mock_generate):
    """
    @definition : Teste que la logique de triage retourne la réponse structurée attendue.
    @args/params : mock_generate (Mock object).
    @return : None.
    """
    from app.schemas import TriageResponse, TriageResult

    # Message assez long pour déclencher E501
    orientation_msg = (
        "Votre état nécessite une consultation immédiate. "
        "Veuillez vous rendre aux urgences."
    )
    mock_generate.return_value = TriageResponse(
        message=orientation_msg,
        triage_result=TriageResult(niveau="maximale", orientation="Urgences"),
    )

    triage_logic = TriageLogic()
    response = await triage_logic.process_triage("douleur thoracique")

    assert response.triage_result.niveau == "maximale"
    assert "Urgences" in response.triage_result.orientation
