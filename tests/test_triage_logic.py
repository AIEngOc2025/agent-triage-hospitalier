import pytest
from unittest.mock import MagicMock, patch
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
        result=TriageResult(niveau="maximale", raison="Urgent", orientation="Urgences"),
    )

    triage_logic = TriageLogic()
    response = await triage_logic.process_triage("douleur thoracique")

    assert response.result.niveau == "maximale"
    assert "Urgences" in response.result.orientation
