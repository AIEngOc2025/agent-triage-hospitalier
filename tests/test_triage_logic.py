from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.remote.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@patch("app.remote.main.engine.client.generate_structured")
def test_no_diagnostic_refusal(mock_generate, client):
    """
    Test 1: Vérifie que le modèle refuse de poser un diagnostic.
    """
    from app.schemas import TriageResponse

    refusal_msg = (
        "Je ne peux pas établir de diagnostic. Mon rôle est d'évaluer l'urgence."
    )
    mock_generate.return_value = TriageResponse(message=refusal_msg, triage_result=None)

    response = client.post(
        "/triage",
        json={
            "patient_id": "PAT-123",
            "history": [{"role": "user", "content": "Quelle est ma maladie ?"}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert refusal_msg in data["message"]


@patch("app.remote.main.engine.client.generate_structured")
def test_conciseness_limit(mock_generate, client):
    """
    Test 2: Vérifie que la réponse respecte la limite de mots.
    """
    from app.schemas import TriageResponse

    short_response = (
        "Je comprends. Depuis combien de temps ressentez-vous cette douleur ?"
    )
    mock_generate.return_value = TriageResponse(
        message=short_response, triage_result=None
    )

    response = client.post(
        "/triage",
        json={
            "patient_id": "PAT-123",
            "history": [{"role": "user", "content": "J'ai mal au ventre."}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    # Vérifie la longueur en mots (max 50)
    word_count = len(data["message"].split())
    assert word_count <= 50


@patch("app.remote.main.engine.client.generate_structured")
def test_service_orientation(mock_generate, client):
    """
    Test 3: Vérifie que le modèle propose une orientation.
    """
    from app.schemas import TriageResponse, TriageResult

    orientation_msg = "Votre état nécessite une consultation immédiate. Veuillez vous rendre aux urgences."
    mock_generate.return_value = TriageResponse(
        message=orientation_msg,
        triage_result=TriageResult(
            niveau="maximale",
            orientation="Urgence vitale immédiate : SAMU (15) ou urgences.",
        ),
    )

    response = client.post(
        "/triage",
        json={
            "patient_id": "PAT-123",
            "history": [{"role": "user", "content": "J'ai très mal."}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "urgences" in data["message"].lower()
