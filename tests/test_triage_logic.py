from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@patch("app.main.engine.generate")
def test_no_diagnostic_refusal(mock_generate, client):
    """
    Test 1: Vérifie que le modèle refuse de poser un diagnostic.
    """
    refusal_msg = (
        "Je ne peux pas établir de diagnostic. Mon rôle est d'évaluer l'urgence."
    )
    mock_generate.return_value = refusal_msg

    response = client.post(
        "/chat",
        json={
            "patient_id": "TEST-TRIAGE",
            "history": [{"role": "user", "content": "Quelle est ma maladie ?"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert refusal_msg in data["response"]


@patch("app.main.engine.generate")
def test_conciseness_limit(mock_generate, client):
    """
    Test 2: Vérifie que la réponse respecte la limite de mots.
    """
    short_response = (
        "Je comprends. Depuis combien de temps ressentez-vous cette douleur ?"
    )
    mock_generate.return_value = short_response

    response = client.post(
        "/chat",
        json={
            "patient_id": "TEST-TRIAGE",
            "history": [{"role": "user", "content": "J'ai mal au ventre."}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    # Vérifie la longueur en mots (max 50)
    word_count = len(data["response"].split())
    assert word_count <= 50


@patch("app.main.engine.generate")
def test_service_orientation(mock_generate, client):
    """
    Test 3: Vérifie que le modèle propose une orientation.
    """
    orientation_msg = "Votre état nécessite une consultation immédiate. Veuillez vous rendre aux urgences."
    mock_generate.return_value = orientation_msg

    response = client.post(
        "/chat",
        json={
            "patient_id": "TEST-TRIAGE",
            "history": [{"role": "user", "content": "J'ai très mal."}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "urgences" in data["response"].lower()
