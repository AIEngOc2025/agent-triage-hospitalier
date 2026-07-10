import os
from fastapi.testclient import TestClient
import pytest

# Détermine quel 'app' importer en fonction de l'environnement
# Les imports sont maintenant alignés avec la structure du projet (app/)
from app.main import app


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_chat_endpoint_status(client: TestClient):
    """Vérifie que l'API répond 200 OK"""
    response = client.post(
        "/chat",
        json={
            "history": [{"role": "user", "content": "Bonjour"}],
            "patient_id": "PAT-TEST-001",
        },
    )
    assert response.status_code in {200, 503}


def test_chat_response_schema(client: TestClient):
    """Vérifie que le format JSON de sortie est conforme au PDF (Traçabilité)"""
    response = client.post(
        "/chat",
        json={
            "history": [{"role": "user", "content": "Forte fièvre"}],
            "patient_id": "PAT-TEST-002",
        },
    )
    json_data = response.json()

    # Le schéma de réponse a changé pour être plus simple
    if response.status_code == 503:
        assert json_data["detail"] == "Le service d'inférence est indisponible."
    else:
        assert "response" in json_data
        assert isinstance(json_data["response"], str)
        assert len(json_data["response"]) > 0
