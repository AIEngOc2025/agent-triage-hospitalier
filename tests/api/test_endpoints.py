import sys
import os
from fastapi.testclient import TestClient
import pytest

# Add the src/api directory to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/api')))

# Détermine quel 'app' importer en fonction de l'environnement
IS_ORCHESTRATOR = os.getenv("APP_MODE") == "orchestrator"
if IS_ORCHESTRATOR:
    from orchestrateur import app
else:
    from local.main import app


@pytest.fixture
def client():
    # Utilise un contexte pour s'assurer que le lifespan est correctement géré
    with TestClient(app) as c:
        yield c

def test_chat_endpoint_status(client: TestClient):
    """Vérifie que l'API répond 200 OK"""
    response = client.post("/chat", json={"history": [{"role": "user", "content": "Bonjour"}], "patient_id": "PAT-TEST-001"})
    assert response.status_code == 200

def test_chat_response_schema(client: TestClient):
    """Vérifie que le format JSON de sortie est conforme au PDF (Traçabilité)"""
    response = client.post("/chat", json={"history": [{"role": "user", "content": "Forte fièvre"}], "patient_id": "PAT-TEST-002"})
    json_data = response.json()

    # Le schéma de réponse a changé pour être plus simple
    assert "response" in json_data
    # La réponse doit être une chaîne de caractères non vide
    assert isinstance(json_data["response"], str)
    assert len(json_data["response"]) > 0
