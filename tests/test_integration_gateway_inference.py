from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.remote.main import (
    app,
)  # Utilisation de app.remote.main car c'est le point d'entrée réel


@pytest.fixture
def client():
    # Nécessaire pour initialiser l'engine et les dépendances avant les tests
    from app.remote.main import engine

    engine.initialize()
    return TestClient(app)


@pytest.mark.asyncio
async def test_api_chat_integration(client):
    """
    Test d'intégration de bout en bout pour api_chat.
    On mocke le client d'inférence pour isoler la Gateway.
    """
    # Mocking de RemoteInferenceClient.generate dans app.remote.client
    with patch(
        "app.remote.client.RemoteInferenceClient.generate", new_callable=AsyncMock
    ) as mock_generate:
        # On définit le comportement attendu du service d'inférence
        mock_generate.return_value = "Réponse simulée de l'IA"

        # Test de la requête
        response = client.post(
            "/chat",
            json={
                "patient_id": "PAT-001",
                "history": [{"role": "user", "content": "Bonjour"}],
                "stream": False,
            },
        )

        # Vérification
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Réponse simulée de l'IA"

        # Vérification que le mock a été appelé avec les bons arguments
        mock_generate.assert_called_once()
