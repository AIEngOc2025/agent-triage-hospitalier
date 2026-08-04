from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import HTTPStatusError, Request, Response

from app.remote.main import app, engine


@pytest.fixture
def client():
    engine.initialize()
    return TestClient(app)


@pytest.mark.asyncio
async def test_api_chat_500_replication(client):
    """
    Replication of the 500 error seen in production.
    We mock the client to raise an HTTPStatusError (like the 404 we saw).
    """

    # Mocking RemoteInferenceClient.generate to raise a 404
    # (replicating production behavior)
    with patch(
        "app.remote.client.RemoteInferenceClient.generate",
        new_callable=AsyncMock,
    ) as mock_generate:
        # Simulate a 404 error from the inference service
        request = Request(
            "POST",
            "https://agent-inference-service-414294705487.europe-west1.run.app/v1/chat/completions",
        )
        response = Response(404, request=request)
        mock_generate.side_effect = HTTPStatusError(
            "404 Not Found", request=request, response=response
        )

        # Test the request
        response = client.post(
            "/chat",
            json={
                "patient_id": "TEST-500-REPL",
                "history": [{"role": "user", "content": "Bonjour"}],
                "stream": False,
            },
        )

        # We expect a 500 because the Gateway catches the exception
        # and raises HTTPException(500)
        assert response.status_code == 500
        data = response.json()
        print(f"DEBUG: Error response detail: {data['detail']}")
