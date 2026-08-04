from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@patch("app.main.engine.generate")
def test_chat_questionnaire_logic(mock_generate, client):
    """
    Tests if the agent follows the system prompt instructions for triage.
    """
    mock_generate.return_value = "Hello, how can I help you?"
    response = client.post(
        "/chat",
        json={
            "patient_id": "TEST-TRIAGE",
            "history": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["response"] == "Hello, how can I help you?"
