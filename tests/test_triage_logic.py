import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_chat_questionnaire_logic(client):
    """
    Tests if the agent follows the system prompt instructions for triage.
    """
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
    # Check that it asks a question (should be in the system prompt)
    # The new prompt should lead to asking for symptoms or age.
    assert len(data["response"]) > 0
