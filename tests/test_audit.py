import json
from unittest.mock import ANY, AsyncMock, mock_open, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app, log_audit


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.asyncio
async def test_log_audit_writes_file():
    """Tests that log_audit writes the correct JSON line to the log file."""
    entry = {"audit_id": "123", "patient_id": "PAT-001", "decision": "Hello"}

    with patch("builtins.open", mock_open()) as mocked_file:
        await log_audit(entry)

        # Check if file was opened in append mode
        mocked_file.assert_called_once_with(ANY, "a", encoding="utf-8")

        # Check if the written content is the JSON string of the entry
        handle = mocked_file()
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)
        assert json.loads(written_data) == entry


@pytest.mark.asyncio
@patch("app.main.engine.generate")
async def test_api_chat_logs_audit(mock_generate, client):
    """Tests that calling /chat triggers an audit log entry."""
    mock_generate.return_value = "Test response"

    with patch("app.main.log_audit", new_callable=AsyncMock) as mock_log:
        response = client.post(
            "/chat",
            json={
                "history": [{"role": "user", "content": "Hello"}],
                "patient_id": "PAT-TEST-LOG",
                "stream": False,
            },
        )

        assert response.status_code == 200
        # Verify log_audit was called
        mock_log.assert_called_once()
        log_entry = mock_log.call_args[0][0]
        assert log_entry["patient_id"] == "PAT-TEST-LOG"
        assert log_entry["decision"] == "Test response"
        assert log_entry["stream"] is False


@pytest.mark.asyncio
@patch("app.main.engine.generate_stream")
async def test_api_chat_streaming_logs_audit(mock_generate_stream, client):
    """Tests that streaming /chat triggers an audit log entry after completion."""

    async def mock_stream(*args, **kwargs):
        yield "Hello "
        yield "World"

    mock_generate_stream.return_value = mock_stream()

    with patch("app.main.log_audit", new_callable=AsyncMock) as mock_log:
        response = client.post(
            "/chat",
            json={
                "history": [{"role": "user", "content": "Hello"}],
                "patient_id": "PAT-TEST-STREAM",
                "stream": True,
            },
        )

        assert response.status_code == 200

        # Since it's a streaming response, we must consume the stream to trigger the log
        full_content = ""
        for line in response.iter_lines():
            if line:
                full_content += line

        # Now verify log_audit was called

        mock_log.assert_called_once()
        log_entry = mock_log.call_args[0][0]
        assert log_entry["patient_id"] == "PAT-TEST-STREAM"
        assert "Hello World" in log_entry["decision"]
        assert log_entry["stream"] is True
