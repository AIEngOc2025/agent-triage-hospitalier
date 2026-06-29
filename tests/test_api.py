import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add the src directory to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/api')))

from main import app # Import app after sys.path.insert

def test_triage_endpoint_success():
    """
    Tests a successful call to the /triage endpoint with valid data.
    """
    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={"symptomes": "Le patient a une forte fièvre et des difficultés à respirer."}
        )
        assert response.status_code == 200
        data = response.json()
        assert "decision" in data
        assert "latency_sec" in data
        assert isinstance(data["decision"], str)
        assert isinstance(data["latency_sec"], float)
        assert len(data["decision"]) > 0