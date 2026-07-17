from fastapi.testclient import TestClient

# L'import est maintenant absolu depuis la racine du projet (src)
from src.api.orchestrateur import app


def test_triage_endpoint_success():
    """
    Tests a successful call to the /triage endpoint with valid data.
    """
    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "symptomes": (
                    "Le patient a une forte fièvre et des "
                    "difficultés à respirer."
                )
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "decision" in data
        assert "latency_sec" in data
        assert isinstance(data["decision"], str)
        assert isinstance(data["latency_sec"], float)
        assert len(data["decision"]) > 0


def test_triage_endpoint_invalid_input():
    """
    Tests a call to the /triage endpoint with invalid or missing data.
    The API should return a 422 Unprocessable Entity error.
    """
    with TestClient(app) as client:
        # Envoi d'une requête avec un champ incorrect ("symptom" au lieu de "symptomes")
        response = client.post("/triage", json={"symptom": "Données invalides"})
        assert response.status_code == 422
