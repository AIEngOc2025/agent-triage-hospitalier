import sys
import os
import .src.api.main as main
from fastapi.testclient import TestClient

# Add the src/api directory to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/api')))

from main import app  # Importe votre application FastAPI

def test_triage_endpoint_status():
    """Vérifie que l'API répond 200 OK"""
    with TestClient(app) as client:
        response = client.post("/triage", json={"symptomes": "J'ai mal au bras."})
        assert response.status_code == 200

def test_triage_response_schema():
    """Vérifie que le format JSON de sortie est conforme au PDF (Traçabilité)"""
    with TestClient(app) as client:
        response = client.post("/triage", json={"symptomes": "Forte fièvre"})
        json_data = response.json()
    
        # Basé sur le schéma de réponse de src/api/main.py
        assert "decision" in json_data
        assert "latency_sec" in json_data

def test_triage_bilingual_logic():
    """Vérifie que l'IA ne mélange pas les langues en sortie"""
    # The model is prompted in French, so it should respond in French.
    with TestClient(app) as client:
        # Test with an English symptom, expecting a French response
        resp_fr = client.post("/triage", json={"symptomes": "Chest pain"})
        # Check for common French triage terms
        assert any(word in resp_fr.json()["decision"].lower() for word in ["urgence", "immédiate", "prioritaire", "consulter", "rapide"])
