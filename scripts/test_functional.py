import httpx

API_URL = "https://agent-api-gateway-414294705487.europe-west1.run.app/chat"

payload = {
    "history": [
        {
            "role": "user",
            "content": "Bonjour, j'ai des douleurs dorsales importantes depuis ce matin, cela me bloque dans mes mouvements.",
        }
    ],
    "patient_id": "PAT-123",
    "stream": True,
}

print(f"🚀 Envoi de la requête de test (STREAMING) à {API_URL}...")
try:
    with httpx.stream("POST", API_URL, json=payload, timeout=360.0) as response:
        response.raise_for_status()
        print("\n✅ Succès ! Flux reçu:")
        for line in response.iter_text():
            print(line, end="", flush=True)
    print("\n\n✅ Fin du flux.")
except httpx.HTTPStatusError as e:
    print(f"\n❌ Erreur API ({e.response.status_code}) : {e.response.text}")
except Exception as e:
    print(f"\n❌ Erreur de connexion : {e}")
