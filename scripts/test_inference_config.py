import asyncio
import os

from app.remote.client import RemoteInferenceClient


async def test_inference_config():
    # Définir les variables d'environnement pour tester la lecture dynamique
    os.environ["TEMPERATURE"] = "0.2"
    os.environ["MAX_TOKENS"] = "60"

    # Initialiser le client
    client = RemoteInferenceClient()

    print(f"DEBUG: Temperature loaded: {client.temperature}")
    print(f"DEBUG: Max Tokens loaded: {client.max_tokens}")

    # Préparer un message de test simple
    messages = [{"role": "user", "content": "Bonjour"}]

    # Préparer le payload et vérifier les paramètres
    payload = client._prepare_payload(messages, stream=False)
    print(f"DEBUG: Payload temperature: {payload['temperature']}")
    print(f"DEBUG: Payload max_tokens: {payload['max_tokens']}")

    # Nettoyage
    await client.close()


if __name__ == "__main__":
    asyncio.run(test_inference_config())
