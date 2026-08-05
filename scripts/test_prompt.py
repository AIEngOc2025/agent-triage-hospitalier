import asyncio

from app.remote.client import RemoteInferenceClient
from app.system_prompts import SYSTEM_PROMPT_FR


async def test_prompt():
    # Initialiser le client avec température 0
    client = RemoteInferenceClient(temperature=0.0)

    # Simuler le comportement de app/main.py
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FR},
        {"role": "user", "content": "Bonjour"},
    ]

    print("--- TEST DE SALUTATION (Temp=0) ---")

    try:
        response = await client.generate(messages)
        print(f"Réponse du modèle : {response}")
    except Exception as e:
        print(f"Erreur lors du test : {e}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(test_prompt())
