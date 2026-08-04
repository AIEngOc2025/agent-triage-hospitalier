import asyncio
import json
import time
from pathlib import Path

import httpx

# URL de l'API déployée
API_URL = "https://agent-triage-hospitalier-rlgcjqsysq-ew.a.run.app"
TEST_FILE = "data/processed/Mpaga_Christophe_1_Dataset_Test_DPO_052026.jsonl"


async def evaluate_api():
    """
    @definition : Envoie des exemples d'un fichier JSONL à l'API déployée
                  et mesure la latence/réponse.
    @args/params : Aucun (utilise les constantes API_URL et TEST_FILE).
    @return : None (affiche les résultats).
    """
    if not Path(TEST_FILE).exists():
        print(f"❌ Erreur : Fichier de test introuvable : {TEST_FILE}")
        return

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    print(f"🚀 Début de l'évaluation sur {len(dataset)} exemples...")

    results = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for item in dataset[:10]:  # Limité à 10 pour le test
            prompt = item.get("prompt", "")

            # Reconstruction simple de l'historique pour l'API
            payload = {
                "patient_id": "EVAL-TEST",
                "history": [{"role": "user", "content": prompt}],
                "stream": False,
            }

            start_time = time.perf_counter()
            try:
                response = await client.post(f"{API_URL}/chat", json=payload)
                latency = time.perf_counter() - start_time

                if response.status_code == 200:
                    data = response.json()
                    results.append(
                        {
                            "prompt": prompt,
                            "response": data.get("response", ""),
                            "latency": latency,
                        }
                    )
                    print(f"✅ Succès (Latence: {latency:.2f}s)")
                else:
                    print(f"❌ Erreur API {response.status_code}: {response.text}")
            except Exception as e:
                print(f"⚠️ Exception lors de l'appel : {e}")

    # Résumé
    avg_latency = sum(r["latency"] for r in results) / len(results) if results else 0
    print("\n" + "=" * 40)
    print("📊 RÉSULTAT ÉVALUATION API")
    print(f"Exemples évalués : {len(results)}")
    print(f"Latence moyenne : {avg_latency:.2f}s")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(evaluate_api())
