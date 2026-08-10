"""
Benchmark CLOUD — appelle l'API déployée sur Cloud Run
Quasi identique au script local, mais avec :
 - URL : L'URL de l'API sur Cloud Run.
 - Authentification OIDC (token GCP).
 - Mesure du cold start Cloud via un endpoint warmup + une pause.
"""

import asyncio
import json
import os
import statistics
import time
from pathlib import Path

import httpx
from google.auth.transport.requests import Request
from google.oauth2 import id_token

# --- CONFIGURATION ---
API_URL = "https://agent-api-gateway-414294705487.europe-west1.run.app"
PROMPTS_FILE = "scripts/benchmark/data/benchmark_prompts.jsonl"
RESULTS_FILE = "scripts/benchmark/data/results_cloud.jsonl"
N_REPEAT = 5

# Auth OIDC (récupère un token pour Cloud Run)


def get_token():
    """Récupère un token d'identité OIDC pour authentifier les requêtes."""
    token = os.environ.get("BENCHMARK_TOKEN")
    if token:
        print("ℹ️  Utilisation du token fourni via BENCHMARK_TOKEN")
        return token

    print("ℹ️  Tentative de récupération d'un token via google.auth...")
    auth_req = Request()
    return id_token.fetch_id_token(auth_req, API_URL)


async def warmup(client: httpx.AsyncClient):
    """Une requête de warmup pour amorcer l'instance Cloud Run."""
    print("🔥 Warmup (peut prendre du temps si l'instance est froide)...")
    payload = {
        "patient_id": "conv-user",
        "history": [{"role": "user", "content": "Bonjour"}],
        "stream": False,
    }
    try:
        await client.post(f"{API_URL}/chat", json=payload, timeout=120)
        print("✅ Instance prête.")
    except Exception as e:
        print(f"⚠️  Le warmup a échoué : {e}. Le benchmark continue...")


async def measure_one(client: httpx.AsyncClient, prompt: str, category: str):
    """Mesure une requête vers l'API Cloud."""
    payload = {
        "patient_id": "PAT-999",
        "history": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    t_start = time.perf_counter()
    try:
        r = await client.post(f"{API_URL}/chat", json=payload, timeout=300)
        t_end = time.perf_counter()
        r.raise_for_status()  # Lève une exception pour les codes 4xx/5xx
        response_data = r.json()

        return {
            "category": category,
            "latency_ms": round((t_end - t_start) * 1000, 2),
            "component_latencies": {
                k: round(v * 1000, 2)
                for k, v in response_data.get("latencies", {}).items()
            },
            "status_code": r.status_code,
            "response_length": len(response_data.get("response", "")),
            "success": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        return {
            "category": category,
            "latency_ms": None,
            "error": str(e),
            "success": False,
        }


async def main():
    prompts = [json.loads(line) for line in open(PROMPTS_FILE)]
    async with httpx.AsyncClient() as client:
        await warmup(client)

        results = []
        for i in range(N_REPEAT):
            for p in prompts:
                print(f"  Run {i + 1}/{N_REPEAT} — {p['category']}...", end=" ")
                r = await measure_one(client, p["prompt"], p["category"])
                print(f"{r.get('latency_ms')} ms" if r.get("latency_ms") else "FAIL")
                results.append(r)

    Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    lats = [r["latency_ms"] for r in results if r.get("latency_ms")]
    if lats:
        print(
            f"\n📊 p50={statistics.median(lats):.0f}ms "
            f"p95={sorted(lats)[int(len(lats) * 0.95)]:.0f}ms "
            f"max={max(lats):.0f}ms"
        )


if __name__ == "__main__":
    asyncio.run(main())
