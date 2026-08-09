"""
Benchmark LOCAL — appelle l'API FastAPI sur localhost:8000.

Mesure :
    - Démarrage du service (cold start + warm start)
    - Latence par requête
    - Latence par composant (instrumentation via hooks d'API)
"""

import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

API_URL = "http://localhost:8000"
PROMPTS_FILE = "scripts/benchmark/data/benchmark_prompts.jsonl"
RESULTS_FILE = "scripts/benchmark/data/results_local.jsonl"
N_REPEAT = 5


async def warmup(client: httpx.AsyncClient):
    """Une requête de warmup pour amorcer vLLM."""
    payload = {
        "patient_id": "conv-user",
        "history": [{"role": "user", "content": "Bonjour"}],
        "stream": False,
    }
    await client.post(f"{API_URL}/chat", json=payload, timeout=300)


async def measure_one(client: httpx.AsyncClient, prompt: str, category: str):
    """Mesure une requête."""
    payload = {
        "patient_id": "PAT-999",
        "history": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    t_start = time.perf_counter()
    try:
        r = await client.post(f"{API_URL}/chat", json=payload, timeout=300)
        t_end = time.perf_counter()
        return {
            "category": category,
            "latency_ms": round((t_end - t_start) * 1000, 2),
            "status_code": r.status_code,
            "response_length": len(r.json().get("response", "")),
            "success": r.status_code == 200,
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
        print("🔥 Warmup...")
        await warmup(client)

        results = []
        for i in range(N_REPEAT):
            for p in prompts:
                print(f"  Run {i + 1}/{N_REPEAT} — {p['category']}...", end=" ")
                r = await measure_one(client, p["prompt"], p["category"])
                print(f"{r.get('latency_ms')} ms" if r.get("latency_ms") else "FAIL")
                results.append(r)

        # Cold start measurement
        print("❄️   Cold start (restart service manually before this)...")
        await asyncio.sleep(2)
        cold = await measure_one(client, "Bonjour", "cold_start")
        results.append(cold)

    Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Stats rapides
    lats = [r["latency_ms"] for r in results if r.get("latency_ms")]
    if lats:
        print(
            f"\n📊 p50={statistics.median(lats):.0f}ms "
            f"p95={sorted(lats)[int(len(lats) * 0.95)]:.0f}ms "
            f"max={max(lats):.0f}ms"
        )


if __name__ == "__main__":
    asyncio.run(main())
