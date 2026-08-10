import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def percentile(data, p):
    if not data:
        return 0
    s_data = sorted(data)
    k = (len(s_data) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s_data) - 1)
    return s_data[f] + (s_data[c] - s_data[f]) * (k - f)


def analyze(file_path: str):
    if not Path(file_path).exists():
        print(f"⚠️ Fichier non trouvé : {file_path}")
        return

    results = [json.loads(line) for line in open(file_path)]

    # Agrégation des latences par composant
    stats = defaultdict(list)
    total_latencies = []

    for r in results:
        if not r.get("success"):
            continue

        total_latencies.append(r["latency_ms"])

        comp_lat = r.get("component_latencies", {})
        for c, val in comp_lat.items():
            stats[c].append(val)

    print(f"\n📊 Analyse détaillée : {file_path}")
    print(f"{'Composant':<20} {'Moy (ms)':>10} {'p50 (ms)':>10} {'p95 (ms)':>10}")
    print(f"{'-' * 50}")
    total_mean = statistics.mean(total_latencies)
    total_med = statistics.median(total_latencies)
    total_p95 = percentile(total_latencies, 95)
    print(
        f"{'TOTAL (Réseau)':<20} {total_mean:>10.0f} {total_med:>10.0f} "
        f"{total_p95:>10.0f}"
    )

    for c, vals in stats.items():
        if not vals:
            continue
        c_mean = statistics.mean(vals)
        c_med = statistics.median(vals)
        c_p95 = percentile(vals, 95)
        print(f"{c:<20} {c_mean:>10.0f} {c_med:>10.0f} {c_p95:>10.0f}")


if __name__ == "__main__":
    file_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "scripts/benchmark/data/results_cloud.jsonl"
    )
    analyze(file_path)
