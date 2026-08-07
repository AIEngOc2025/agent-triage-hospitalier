"""
Analyse des résultats de benchmark :
- Génère des statistiques (percentiles, moyenne, etc.) pour chaque fichier.
- Compare les performances local vs. cloud si les deux fichiers sont fournis.
"""
import json
import sys
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(data, p):
    """Calcul du percentile sans dépendance externe comme numpy."""
    if not data:
        return 0
    s_data = sorted(data)
    k = (len(s_data) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s_data) - 1)
    if f == c:
        return s_data[f]
    return s_data[f] + (s_data[c] - s_data[f]) * (k - f)


def analyze(file_path: str):
    """Analyse un fichier de résultats et affiche les statistiques."""
    if not Path(file_path).exists():
        print(f"⚠️  Fichier non trouvé : {file_path}. Analyse ignorée.")
        return None

    results = [json.loads(l) for l in open(file_path)]
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]

    print(f"\n{'='*60}")
    print(f"📊 Analyse de : {file_path}")
    print(f"{'='*60}")

    if not latencies:
        print("  Aucune requête réussie à analyser.")
        return None

    print(f"  Total requêtes : {len(results)}")
    print(f"  Réussies       : {len(latencies)}")
    print(f"  Échecs         : {len(results) - len(latencies)}")
    print("  ---")
    print(f"  Min            : {min(latencies):.0f} ms")
    print(f"  Max            : {max(latencies):.0f} ms")
    print(f"  Moyenne        : {statistics.mean(latencies):.0f} ms")
    print(f"  Médiane (p50)  : {statistics.median(latencies):.0f} ms")
    print(f"  p90            : {percentile(latencies, 90):.0f} ms")
    print(f"  p95            : {percentile(latencies, 95):.0f} ms")
    print(f"  p99            : {percentile(latencies, 99):.0f} ms")

    # Analyse par catégorie
    by_cat = defaultdict(list)
    for r in results:
        if r.get("latency_ms"):
            by_cat[r["category"]].append(r["latency_ms"])

    print(f"\n  {'Par catégorie':<25} {'n':>4} {'moy':>8} {'p95':>8} {'max':>8}")
    print(f"  {'-'*25} {'-'*4} {'-'*8} {'-'*8} {'-'*8}")
    for cat, vals in sorted(by_cat.items()):
        print(f"  {cat:<25} {len(vals):>4} "
              f"{statistics.mean(vals):>7.0f}ms "
              f"{percentile(vals, 95):>7.0f}ms "
              f"{max(vals):>7.0f}ms")

    return {
        "p50": statistics.median(latencies),
        "p95": percentile(latencies, 95),
        "mean": statistics.mean(latencies),
    }


def compare(local_stats, cloud_stats):
    """Compare les statistiques locales et cloud."""
    print(f"\n{'='*60}")
    print("🔄 COMPARAISON Local vs. Cloud")
    print(f"{'='*60}")
    print(f"{'Métrique':<10} {'Local (ms)':>12} {'Cloud (ms)':>12} {'Ratio':>8}")
    print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*8}")
    for metric in ["p50", "p95", "mean"]:
        ratio = cloud_stats[metric] / local_stats[metric] if local_stats[metric] else float('inf')
        print(f"{metric.upper():<10} {local_stats[metric]:>12.0f} {cloud_stats[metric]:>12.0f} {ratio:>7.2f}x")


if __name__ == "__main__":
    local_file = "scripts/benchmark/data/results_local.jsonl"
    cloud_file = "scripts/benchmark/data/results_cloud.jsonl"

    local_stats = analyze(local_file)
    cloud_stats = analyze(cloud_file)

    if local_stats and cloud_stats:
        compare(local_stats, cloud_stats)