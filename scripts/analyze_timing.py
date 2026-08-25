import re
import sys
from collections import defaultdict
from statistics import mean, median


def analyze_logs(log_file: str):
    metrics = defaultdict(list)
    # Pattern pour extraire les logs de métriques : ⏱️  [METRIC] label: value ms
    pattern = re.compile(
        r"⏱️\s+\[METRIC\]\s+(?P<label>[\w\s_]+):\s+(?P<value>[\d\.]+)ms"
    )

    try:
        with open(log_file, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    label = match.group("label").strip()
                    value = float(match.group("value"))
                    metrics[label].append(value)
    except FileNotFoundError:
        print(f"Erreur : Fichier {log_file} introuvable.")
        return

    print(f"--- Analyse des Performances ({log_file}) ---")
    for label, values in metrics.items():
        if not values:
            continue
        print(f"\nMetric: {label}")
        print(f"  Count:  {len(values)}")
        print(f"  Min:    {min(values):.2f} ms")
        print(f"  Max:    {max(values):.2f} ms")
        print(f"  Mean:   {mean(values):.2f} ms")
        print(f"  Median: {median(values):.2f} ms")


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "logs/app.log"
    analyze_logs(log_path)
