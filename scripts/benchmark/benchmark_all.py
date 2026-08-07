"""
Orchestrateur : lance le bench local + cloud + analyse.

Usage : python scripts/benchmark/benchmark_all.py [--skip-local] [--skip-cloud]
"""

import subprocess
import sys
from pathlib import Path


def main():
    skip_local = "--skip-local" in sys.argv
    skip_cloud = "--skip-cloud" in sys.argv

    if not skip_local:
        print("\n🖥️   Lancement benchmark LOCAL...")
        print("⚠️   Assurez-vous que docker-compose est démarré !")
        subprocess.run(
            [sys.executable, "scripts/benchmark/benchmark_local.py"], check=True
        )

    if not skip_cloud:
        print("\n☁️   Lancement benchmark CLOUD...")
        subprocess.run(
            [sys.executable, "scripts/benchmark/benchmark_cloud.py"], check=True
        )

    print("\n📊 Analyse...")
    local = "scripts/benchmark/data/results_local.jsonl"
    cloud = "scripts/benchmark/data/results_cloud.jsonl"

    files = []
    if Path(local).exists():
        files.append(local)
    if Path(cloud).exists():
        files.append(cloud)

    subprocess.run([sys.executable, "scripts/benchmark/analyze_results.py"] + files)


if __name__ == "__main__":
    main()
