"""
Orchestrateur de test : exécute un bench tout en capturant les logs pour analyse.

Usage : python scripts/benchmark/run_with_timing.py <script_a_benchmarker>
"""

import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/benchmark/run_with_timing.py <script_a_benchmarker>"
        )
        sys.exit(1)

    script_to_run = sys.argv[1]
    log_file = "logs/benchmark_timing.log"
    Path("logs").mkdir(exist_ok=True)

    print(f"🚀 Lancement du benchmark avec capture de timing : {script_to_run}")

    # Exécution avec redirection des logs (on laisse stdout du script affiché pour le suivi)
    with open(log_file, "w") as f:
        # Note: on loggue stderr/stdout dans le fichier pour l'analyse
        subprocess.run(
            [sys.executable, script_to_run], stdout=f, stderr=subprocess.STDOUT
        )

    print(f"✅ Benchmark terminé. Logs sauvegardés dans {log_file}")

    # Analyse immédiate
    print("\n🔍 Analyse des performances capturées :")
    subprocess.run([sys.executable, "scripts/analyze_timing.py", log_file])


if __name__ == "__main__":
    main()
