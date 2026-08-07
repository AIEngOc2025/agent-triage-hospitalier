import subprocess
import time

def run_benchmark():
    print("🚀 Démarrage du benchmark avec instrumentation...")
    # Lancer le script de test existant qui appelle l'API instrumentée
    subprocess.run(["python", "scripts/benchmark/benchmark_cloud.py"], check=True)
    
    print("\n📈 Analyse des latences des composants...")
    # Analyser le log de résultats généré par benchmark_cloud.py
    subprocess.run(["python", "scripts/analyze_timing.py", "scripts/benchmark/data/results_cloud.jsonl"])

if __name__ == "__main__":
    run_benchmark()
