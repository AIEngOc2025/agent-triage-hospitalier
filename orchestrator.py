import argparse
import json
import subprocess
from pathlib import Path


class ProjectOrchestrator:
    SERVICES = {
        "api": "cloudbuild.api.yaml",
        "inference": "cloudbuild.inference.yaml",
        "ui": "cloudbuild.ui.yaml",
    }

    def calculate_metrics(self):
        log_file = Path("logs/audit_medical.jsonl")
        if not log_file.exists():
            return 0.0, 0

        total_latency = 0.0
        count = 0
        with open(log_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    total_latency += entry.get("latency_sec", 0.0)
                    count += 1
                except json.JSONDecodeError:
                    continue

        avg_latency = total_latency / count if count > 0 else 0.0
        return round(avg_latency * 1000, 2), count  # Conversion en ms

    def generate_technical_overview(self):
        avg_latency_ms, count = self.calculate_metrics()

        content = f"""# Technical Overview : Agent de Triage Hospitalier

## 1. Vision et Objectifs
Ce projet vise à fournir une solution d'IA pour le triage hospitalier,
garantissant précision, rapidité et conformité RGPD.

## 2. Architecture Technique
Architecture découplée en 3 services : API Gateway (FastAPI),
Inference Engine (vLLM), et Frontend UI (Streamlit).

## 3. Métriques de Performance (Vérifiables)
*Données calculées à partir de {count} interactions enregistrées.*

| Métrique | Cible / Objectif | Valeur Actuelle | Méthode de vérification |
| :--- | :--- | :--- | :--- |
| **Latence API Gateway** | < 200ms (p95) | {avg_latency_ms} ms | Logs d'audit |
| **Précision du Triage** | > 90% | À auditer | Évaluation test |
| **Anonymisation PII** | > 99% | À valider | Tests `test_audit.py` |
| **Disponibilité** | > 99.9% | - | Monitoring `/health` |

## 4. Roadmap
- Court terme : Validation clinique sur site.
- Long terme : Passage à l'échelle (32B+ paramètres).
"""
        with open("TECHNICAL_OVERVIEW.md", "w") as f:
            f.write(content)
        print(
            f"✅ TECHNICAL_OVERVIEW.md généré (Latence moyenne: {avg_latency_ms} ms)."
        )

    def deploy(self, service):
        if service not in self.SERVICES:
            print(f"❌ Service inconnu. Choix: {list(self.SERVICES.keys())}")
            return
        config = self.SERVICES[service]
        print(f"🚀 Déploiement du service '{service}' avec {config}...")
        subprocess.run(
            ["gcloud", "builds", "submit", "--config", config, "."], check=True
        )

    def check_status(self):
        print("📊 Statut des 5 derniers builds :")
        subprocess.run(
            [
                "gcloud",
                "builds",
                "list",
                "--limit",
                "5",
                "--format=table(id, status, startTime)",
            ]
        )


def main():
    parser = argparse.ArgumentParser(description="Orchestrateur du projet Triage")
    parser.add_argument("action", choices=["docs", "deploy", "status"])
    parser.add_argument("--service", choices=["api", "inference", "ui"])

    args = parser.parse_args()
    orch = ProjectOrchestrator()

    if args.action == "docs":
        orch.generate_technical_overview()
    elif args.action == "deploy":
        if not args.service:
            print("❌ Veuillez spécifier un service avec --service")
        else:
            orch.deploy(args.service)
    elif args.action == "status":
        orch.check_status()


if __name__ == "__main__":
    main()
