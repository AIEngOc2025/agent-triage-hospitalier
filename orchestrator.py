import argparse
import json
import logging
import subprocess
from pathlib import Path

# Configure logging for the orchestrator
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProjectOrchestrator:
    """
    @definition : Orchestrates project operations such as deployment, status checks,
                  and technical documentation generation.
    @args/params : None
    @return : None
    """

    SERVICES = {
        "api": "cloudbuild.api.yaml",
        "inference": "cloudbuild.inference.yaml",
        "ui": "cloudbuild.ui.yaml",
    }
    AUDIT_LOG_FILE = Path("logs/audit_medical.jsonl")
    TECHNICAL_OVERVIEW_FILE = "TECHNICAL_OVERVIEW.md"

    def calculate_metrics(self):
        log_file = Path("logs/audit_medical.jsonl")
        if not log_file.exists():
            return 0.0, 0

        total_latency = 0.0
        latencies = []
        count = 0
        try:
            with open(self.AUDIT_LOG_FILE, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        latency = entry.get("latency_sec")
                        if isinstance(latency, (int, float)):
                            total_latency += latency
                            latencies.append(latency)
                            count += 1
                        else:
                            logger.warning(
                                f"Skipping log entry with invalid latency: {line.strip()}"
                            )
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed log entry: {line.strip()}")
                        continue
        except FileNotFoundError:
            logger.warning(
                f"Audit log file not found at {self.AUDIT_LOG_FILE}. Returning default metrics."
            )
            return 0.0, 0
        except Exception as e:
            logger.error(f"Error reading audit log file {self.AUDIT_LOG_FILE}: {e}")
            return 0.0, 0

        avg_latency = total_latency / count if count > 0 else 0.0
        # For a more robust system, consider calculating p95/p99 latencies here.
        return round(avg_latency * 1000, 2), count  # Convert to ms

    def generate_technical_overview(self):
        """
        @definition : Generates the TECHNICAL_OVERVIEW.md file with project metrics and roadmap.
        @args/params : None
        @return : None
        """
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
| **Latence API Gateway** | < 200ms (p95) | {avg_latency_ms if count > 0 else "N/A"} ms | Logs d'audit (Moyenne) |
| **Précision du Triage** | > 90% | À auditer | Évaluation dataset test |
| **Anonymisation PII** | > 99% | À valider | Tests `test_audit.py` |
| **Disponibilité** | > 99.9% | - | Monitoring `/health` |

## 4. Roadmap
- Court terme : Validation clinique sur site.
- Long terme : Passage à l'échelle (32B+ paramètres).
"""
        with open(self.TECHNICAL_OVERVIEW_FILE, "w") as f:
            f.write(content)
        logger.info(
            f"✅ {self.TECHNICAL_OVERVIEW_FILE} généré (Latence moyenne: {avg_latency_ms} ms sur {count} échantillons)."
        )

    def deploy(self, service):
        """
        @definition : Deploys a specified service to Google Cloud Run via Cloud Build.
        @args/params :
            - service (str): The name of the service to deploy ('api', 'inference', 'ui').
        @return : None
        """
        if service not in self.SERVICES:
            logger.error(f"❌ Service inconnu. Choix: {list(self.SERVICES.keys())}")
            return
        config = self.SERVICES[service]
        logger.info(f"🚀 Déploiement du service '{service}' avec {config}...")
        try:
            subprocess.run(
                ["gcloud", "builds", "submit", "--config", config, "."],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"✅ Déploiement du service '{service}' terminé avec succès.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erreur lors du déploiement du service '{service}':")
            logger.error(f"  Commande: {e.cmd}")
            logger.error(f"  Code de retour: {e.returncode}")
            logger.error(f"  Sortie standard: {e.stdout}")
            logger.error(f"  Erreur standard: {e.stderr}")
        except FileNotFoundError:
            logger.error(
                "❌ 'gcloud' command not found. Please ensure Google Cloud SDK is installed and configured."
            )
        except Exception as e:
            logger.error(
                f"❌ Une erreur inattendue est survenue lors du déploiement: {e}"
            )

    def check_status(self):
        """
        @definition : Displays the status of the last 5 Google Cloud Builds.
        @args/params : None
        @return : None
        """
        logger.info("📊 Statut des 5 derniers builds :")
        try:
            subprocess.run(
                [
                    "gcloud",
                    "builds",
                    "list",
                    "--limit",
                    "5",
                    "--format=table(id, status, startTime)",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erreur lors de la récupération du statut des builds: {e}")
        except FileNotFoundError:
            logger.error(
                "❌ 'gcloud' command not found. Please ensure Google Cloud SDK is installed and configured."
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
