# Technical Overview : Agent de Triage Hospitalier

## 1. Vision et Objectifs
Ce projet vise à fournir une solution d'IA pour le triage hospitalier,
garantissant précision, rapidité et conformité RGPD.

## 2. Architecture Technique
Architecture découplée en 3 services : API Gateway (FastAPI),
Inference Engine (vLLM), et Frontend UI (Streamlit).

## 3. Métriques de Performance (Vérifiables)
*Données calculées à partir de 0 interactions enregistrées.*

| Métrique | Cible / Objectif | Valeur Actuelle | Méthode de vérification |
| :--- | :--- | :--- | :--- |
| **Latence API Gateway** | < 200ms (p95) | N/A ms | Logs d'audit (Moyenne) |
| **Précision du Triage** | > 90% | À auditer | Évaluation dataset test |
| **Anonymisation PII** | > 99% | À valider | Tests `test_audit.py` |
| **Disponibilité** | > 99.9% | - | Monitoring `/health` |

## 4. Roadmap
- Court terme : Validation clinique sur site.
- Long terme : Passage à l'échelle (32B+ paramètres).
