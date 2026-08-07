# Rapport de Performance - Agent de Triage

Date : 2026-08-07
Cible : Cloud Run (API Gateway)

## Analyse détaillée par composant

| Composant | Moyenne (ms) | p50 (ms) | p95 (ms) |
| :--- | :--- | :--- | :--- |
| **TOTAL (Réseau)** | **774** | **840** | **944** |
| **vLLM** (Inférence) | 674 | 754 | 763 |
| **Audit** | 23 | 22 | 33 |
| **Presidio** (Anonymisation) | 6 | 6 | 7 |

## Conclusion
Le moteur vLLM représente le goulot d'étranglement principal (>85% de la latence). Les composants de sécurité (Audit, Presidio) sont optimisés et ne dégradent pas significativement l'expérience utilisateur.
