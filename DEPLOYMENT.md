# Documentation du Déploiement : Agent de Triage Hospitalier

Cette documentation détaille les procédures de déploiement et d'exploitation du système d'agent de triage (Architecture Consolidée à 3 services).

## 1. Architecture Consolidée
Le système utilise une architecture découplée pour le build et le déploiement :
- **Backend API (`agent-triage-hospitalier`) :** FastAPI, déploie via `cloudbuild.api.yaml`.
- **Inference Engine (`agent-inference-service`) :** vLLM/GPU, déploie via `cloudbuild.inference.yaml`.
- **Frontend UI (`agent-triage-ui`) :** Streamlit, déploie via `cloudbuild.ui.yaml`.

## 2. CI/CD : Déploiement Cloud Build
Chaque service possède sa propre configuration de build indépendante.

### Procédure de déploiement (Plan d'action)
1. **Déploiement du Service d'Inférence :**
   ```bash
   gcloud builds submit --config cloudbuild.inference.yaml .
   ```
2. **Déploiement de l'API Gateway :**
   ```bash
   gcloud builds submit --config cloudbuild.api.yaml .
   ```
3. **Déploiement du Frontend (UI) :**
   ```bash
   gcloud builds submit --config cloudbuild.ui.yaml .
   ```

## 3. Configuration & Variables d'environnement
| Variable | Description | Exemple |
| :--- | :--- | :--- |
| `REMOTE_INFERENCE_ENABLED` | Activer l'inférence distante dans l'API | `true` |
| `INFERENCE_SERVICE_URL` | URL du service `agent-inference-service` | `https://...` |
| `MODEL_PATH` | Chemin GCS (pour le service d'inférence) | `gs://bucket/model` |
| `APP_ENV` | Environnement | `production` |

## 4. Dépannage
- **Erreur `NameError: os not defined` :** Vérifier que `import os` est présent en haut de `app/main.py`.
- **Échec au démarrage (Timeout) :** Augmenter le délai de santé (`--startup-probe-timeout`) dans le fichier Cloud Build correspondant.
- **Quota GPU :** S'assurer de disposer des quotas nécessaires pour les types de GPU requis (`nvidia-l4`).
