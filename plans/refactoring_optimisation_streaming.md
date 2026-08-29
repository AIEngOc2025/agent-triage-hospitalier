# Plan de Refactorisation et Optimisation - Agent Triage Hospitalier

## 1. Objectifs
- **Centraliser la logique métier** : Unifier le triage dans l'orchestrateur pour éliminer la redondance entre `/triage` et `/chat`.
- **Stabiliser le streaming** : Adopter un pattern hybride ("Reasoning" en streaming + "JSON" final) pour garantir la robustesse face aux limites de jetons tout en gardant l'interactivité.
- **Améliorer la performance** : Réduire le temps de latence au premier jeton (Time to First Token) et fiabiliser la production (gérer les cold starts).

## 2. Étapes de Mise en Œuvre

### Phase 1 : Consolidation de la logique (Refactoring)
- **Objectif** : Faire de `TriageAgentOrchestrator` la source de vérité.
- **Tâches** :
  - Déplacer la logique de `api_triage` (`app/main.py`) vers `TriageAgentOrchestrator`.
  - Intégrer les vérifications de `triage_veto` et `nlp_classifier` directement dans le workflow de l'orchestrateur.
  - Nettoyer `app/main.py` pour qu'il ne contienne que la gestion des requêtes HTTP/Streaming.

### Phase 2 : Pattern hybride pour le Streaming
- **Objectif** : Supprimer l'erreur `IncompleteOutputException` tout en gardant la structure.
- **Tâches** :
  - **Étape A** : Streamer le champ `reasoning` (texte libre).
  - **Étape B** : Envoyer un délimiteur (ex: `###JSON_START###`).
  - **Étape C** : Envoyer la réponse structurée finale (JSON/Pydantic) en un bloc pour validation.
  - Mettre à jour `ui_agent_streamlit.py` pour interpréter ce flux hybride.

### Phase 3 : Fiabilisation de la production
- **Objectif** : Éviter les timeouts et erreurs de déploiement.
- **Tâches** :
  - Ajouter un monitoring de santé (health check) strict sur le service d'inférence avant de démarrer le gateway.
  - Implémenter un log d'audit plus granulaire qui capture les raisons de troncature de jetons avant qu'elles ne deviennent des erreurs 500.

## 3. Vérification et Validation
- **Tests Unitaires** : Tester l'orchestrateur isolément avec le nouveau flux hybride.
- **Test d'Intégration** : Utiliser `scripts/test_functional.py` pour valider le streaming complet.
- **Test de charge léger** : Simuler une réponse longue pour valider la gestion de la limite de jetons.
