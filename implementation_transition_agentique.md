# Plan d'Implémentation : Transition Agentique

Ce document détaille les étapes techniques pour transformer le POC actuel en une architecture agentique contrôlée.

## 1. Phase 1 : Consolidation des Outils (`app/agent_tools.py`)
*   **Objectif :** Créer une bibliothèque d'outils typés pour le LLM.
*   **Tâches :**
    *   Adapter `classify_triage_urgency` pour qu'il soit compatible avec l'interface d'outil de l'agent.
    *   Implémenter `ClinicalVetoTool` (interface de pause).
    *   Ajouter `AnonymizerTool` utilisant `app.api_utils.anonymize_text`.
    *   Vérification rigoureuse des docstrings (format requis).
    *   **Linting :** `uv run ruff check app/agent_tools.py` et `uv run ruff format app/agent_tools.py`.

## 2. Phase 2 : Orchestrateur (`app/agent_orchestrator.py`)
*   **Objectif :** Créer la machine à états gérant le flux sécurisé.
*   **Tâches :**
    *   Définir les états : `START`, `ANONYMIZATION`, `NLP_CLASSIFICATION`, `LLM_SYNTHESIS`, `VETO_WAIT`, `FINALIZATION`.
    *   Implémenter la transition forcée : `ANONYMIZATION` -> `VETO_WAIT` est obligatoire.
    *   Gestion du `State` : Implémenter une classe pour stocker le contexte de la conversation, les résultats intermédiaires et le statut du veto.
    *   **Linting :** `uv run ruff check app/agent_orchestrator.py` et `uv run ruff format app/agent_orchestrator.py`.

## 3. Phase 3 : Interface Utilisateur (`ui_streamlit.py`)
*   **Objectif :** Rendre le raisonnement et le veto humain interactifs.
*   **Tâches :**
    *   Intégrer `st.expander` pour loguer les appels d'outils et la chaîne de pensée.
    *   Ajouter un bloc de validation clinique (`st.radio` ou `st.button` pour Accepter/Refuser) quand l'état est `VETO_WAIT`.
    *   Mettre à jour la logique de session pour supporter la reprise après veto.

## 4. Phase 4 : Validation (`tests/test_agent_triage.py`)
*   **Objectif :** Assurer la robustesse clinique.
*   **Tâches :**
    *   Test unitaire : Vérifier que l'anonymisation supprime bien les PII.
    *   Test d'intégration : Simuler une conversation complète, vérifier l'arrêt sur le `VETO_WAIT` et la reprise après validation.
    *   Test de conformité : Vérifier qu'il est impossible de finaliser sans validation clinique.

## 5. Indicateurs de Succès
*   Code linté (0 erreur Ruff).
*   Documentation conforme (docstrings).
*   Flux de sécurité clinique respecté (100% des cas passent par veto).
