# Plan : Transition vers Streaming Intégral et Suppression du Mode Structuré

## Objectif
Finaliser la transition de l'agent de triage hospitalier vers une architecture exclusivement basée sur le streaming texte. Cela implique de supprimer la dépendance à `instructor` (mode structuré) dans l'orchestrateur pour éliminer les erreurs de limite de jetons (`IncompleteOutputException`) et simplifier la logique.

## État des lieux
- `/app/agent_orchestrator.py` appelle actuellement `engine.generate_structured`.
- `/app/main.py` gère maintenant `StreamingResponse`, mais l'orchestrateur sous-jacent dépend encore de `generate_structured`.
- Le passage au streaming texte pur nécessite de modifier la manière dont l'orchestrateur interagit avec l'engine.

## Étapes de mise en œuvre

### 1. Refactorisation de `TriageAgentOrchestrator` (`app/agent_orchestrator.py`)
- Remplacer l'appel à `engine.generate_structured` par une méthode de streaming texte (ex: `engine.generate_stream`).
- Mettre à jour `run` pour ne plus dépendre d'une réponse structurée mais traiter le flux textuel.
- Adapter `run_stream` pour qu'il consomme directement le générateur de jetons de l'engine.

### 2. Adaptation des System Prompts (`app/system_prompts.py`)
- S'assurer que `SYSTEM_PROMPT_JSON_FR` n'est plus forcé si on passe à du texte libre, ou le modifier pour demander explicitement un formatage compatible avec le streaming (ex: Markdown clair).

### 3. Mise à jour de l'API Gateway (`app/main.py`)
- Nettoyer le code : supprimer les gestionnaires d'erreurs `IncompleteOutputException` qui ne seront plus nécessaires en streaming pur.
- S'assurer que le pipeline de logs (`create_log_entry`) peut toujours gérer les logs d'audit avec les résultats textuels.

### 4. Vérification et Test
- Exécuter le script `scripts/test_functional.py` pour valider le flux complet.
- Vérifier que le comportement dans l'UI (`ui_agent_streamlit.py`) est conforme aux attentes.

## Risques identifiés
- **Perte de structure** : Si le frontend ou la base de données attendent un JSON rigide, il faudra parser le texte renvoyé par le LLM pour extraire les informations clés (niveau de triage, orientation).
- **Consistance des données** : Nécessite une rigueur accrue dans le prompt système pour garantir la qualité de la réponse textuelle.
