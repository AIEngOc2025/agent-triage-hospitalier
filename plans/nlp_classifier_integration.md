# Plan d'implémentation : Intégration du Classifieur NLP de Triage

## Objectif
Entraîner un modèle `distil-xlm-roberta-base` (bilingue FR/EN) pour le triage médical et l'intégrer comme garde-fou déterministe dans l'API `/triage`.

## Étapes d'implémentation

### 1. Préparation des données (`scripts/label_triage_data.py`)
- Mettre à jour `scripts/label_triage_data.py` pour cibler l'endpoint `/triage`.
- Collecter `niveau`, `raison`, et `confidence` du LLM pour 300 exemples.
- Générer `data/processed/labeled_triage.jsonl`.
- Produire un rapport de relecture sur 30 exemples (stratifié par confiance).

### 2. Entraînement (`scripts/train_nlp_classifier.py`)
- Créer `scripts/train_nlp_classifier.py`.
- Charger `data/processed/labeled_triage.jsonl`.
- Effectuer un split stratifié (train/val/test).
- Fine-tuner `distil-xlm-roberta-base`.
- Sauvegarder dans `models/triage_nlp_model/`.
- Générer `reports/nlp_classifier_eval.json`.

### 3. Intégration API (`app/nlp_triage.py`)
- Mettre à jour `TriageClassifier` pour charger le modèle local (`models/triage_nlp_model/`).
- Implémenter le fallback `zero-shot` si le modèle local est absent.

### 4. Veto (`app/main.py`)
- Mettre à jour `api_triage` pour appeler le classifieur.
- Implémenter la logique de veto bidirectionnel (décide_veto déjà en place, à brancher).

### 5. Tests
- Ajouter des tests de bout en bout dans `tests/test_nlp_triage.py` et `tests/test_triage_veto.py`.

## Vérification
- [ ] Dataset généré ≥300 exemples.
- [ ] Accuracy du modèle > 85%.
- [ ] Tests unitaires et d'intégration réussis (`uv run pytest tests/`).
