# Plan d'implémentation : Classifieur NLP bilingue (Fine-tuning)

## Objectif
Entraîner un modèle `distil-xlm-roberta-base` (bilingue) pour le triage médical et l'intégrer comme guardrail déterministe dans l'API.

## Étapes d'implémentation

1.  **Création du dataset étiqueté** : 
    - Développer `scripts/label_triage_data.py` pour étiqueter automatiquement 100-500 exemples à l'aide de l'API LLM existante.
2.  **Fine-tuning** : 
    - Développer `scripts/train_nlp_classifier.py` pour entraîner `distil-xlm-roberta-base` sur `data/processed/labeled_triage.jsonl`.
    - Sauvegarder le modèle entraîné dans `models/triage_nlp_model/`.
3.  **Inférence (Classifieur)** : 
    - Mettre à jour `app/nlp_triage.py` pour charger le modèle fine-tuné et l'utiliser dans la méthode `predict`.
4.  **Intégration API** : 
    - Mettre à jour `app/main.py` pour utiliser le classifieur NLP avant ou en parallèle de l'appel LLM.

## Vérification et Tests
- [ ] Vérifier que `scripts/label_triage_data.py` génère un fichier valide.
- [ ] Valider que `scripts/train_nlp_classifier.py` termine sans erreur.
- [ ] Confirmer que `app/nlp_triage.py` prédit correctement les niveaux de triage sur de nouveaux exemples.
