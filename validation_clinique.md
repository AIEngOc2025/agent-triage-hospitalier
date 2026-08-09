# Plan de Validation Clinique : Agent de Triage Hospitalier

## 1. Objectif
Assurer la sécurité clinique et la conformité du POC de triage aux protocoles du CHSA.

## 2. Étapes d'Implémentation

### Étape 1 : Constitution du Golden Set
- Créer `data/golden_set.jsonl`.
- Sélectionner 25 cas critiques (5 Maximale, 10 Modérée, 10 Différée) représentatifs.
- Chaque cas doit être validé avec une classification "Ground Truth".

### Étape 2 : Automatisation de l'évaluation
- Adapter `scripts/evaluate/quantitative_matrix.py` pour évaluer spécifiquement ce `golden_set.jsonl`.
- Sortir un rapport de conformité (Précision par zone de triage).

### Étape 3 : Tests de sécurité (Red Flags)
- Créer un sous-ensemble "Red Flags" (cas d'urgence vitale).
- Vérifier que le modèle ne propose JAMAIS de priorité inférieure à "Maximale".

### Étape 4 : Documentation & Soutenance
- Mettre à jour `rapport_technique.md` avec les résultats de cette validation.
- Préparer les slides de soutenance basées sur ces preuves cliniques.

## 3. Livrables
- Fichier `data/golden_set.jsonl`.
- Rapport d'évaluation clinique intégré au rapport technique.
- Plan de remédiation en cas d'échec sur les "Red Flags".
