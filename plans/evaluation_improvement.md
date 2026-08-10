# Plan : Amélioration du script d'évaluation

## Objectif
Rendre `scripts/evaluate/quantitative_matrix.py` plus robuste, performant et flexible.

## Modifications prévues
1.  **Fiabilisation de l'extraction** : Remplacer le `split` manuel par une expression régulière (`re.search`) pour extraire le tag `[niveau: ...]`.
2.  **Optimisation matérielle** : Ajouter une détection dynamique du périphérique (préférence GPU/MPS, repli sur CPU).
3.  **Flexibilité** : Implémenter `argparse` pour permettre de spécifier `model_id`, `adapters` et `test_file` en ligne de commande.
4.  **Nettoyage** : Ajuster le typage des données (`torch.float16` sur GPU) pour économiser la mémoire.

## Vérification et Tests
1.  Exécuter `uv run ruff format .` et `uv run ruff check .` pour garantir la conformité.
2.  Lancer une exécution de test pour valider que le script s'exécute correctement avec les nouveaux arguments.
