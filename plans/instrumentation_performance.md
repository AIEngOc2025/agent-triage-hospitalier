# Plan d'implémentation : Instrumentation de performance détaillée

## Objectif
Mettre en place un système de mesure de latence robuste et granulaire pour identifier les goulots d'étranglement par composant (Réseau, Presidio, vLLM, Audit) dans l'API de triage.

## Fichiers à créer/modifier

| Étape | Fichier | Type modif | Description |
| :--- | :--- | :--- | :--- |
| 1 | `app/timing.py` | Création | Définition des utilitaires de mesure (contexte manager/décorateur). |
| 2 | `app/middleware_timing.py` | Création | Middleware FastAPI pour mesurer la latence réseau globale. |
| 3 | `app/main.py` | Modification | Intégration du middleware et injection du contexte de timing. |
| 4 | `app/remote/client.py` | Modification | Wrapping des méthodes `generate` et `generate_stream` pour mesurer le temps réseau. |
| 5 | `scripts/analyze_timing.py` | Création | Script d'analyse des logs générés pour calculer les percentiles. |
| 6 | `scripts/benchmark/run_with_timing.py` | Création | Orchestrateur de test automatisé avec capture de timings. |

## Étapes d'implémentation

1.  **Création du module `app/timing.py`** : Centraliser la logique de mesure de temps.
2.  **Mise en place du `app/middleware_timing.py`** : Capturer la latence totale par requête.
3.  **Intégration** : Enregistrer le middleware dans `app/main.py`.
4.  **Instrumentation métier** : Envelopper les appels réseau et Presidio dans `app/remote/client.py` et `app/api_utils.py` (via les nouveaux utilitaires).
5.  **Analyse** : Développer le script d'analyse pour transformer les logs en rapport (format Markdown/Tableau).

## Vérification et Tests
- [ ] Valider que les logs contiennent bien les champs `vllm_ms`, `presidio_ms`, `network_ms`.
- [ ] Vérifier que la somme des composants est cohérente avec la latence totale.
- [ ] Générer un rapport de performance avec `scripts/analyze_timing.py` après une exécution de benchmark.

---
*Approbation requise avant de procéder à la création des fichiers.*
