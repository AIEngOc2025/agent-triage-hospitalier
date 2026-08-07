# Benchmark de latence — Agent de Triage CHSA

  ## Protocole

  1. **Pré-requis** :
     - Local : `docker-compose up` (inference + api + ui)
     - Cloud : service déployé et accessible

  2. **Lancement** :
     ```bash
     # Local + Cloud + analyse
     python scripts/benchmark/benchmark_all.py

     # Seulement local
     python scripts/benchmark/benchmark_all.py --skip-cloud

     # Seulement cloud
     python scripts/benchmark/benchmark_all.py --skip-local

  3. Prompts utilisés :
    - 8 catégories (salutation, symptôme, urgence vitale, etc.)
    - 5 répétitions par catégorie = 40 mesures
    - 1 mesure cold start = 41 mesures
  4. Métriques calculées :
    - Min / Max / Moyenne / Médiane
    - p50 / p90 / p95 / p99
    - Par catégorie et global
    - Comparaison local vs cloud
  5. Reproductibilité :
    - Mêmes prompts pour local et cloud
    - Fichiers de résultats horodatés
    - Analyse reproductible via analyze_results.py

  Lecture des résultats

  ┌────────────┬────────────────────────────────┐
  │    p95     │         Interprétation         │
  ├────────────┼────────────────────────────────┤
  │ < 500ms    │ Excellent pour UX streaming    │
  ├────────────┼────────────────────────────────┤
  │ 500ms – 1s │ Acceptable pour chat           │
  ├────────────┼────────────────────────────────┤
  │ 1s – 3s    │ Limite pour UX acceptable      │
  ├────────────┼────────────────────────────────┤
  │ > 3s       │ Mauvais pour triage temps réel │
  └────────────┴────────────────────────────────┘


  