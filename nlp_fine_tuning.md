# Plan d'implémentation : Classifieur NLP bilingue (Fine-tuning)

## Objectif
Entraîner un modèle `distil-xlm-roberta-base` (bilingue FR/EN) pour le triage médical et l'intégrer comme **guardrail déterministe bidirectionnel** dans l'API `/triage`.

## Choix de conception (validés)

| Décision | Choix |
|---|---|
| Taille du dataset | **~200-500 exemples** |
| Granularité des labels | **Niveau de triage + raison brève** |
| Stratégie d'étiquetage | **LLM avec confiance + relecture échantillonnée** |
| Mode d'intégration API | **Veto bidirectionnel** |
| Modèle cible | `distil-xlm-roberta-base` (bilingue, léger, rapide) |

## État actuel vs plan

| Fichier | État | Action |
|---|---|---|
| `scripts/label_triage_data.py` | Existe (100 ex., niveau seul, via `/chat`) | **Compléter** : passer à 300 ex., capturer `/triage` (niveau + raison), ajouter score de confiance LLM |
| `scripts/train_nlp_classifier.py` | **Manquant** | **Créer** |
| `app/nlp_triage.py` | Existe mais utilise `bart-large-mnli` zero-shot | **Remplacer** par chargement du modèle fine-tuné |
| `app/main.py` | Aucun appel au classifieur | **Brancher** le veto bidirectionnel dans `/triage` |
| `models/triage_nlp_model/` | **Manquant** | Créer par le script d'entraînement |

## Étapes d'implémentation

### 1. Compléter `scripts/label_triage_data.py`
- Cible : **300 exemples** (équilibrés FR/EN, distribution proche des urgences réelles).
- Appeler l'endpoint **`/triage`** (pas `/chat`) pour récupérer :
  - `niveau` (maximale / modérée / différée)
  - `raison` (texte explicatif court)
  - `confidence` du LLM si disponible (sinon recalculer par re-run court)
- Format de sortie (`data/processed/labeled_triage.jsonl`) :
  ```json
  {"text": "...", "niveau": "maximale", "raison": "...", "llm_confidence": 0.82}
  ```
- **Relecture échantillonnée** : produire un rapport sur 30 exemples aléatoires (validation humaine / spot-check) → flag les cas ambigus (< 0.6 confiance).

### 2. Créer `scripts/train_nlp_classifier.py`
- Charger `data/processed/labeled_triage.jsonl`.
- Split **train/val/test** : 70 / 15 / 15 (stratifié par niveau).
- Fine-tuner `distil-xlm-roberta-base` (Hugging Face `Trainer`) sur la **classification du niveau uniquement** (la raison reste portée par le LLM).
- Sauvegarder dans `models/triage_nlp_model/` (config + poids + tokenizer).
- Logger les métriques (accuracy, F1 macro) dans `reports/nlp_classifier_eval.json`.

### 3. Mettre à jour `app/nlp_triage.py`
- Remplacer le pipeline `bart-large-mnli` par chargement du modèle fine-tuné local (`AutoModelForSequenceClassification`).
- Méthode `predict(text)` → retourne `{niveau, confiance}` + `model_version`.
- Garder le **fallback zero-shot** si le modèle fine-tuné est absent (mode dégradé).
- Singleton `triage_classifier` conservé.

### 4. Brancher le veto bidirectionnel dans `app/main.py`
- Dans l'endpoint `/triage`, **après** `generate_structured` :
  1. Appeler `triage_classifier.predict(user_input)` → obtient `nlp_niveau`, `nlp_conf`.
  2. Si `nlp_conf < 0.7` → **ne rien faire** (classifieur pas sûr, LLM tranche).
  3. Si accord LLM/NLP → renvoyer la réponse LLM (vrai label de raison).
  4. Si **désaccord** ET `nlp_conf >= 0.7` :
     - **Cas A** — NLP=maximale, LLM≠maximale → **NLP l'emporte** (sécurité patient). Loguer l'override.
     - **Cas B** — NLP=modérée/différée, LLM=maximale → **LLM l'emporte** (sécurité patient, on garde la montée). Loguer l'override.
     - En pratique : le NLP **peut surclasser vers le haut** (urgence manquée) mais ne peut pas **baisser** une urgence détectée par le LLM (faux positif dangereux).
- Métadonnée `triage_source` exposée dans la réponse : `"llm"`, `"nlp"`, `"llm_vetoed_by_nlp"`, `"nlp_vetoed_by_llm"`.

### 5. Tests
- `tests/test_nlp_triage.py` — vérifie chargement + prédiction sur 5 cas étalons.
- `tests/test_triage_veto.py` — cas d'accord, désaccord NLP vers le haut, désaccord NLP vers le bas.

## Vérification
- [ ] `scripts/label_triage_data.py` génère ≥300 exemples valides avec niveau + raison.
- [ ] Rapport de relecture échantillonnée sauvegardé et ≥ 80% de cohérence.
- [ ] `scripts/train_nlp_classifier.py` termine sans erreur, accuracy > 0.85 sur le split val.
- [ ] `app/nlp_triage.py` prédit correctement sur 5 cas étalons (charge le modèle fine-tuné).
- [ ] `app/main.py` : veto bidirectionnel actif et tracé dans les logs d'audit.