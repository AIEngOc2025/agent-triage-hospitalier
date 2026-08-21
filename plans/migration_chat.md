# Migration du chat vers vLLM in-process (mode hybride)

## Contexte

Comparaison de `app/main.py` (gateway FastAPI → `RemoteInferenceClient` via HTTP vers
un serveur vLLM séparé) et de `app/template_main.py` (script CLI vLLM in‑process).
Le template n'est pas importable tel quel, mais il montre la mécanique `LLM()` /
`EngineArgs` / `get_default_sampling_params()` qui ouvre la voie à un chat plus
fluide.

Objectif : fluidifier `/chat` (latence P99, premier token, streaming natif) sans
casser le mode distant utilisé en production. Approche retenue : **mode hybride
piloté par l'environnement**, `ENGINE_MODE=local|remote`. Le mode `local` charge
vLLM dans le process de l'API (utile en dev/CI), le mode `remote` conserve
l'architecture actuelle (cible prod).

Périmètre du présent document : **plan seul**. Aucune modification de code n'est
effectuée dans cette livraison ; le plan décrit les étapes à exécuter
ultérieurement.

## Comparaison synthétique des deux fichiers

| Aspect | `app/main.py` | `app/template_main.py` |
|---|---|---|
| Inférence | HTTP distant (httpx → vLLM externe) | `LLM(...)` in-process |
| Streaming | SSE reconstruit sur chunks HTTP | Natif via `RequestOutput` itéré |
| Latence 1er token | +1 round-trip + sérialisation JSON | Appel direct C++/CUDA |
| Coût démarrage | Zéro (vLLM ailleurs) | 5–15 s cold start (déjà couvert par `WARMUP_TIMEOUT_SEC`) |
| Mémoire pod | ~200 Mo (API stateless) | Plusieurs Go VRAM dans le pod |
| Scale | Horizontal (N API → 1 vLLM) | Vertical (1 modèle/replica) |
| Sampling params | Passés via payload HTTP | Contrôlés via `sampling_params` Python |

Conclusion : le template est un **squelette de migration**, pas un composant
réutilisable tel quel.

## Architecture cible (mode hybride)

```
                ┌───────────────────────────────────────┐
                │       FastAPI (app/main.py)           │
                │                                       │
ENGINE_MODE=    │   �─────────────────┐                 │
  remote ─────► │   │ RemoteInference │ → httpx → vLLM  │  (prod)
                │   │ Client          │   distant        │
                │   └─────────────────┘                  │
                │                                       │
ENGINE_MODE=    │   ┌─────────────────┐                 │
  local  ─────► │   │ vllm.LLM(...)   │ in-process      │  (dev/CI)
                │   │ + SamplingParams │                 │
                │   └─────────────────┘                  │
                └───────────────────────────────────────┘
```

Le `ModelEngine` existant (lignes 29‑61 de `app/main.py`) devient une fabrique
qui choisit l'implémentation à `initialize()` selon `settings.engine_mode`.

## Étapes d'implémentation

### 1. Configuration
- Ajouter `engine_mode: Literal["remote", "local"] = "remote"` dans
  `app/core/settings.py` (source unique de vérité, déjà lu par `lifespan`).
- Ajouter `model_name`, `max_model_len`, `dtype`, `gpu_memory_utilization`,
  `enforce_eager`, `enable_prefix_caching` dans `settings`, avec valeurs par
  défaut alignées sur le modèle de prod.

### 2. Abstraction `ModelEngine` (réutilisation)
Le contrat existant (`generate`, `generate_stream`, `generate_structured`) est
conservé. On ajoute deux implémentations derrière une factory :

- `RemoteEngine` (code actuel, déplacé dans `app/remote/engine.py`) — aucun
  changement de comportement.
- `LocalEngine` (nouveau, `app/local/engine.py`) — s'inspire de
  `app/template_main.py:35‑46` :
  - `LLM(**engine_args)`
  - `self._sampling = llm.get_default_sampling_params()`
  - `generate_stream` : boucle sur `llm.chat(conversation, sampling_params,
    use_tqdm=False)` ou, mieux, `llm.chat(..., stream=True)` puis itération
    `RequestOutput.outputs[0].text` avec yield incrémental.
  - `generate` : concatène les chunks.
  - `generate_structured` : brancher `instructor.from_vllm(llm)` pour préserver
    le contrat `TriageResponse` sans changer l'appelant.

Le `ModelEngine.initialize()` choisit la classe en fonction de
`settings.engine_mode`.

### 3. Streaming natif
- `LocalEngine.generate_stream` yield directement les deltas texte (pas
  d'enveloppe `data: ` côté Python, FastAPI `StreamingResponse` les ajoute).
- Conserver le format SSE inchangé côté HTTP pour ne pas casser les clients.

### 4. Warmup
- Garder la logique `lifespan` (lignes 67‑93) ; elle fonctionne pour les deux
  modes. En local, le warmup déclenche le cold start une seule fois au boot.

### 5. Audit et veto
- `create_log_entry` / `log_audit` (`app/api_utils.py`) et `decide_veto`
  (`app/triage_veto.py`) ne dépendent pas du moteur — aucune modification.
- `_extract_user_input` et `_ensure_system_prompt` (`app/main.py:123‑135`)
  restent valides.

### 6. Dockerfile (référence : `Dockerfile.backend`, déjà modifié)
- Ajouter une cible `dev-local` ou une variable `BUILD_VLLM=1` qui installe
  `vllm` + CUDA runtime, pour activer `ENGINE_MODE=local`.
- La cible prod reste identique (pas de vLLM dans l'image API).

### 7. Tests
- `tests/test_integration_gateway_inference.py` et `tests/test_replication_500.py`
  couvrent le chemin distant — les garder tels quels.
- Ajouter `tests/test_local_engine.py` : mock `vllm.LLM` (réutiliser le pattern
  de mock déjà présent dans `tests/test_nlp_triage.py` / `tests/test_triage_veto.py`)
  et vérifier que `LocalEngine` produit les mêmes outputs que `RemoteEngine`
  sur un set de prompts fixes.
- Ajouter `tests/test_engine_mode_switch.py` : `ENGINE_MODE=remote` ne charge
  jamais `vllm` (import différé / lazy import dans `LocalEngine` pour éviter
  la dépendance CUDA en CI CPU).

## Fichiers critiques

- `app/main.py` — factory `ModelEngine.initialize()`.
- `app/core/settings.py` — nouveau champ `engine_mode`.
- `app/remote/engine.py` (**nouveau**) — extraction de la logique distante.
- `app/local/engine.py` (**nouveau**) — portage du template in-process.
- `app/template_main.py` — référence, à supprimer une fois `LocalEngine` validé
  (ou à conserver comme exemple CLI).
- `Dockerfile.backend` — cible `dev-local` conditionnelle.
- `tests/test_local_engine.py`, `tests/test_engine_mode_switch.py`
  (**nouveaux**).
- `app/api_utils.py`, `app/triage_veto.py`, `app/nlp_triage.py` — réutilisés
  sans modification.

## Réutilisations identifiées (pas de nouveau code si possible)

- `call_with_retry` (`app/remote/client.py`) — réutilisé tel quel en mode
  distant ; non applicable en local (le `LLM` in-process n'a pas de retry HTTP
  à ajouter, lever des exceptions vLLM natives).
- `create_log_entry` / `log_audit` (`app/api_utils.py`) — réutilisés.
- `SYSTEM_PROMPT_FR` (`app/system_prompts.py`) — réutilisé via
  `_ensure_system_prompt`.
- `TriageResponse` (`app/schemas.py`) — réutilisé via `instructor.from_vllm`.

## Critères de "chat plus fluide"

Mesures à collecter avant/après, en mode `local` sur un GPU unique :

| Métrique | Mode `remote` (référence) | Mode `local` (cible) |
|---|---|---|
| TTFT P50 / P95 (ms) | à mesurer | -30 % à -60 % attendu |
| Latence E2E chat 256 tok (P95) | à mesurer | -15 % à -40 % attendu |
| Tokens/s en streaming (P50) | à mesurer | stable ou +10 % |
| Cold start pod | ~0 s | 5–15 s (1× au boot) |
| Mémoire pod (RSS + VRAM) | ~200 Mo | ~VRAM modèle + ~1 Go |

Si les gains TTFT < 20 % en P95, ne pas activer `local` en prod : rester sur
`remote` + prefix caching + HTTP/2.

## Vérification end‑to‑end

1. `docker compose up vllm` (serveur distant de référence inchangé).
2. `ENGINE_MODE=remote uvicorn app.main:app` → smoke test :
   - `curl localhost:8080/health` → `{"status":"ok","engine":"RemoteInference"}`
   - `curl -X POST localhost:8080/chat -d '{...}'` → réponse texte + `audit_ref`.
   - `curl -X POST localhost:8080/triage -d '{...}'` → `triage_result` + `nlp_veto_meta`.
3. `ENGINE_MODE=local uvicorn app.main:app --reload` (machine avec GPU) :
   - `curl localhost:8080/health` → `{"status":"ok","engine":"LocalVLLM"}`
   - Mêmes endpoints, comparaison A/B sur 50 prompts.
4. `pytest tests/test_local_engine.py tests/test_engine_mode_switch.py -v`.
5. Benchmark : `python scripts/benchmark/run_benchmark.py --engine local` (ou
   créer `scripts/benchmark/run_benchmark.py` à partir de
   `scripts/benchmark/data/results_cloud.jsonl`).

## Risques et mitigations

- **Import de `vllm` en CI CPU** : lazy import dans `app/local/engine.py`,
  chargé uniquement quand `settings.engine_mode == "local"`.
- **Divergence local/remote** : tests d'égalité structurelle sur un corpus
  figé (50 prompts, golden responses). Toute divergence = bug d'un des deux
  moteurs.
- **Régression du warmup** : en local, un échec vLLM au boot tue l'API ;
  conserver le `try/except` autour du warmup et basculer sur une 503 propre
  si l'init échoue.
- **Supervision** : exposer `engine.engine_type` dans `/health` (déjà le cas)
  et ajouter `engine_model` (nom du modèle chargé) pour le runbook.

## Hors périmètre (à traiter séparément)

- Migration du `/triage` vers in-process (même mécanique, mais schema JSON
  Pydantic → instructor).
- Optimisations HTTP/2 / keep-alive du chemin distant.
- Prefix caching côté serveur vLLM.
- Multi-GPU / tensor parallel dans `LocalEngine`.
