# Plan de Transition Agentique : Agent de Triage Hospitalier

## 1. Objectif
Faire évoluer le POC actuel (pipeline de classification de triage séquentiel) vers une architecture **agentique autonome contrôlée** (LLM + Orchestrateur + Outils + UI interactive). Cette architecture permet d'avoir un triage dynamique et adaptatif tout en garantissant une sécurité clinique absolue grâce à un veto humain systématique (Human-in-the-loop).

---

## 2. Architecture Cible

L'agent sera structuré autour de quatre piliers :

1.  **Le Moteur de Raisonnement (LLM) :**
    *   Utilisation de notre modèle fine-tuné ou fusionné DPO (ou modèle externe de repli).
    *   Fonctionnement en boucle de raisonnement (ex: ReAct ou Plan-and-Solve) avec génération de pensées intermédiaires (*Chain of Thought*) avant chaque action.

2.  **L'Orchestrateur (Graphe d'États) :**
    *   Implémentation d'un graphe d'états contrôlé (inspiré de **LangGraph** ou implémentation d'une machine à états légère et robuste en Python).
    *   Garantie d'un flux d'exécution strict : interdiction de finaliser une décision sans passer par les étapes d'anonymisation et de veto clinique.

3.  **Les Outils (Tools) :**
    *   `AnonymizerTool` : Nettoyage automatique des PII (RGPD) avant traitement par le LLM.
    *   `NlpClassifierTool` : Appel de notre classifieur léger local (`triage_nlp_model`) pour fournir une prédiction de base très rapide.
    *   `PatientHistoryTool` : Consultation (simulée ou réelle) des antécédents médicaux du patient pour enrichir le contexte.
    *   `ClinicalVetoTool` : Suspension de la boucle de l'agent pour solliciter la validation ou correction d'un médecin/infirmier (Human-in-the-loop).

4.  **L'Interface Utilisateur (Streamlit UI) :**
    *   Affichage transparent de la "pensée" de l'agent (étapes intermédiaires, outils invoqués).
    *   Widget interactif de veto permettant au soignant d'approuver ou modifier la décision avant enregistrement final.

---

## 3. Étapes d'Implémentation

### Étape 1 : Définition des Outils (`app/agent_tools.py`)
Encapsuler nos scripts et fonctionnalités existantes sous forme d'outils réutilisables par l'agent. Chaque fonction doit respecter scrupuleusement nos normes de documentation (docstrings spécifiques).

*Exemple de structure d'outil conforme :*
```python
def anonymize_clinical_data(raw_text: str) -> str:
    """
    @definition : Nettoie et anonymise les données cliniques brutes du patient (RGPD).
    @args/params : raw_text (str) - Texte brut contenant potentiellement des PII.
    @return : str - Texte nettoyé des informations personnelles identifiables.
    """
    # Utilisation de nos expressions régulières ou modèle d'anonymisation existant
    pass
```

### Étape 2 : Conception de l'Orchestrateur (`app/agent_orchestrator.py`)
Développer la machine à états de l'agent.
*   **Structure du graphe :**
    `START ➔ [Anonymisation] ➔ [Analyse Initiale LLM] ➔ [Appel Outils (Classifieur/Historique)] ➔ [Synthèse Finale LLM] ➔ [Attente Veto Humain] ➔ [Enregistrement/Fin]`
*   **Sauvegarde d'état (State Checkpointing) :** Gérer l'arrêt de la boucle de l'agent au niveau de l'étape de veto pour attendre la saisie utilisateur, puis reprendre l'exécution avec les nouvelles données saisies par le soignant.

### Étape 3 : Intégration dans Streamlit UI (`ui_streamlit.py`)
*   **Visualisation :** Ajouter un expander (`st.expander("🧠 Chaîne de pensée de l'agent")`) affichant le journal d'exécution (logs de l'agent, étapes de raisonnement, appels d'outils).
*   **Interaction :** Implémenter le widget de blocage interactif pour le veto clinique de l'infirmier(e).

### Étape 4 : Tests d'Intégration (`tests/test_agent_triage.py`)
Écrire des tests robustes pour valider le comportement de l'agent :
*   Vérifier que le graphe d'états s'exécute dans l'ordre obligatoire.
*   Valider que l'agent s'arrête correctement en attente du veto humain.
*   Tester la résilience de l'agent si un outil (ex: l'historique du patient) échoue ou retourne une erreur.

---

## 4. Livrables et Indicateurs de Succès

### Livrables
- 📝 `plans/transition_agentique.md` (Ce document).
- 🐍 `app/agent_tools.py` (Bibliothèque des outils de l'agent).
- 🐍 `app/agent_orchestrator.py` (Moteur et graphe d'états).
- 🎨 `ui_streamlit.py` mis à jour avec le panneau de pensée de l'agent et la gestion des sessions d'attente de veto.
- 🧪 `tests/test_agent_triage.py` pour valider toute la chaîne de bout en bout.

### Indicateurs de Succès (KPIs)
*   **Conformité du flux de sécurité :** 100% des cas traités doivent obligatoirement passer par l'outil d'anonymisation et l'outil de veto humain avant enregistrement.
*   **Transparence :** Les étapes intermédiaires de raisonnement sont lisibles et compréhensibles par l'utilisateur final en moins de 1 seconde d'affichage.
*   **Résilience :** En cas de panne d'un outil secondaire (historique), l'agent doit être capable de dégrader son mode de fonctionnement et de proposer un triage basé sur les outils restants, sans bloquer le processus global.
