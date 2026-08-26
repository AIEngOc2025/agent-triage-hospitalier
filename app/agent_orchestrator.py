from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TriageState(Enum):
    """États possibles du graphe de triage."""

    START = auto()
    ANONYMIZATION = auto()
    NLP_CLASSIFICATION = auto()
    LLM_SYNTHESIS = auto()
    VETO_WAIT = auto()
    FINALIZATION = auto()


class TriageAgentOrchestrator:
    """Graphe d'états contrôlé pour le triage hospitalier."""

    def __init__(self):
        self.state = TriageState.START
        self.context: Dict[str, Any] = {}
        logger.info("Orchestrator initialized at START.")

    def transition_to(self, new_state: TriageState) -> None:
        """
        @definition : Effectue une transition vers un nouvel état du graphe.
        @args/params : new_state (TriageState) - État cible.
        @return : None
        """
        logger.info("Transitioning from %s to %s", self.state, new_state)
        self.state = new_state

    def process_step(self, data: Any = None) -> Any:
        """
        @definition : Traite l'étape actuelle en fonction de l'état du graphe.
        @args/params : data (Any) - Données d'entrée pour l'étape.
        @return : Any - Résultat de l'étape.
        """
        if self.state == TriageState.START:
            self.context["raw_data"] = data
            self.transition_to(TriageState.ANONYMIZATION)
            return "Anonymization next"

        if self.state == TriageState.ANONYMIZATION:
            # Placeholder pour appel à l'outil anonymisation
            self.context["anonymized_data"] = f"Anonymized: {self.context['raw_data']}"
            self.transition_to(TriageState.NLP_CLASSIFICATION)
            return "NLP Classification next"

        if self.state == TriageState.NLP_CLASSIFICATION:
            # Placeholder pour appel à l'outil classifieur
            self.context["triage_result"] = "Modérée"
            self.transition_to(TriageState.LLM_SYNTHESIS)
            return "LLM Synthesis next"

        if self.state == TriageState.LLM_SYNTHESIS:
            # Placeholder pour appel au LLM
            self.context["llm_synthesis"] = "Patient needs evaluation in 2h."
            self.transition_to(TriageState.FINALIZATION)
            return "Triage Complete"

        if self.state == TriageState.VETO_WAIT:
            # État d'attente désactivé
            self.transition_to(TriageState.FINALIZATION)
            return "Triage Complete"

        if self.state == TriageState.FINALIZATION:
            return "Triage Complete"

        return "Unknown State"

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        @definition : Exécute le graphe d'états jusqu'à un état d'attente ou final.
        @args/params : user_input (str) - Entrée utilisateur pour démarrer ou poursuivre.
        @return : Dict[str, Any] - Résultat de l'exécution agentique.
        """
        # Exécution simple du workflow jusqu'à VETO_WAIT ou FINALIZATION
        if self.state == TriageState.START:
            self.process_step(user_input)

        # Simulation d'exécution auto jusqu'à l'état VETO_WAIT
        while self.state not in [TriageState.VETO_WAIT, TriageState.FINALIZATION]:
            self.process_step()

        return {
            "final_decision": self.context.get("llm_synthesis"),
            "reasoning": "Exécution agentique terminée jusqu'au point de veto.",
            "state": self.state.name,
        }

    def handle_veto(self, approved: bool, comment: str) -> str:
        """
        @definition : Gère la décision du soignant après l'état de VETO_WAIT.
        @args/params : approved (bool) - True si validé, False si refusé.
        @args/params : comment (str) - Justification.
        @return : str - Résultat de la transition après veto.
        """
        if self.state != TriageState.VETO_WAIT:
            return "Not in Veto Wait state"

        self.context["veto_decision"] = {"approved": approved, "comment": comment}
        self.transition_to(TriageState.FINALIZATION)
        return "Finalization next"
