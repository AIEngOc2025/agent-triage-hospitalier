import logging
from typing import Any, Dict

from app.agent_tools import anonymize_clinical_data, classify_triage_urgency

logger = logging.getLogger(__name__)


class TriageAgentOrchestrator:
    """
    @definition : Gère le flux de triage agentique (états: anonymisation -> classification -> veto).
    @args/params : Aucun.
    @return : Aucun.
    """

    def __init__(self):
        self.state = "INIT"
        self.context = {}

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        @definition : Flux agentique totalement autonome. Toutes les décisions sont finalisées par l'agent.
        @args/params : user_input (str) - Entrée patient/infirmier.
        @return : Dict - Résultat avec 'status' (AUTO_FINALIZED).
        """
        logger.info("🚀 Flux agentique totalement autonome (MVP - Sans garde-fou).")
        self.context.setdefault("history", []).append(user_input)

        # 1. Dialogue si insuffisant
        if len(user_input) < 20:
            return {
                "status": "PENDING_CLARIFICATION",
                "question": "Pourriez-vous préciser vos symptômes et leur durée ?",
            }

        # 2. Triage autonome
        anonymized_text = anonymize_clinical_data(" ".join(self.context["history"]))
        nlp_result = classify_triage_urgency(anonymized_text)
        self.context["nlp_result"] = nlp_result

        # Autonomie totale : Pas de vérification humaine
        self.state = "AUTO_FINALIZED"
        return {
            "status": "AUTO_FINALIZED",
            "final_decision": nlp_result["niveau"],
            "comment": "Triage autonome finalisé par l'agent.",
            "reasoning": f"Classifieur NLP appliqué sur données anonymisées : Niveau {nlp_result['niveau']} (Confiance: {nlp_result['confiance']:.2%})",
        }

    def process_validation(
        self, validation: bool, comment: str, user_id: str
    ) -> Dict[str, Any]:
        """
        @definition : Consigne la validation humaine obligatoire pour les cas limites.
        @args/params :
            - validation (bool) - True si validé.
            - comment (str) - Justification.
            - user_id (str) - ID du validateur.
        @return : Dict - Décision finale consignée.
        """
        self.state = "COMPLETED"
        log_entry = {
            "action": "HUMAN_VALIDATION",
            "validated": validation,
            "comment": comment,
            "user_id": user_id,
            "context": self.context["nlp_result"],
        }
        logger.info(f"📝 Validation consignée : {log_entry}")

        return {
            "status": "FINALIZED",
            "final_decision": self.context["nlp_result"]["niveau"]
            if validation
            else "REVISED",
            "audit_log": log_entry,
        }
