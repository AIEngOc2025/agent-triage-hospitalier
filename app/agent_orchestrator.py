from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any, AsyncGenerator, Dict, List

from app.agent_tools import (
    anonymize_clinical_data,
    classify_triage_urgency,
    is_vital_emergency_suspected,
)
from app.engine_factory import engine
from app.system_prompts import SYSTEM_PROMPT_AGENT_CONVERSATIONAL_FR
from app.triage_veto import decide_veto

logger = logging.getLogger(__name__)


class TriageState(Enum):
    """États possibles du graphe de triage agentique."""

    START = auto()
    ANONYMIZATION = auto()
    NLP_CLASSIFICATION = auto()
    LLM_SYNTHESIS = auto()
    VETO_WAIT = auto()
    FINALIZATION = auto()


class TriageAgentOrchestrator:
    """
    Graphe d'états contrôlé pour le triage hospitalier conversationnel.
    Gère le dialogue adaptatif, l'anonymisation RGPD et la sécurité clinique.
    """

    def __init__(self):
        self.state = TriageState.START
        self.context: Dict[str, Any] = {
            "raw_data": "",
            "anonymized_data": "",
            "nlp_prediction": {},
            "is_emergency": False,
            "llm_synthesis": "",
            "triage_result": None,
            "history": [],
        }
        logger.info("Orchestrateur de triage initialisé à l'état START.")

    def transition_to(self, new_state: TriageState) -> None:
        """
        @definition : Effectue une transition vers un nouvel état du graphe.
        @args/params : new_state (TriageState) - État cible.
        @return : None
        """
        logger.info("Transition d'état : %s -> %s", self.state.name, new_state.name)
        self.state = new_state

    async def process_step(self, data: Any = None) -> Any:
        """
        @definition : Traite l'étape actuelle du workflow agentique.
        @args/params : data (Any) - Données d'entrée (texte utilisateur).
        @return : Any - Résultat ou description de l'étape courante.
        """
        # Étape 1 : Réception des données brutes
        if self.state == TriageState.START:
            if data is not None:
                self.context["raw_data"] = str(data)
            self.transition_to(TriageState.ANONYMIZATION)
            return "Anonymization step"

        # Étape 2 : Anonymisation des PII (Presidio RGPD)
        if self.state == TriageState.ANONYMIZATION:
            raw_text = self.context.get("raw_data", "")
            anonymized = anonymize_clinical_data(raw_text)
            self.context["anonymized_data"] = anonymized
            self.transition_to(TriageState.NLP_CLASSIFICATION)
            return "NLP classification step"

        # Étape 3 : Classification discriminative NLP & Red Flags
        if self.state == TriageState.NLP_CLASSIFICATION:
            text_to_analyze = self.context.get("anonymized_data", "")
            nlp_pred = classify_triage_urgency(text_to_analyze)
            red_flag_check = is_vital_emergency_suspected(text_to_analyze)

            self.context["nlp_prediction"] = nlp_pred
            self.context["is_emergency"] = red_flag_check.get(
                "vital_emergency", False
            ) or (
                nlp_pred.get("niveau") == "maximale"
                and float(nlp_pred.get("confiance", 0.0)) >= 0.7
            )
            self.context["triage_result"] = nlp_pred.get("niveau", "modérée")
            self.transition_to(TriageState.LLM_SYNTHESIS)
            return "LLM synthesis step"

        # Étape 4 : Synthèse conversationnelle et clinique par le LLM
        if self.state == TriageState.LLM_SYNTHESIS:
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT_AGENT_CONVERSATIONAL_FR}
            ]
            # Ajout du contexte conversationnel s'il existe
            for msg in self.context.get("history", []):
                messages.append(
                    {
                        "role": msg.get("role", "user"),
                        "content": anonymize_clinical_data(msg.get("content", "")),
                    }
                )

            # Ajout du message actuel si non présent dans l'historique
            current_user_msg = self.context.get("anonymized_data", "")
            if not any(m.get("content") == current_user_msg for m in messages):
                messages.append({"role": "user", "content": current_user_msg})

            try:
                synthesis_text = await engine.generate(messages)
            except Exception as exc:
                logger.warning("Échec génération LLM direct : %s", exc)
                synthesis_text = (
                    "Vos symptômes ont été enregistrés. Un soignant va vous évaluer."
                )

            self.context["llm_synthesis"] = synthesis_text

            # Veto bidirectionnel de sécurité
            nlp_pred = self.context.get("nlp_prediction", {})
            veto = decide_veto(
                llm_niveau=self.context.get("triage_result"),
                llm_orientation=synthesis_text,
                nlp_niveau=nlp_pred.get("niveau"),
                nlp_confiance=float(nlp_pred.get("confiance", 0.0)),
            )
            self.context["veto_outcome"] = veto
            self.context["final_niveau"] = veto.final_niveau

            self.transition_to(TriageState.FINALIZATION)
            return "Triage completed"

        # Étape 5 : Attente de validation soignant (Human-in-the-loop)
        if self.state == TriageState.VETO_WAIT:
            self.transition_to(TriageState.FINALIZATION)
            return "Triage finalized"

        # Étape 6 : État terminal
        if self.state == TriageState.FINALIZATION:
            return "Triage complete"

        return "Unknown State"

    async def run(
        self, user_input: str, history: List[Dict[str, str]] | None = None
    ) -> Dict[str, Any]:
        """
        @definition : Exécute le graphe d'états jusqu'à un état d'attente ou final.
        @args/params : user_input (str) - Entrée patient / soignant.
        @args/params : history (List[Dict[str, str]] | None) - Historique conversationnel.
        @return : Dict[str, Any] - Résultat complet de l'évaluation agentique.
        """
        if history:
            self.context["history"] = history

        # Démarrage du cycle si l'état est initial
        if self.state == TriageState.START:
            await self.process_step(user_input)

        # Progression automatique dans les étapes de traitement
        while self.state not in [TriageState.VETO_WAIT, TriageState.FINALIZATION]:
            await self.process_step()

        # Raisonnement clinique explicatif
        nlp_pred = self.context.get("nlp_prediction", {})
        emergency = self.context.get("is_emergency", False)
        if emergency:
            reasoning = (
                "🚨 Signal d'urgence vitale détecté : orientation immédiate en priorité "
                "maximale (déchocage / box d'urgence vitale)."
            )
        elif self.state == TriageState.FINALIZATION:
            reasoning = (
                f"Évaluation clinique effectuée (NLP: {nlp_pred.get('niveau', 'N/A')}, "
                f"Confiance: {nlp_pred.get('confiance', 0.0):.2f})."
            )
        else:
            reasoning = "Dialogue de recueil des symptômes en cours."

        return {
            "final_decision": self.context.get("llm_synthesis"),
            "reasoning": reasoning,
            "state": self.state.name,
            "triage_level": self.context.get("final_niveau", "modérée"),
            "is_emergency": emergency,
        }

    async def run_stream(
        self, user_input: str, history: List[Dict[str, str]] | None = None
    ) -> AsyncGenerator[str, None]:
        """
        @definition : Flux conversationnel agentique en mode streaming.
        @args/params : user_input (str) - Entrée patient / soignant.
        @args/params : history (List[Dict[str, str]] | None) - Historique du chat.
        @return : AsyncGenerator[str, None] - Flux de texte streamé.
        """
        logger.info("🚀 Démarrage du flux de triage en streaming.")

        # 1. Anonymisation RGPD de l'entrée
        anonymized_input = anonymize_clinical_data(user_input)
        self.context["anonymized_data"] = anonymized_input

        # 2. Détection rapide des urgences vitales (red flags)
        nlp_pred = classify_triage_urgency(anonymized_input)
        red_flag_check = is_vital_emergency_suspected(anonymized_input)
        self.context["nlp_prediction"] = nlp_pred

        if red_flag_check.get("vital_emergency") or (
            nlp_pred.get("niveau") == "maximale"
            and float(nlp_pred.get("confiance", 0.0)) >= 0.8
        ):
            yield (
                "🚨 **URGENCE VITALE DÉTECTÉE (PRIORITÉ MAXIMALE)** 🚨\n\n"
                "Vos symptômes nécessitent une prise en charge médicale immédiate. "
                "Veuillez vous adresser sans délai à l'équipe infirmière d'accueil "
                "pour une installation en box de déchocage."
            )
            self.transition_to(TriageState.FINALIZATION)
            return

        # 3. Construction des messages pour la synthèse LLM
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT_AGENT_CONVERSATIONAL_FR}
        ]
        if history:
            for msg in history:
                messages.append(
                    {
                        "role": msg.get("role", "user"),
                        "content": anonymize_clinical_data(msg.get("content", "")),
                    }
                )
        if not history or not any(
            m.get("content") == anonymized_input for m in history
        ):
            messages.append({"role": "user", "content": anonymized_input})

        # 4. Streaming de la réponse
        try:
            async for chunk in engine.generate_stream(
                messages, request_id="triage-stream"
            ):
                yield chunk
        except Exception as exc:
            logger.error("Erreur lors du streaming agentique : %s", exc)
            yield "Une erreur est survenue lors de l'analyse. Un soignant va vous assister."

        self.transition_to(TriageState.FINALIZATION)
        logger.info("✅ Flux de triage agentique terminé.")

    def handle_veto(self, approved: bool, comment: str) -> str:
        """
        @definition : Gère la décision du soignant après l'état de VETO_WAIT.
        @args/params : approved (bool) - True si validé, False si refusé.
        @args/params : comment (str) - Justification clinique.
        @return : str - Résultat de la transition après veto.
        """
        if self.state != TriageState.VETO_WAIT:
            return "Not in Veto Wait state"

        self.context["veto_decision"] = {"approved": approved, "comment": comment}
        self.transition_to(TriageState.FINALIZATION)
        return "Finalization next"
