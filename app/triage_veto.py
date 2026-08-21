"""Logique du veto bidirectionnel entre le LLM et le classifieur NLP.

Règle de sécurité (asymétrique) :
- Le NLP peut **promouvoir** vers "maximale" (rattrape une urgence manquée
  par le LLM) mais ne peut pas **rétrograder** une urgence maximale déjà
  détectée par le LLM (faux positif dangereux).
- Si le NLP est peu confiant (< 0.7), il ne s'exprime pas : le LLM tranche.
- En cas d'accord, la réponse LLM est conservée telle quelle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Ordre de criticité croissante : diff < mod < max
PRIORITY = {"différée": 0, "modérée": 1, "maximale": 2}
CONF_THRESHOLD = 0.7


@dataclass
class VetoOutcome:
    """Résultat du veto."""

    final_niveau: str
    orientation: str
    source: str  # "llm" | "nlp_vetoed_by_llm" | "llm_vetoed_by_nlp" | "abort"
    nlp_niveau: Optional[str]
    nlp_confiance: float
    llm_niveau: Optional[str]
    rationale: str


def decide_veto(
    llm_niveau: Optional[str],
    llm_orientation: str,
    nlp_niveau: Optional[str],
    nlp_confiance: float,
) -> VetoOutcome:
    """Applique la règle de veto bidirectionnel asymétrique.

    Args:
        llm_niveau: niveau détecté par le LLM (peut être None si pas classé).
        llm_orientation: texte d'orientation du LLM.
        nlp_niveau: niveau prédit par le NLP.
        nlp_confiance: confiance NLP [0, 1].
    """
    # --- Cas 1 : NLP pas confiant ou LLM pas encore classé ---
    if llm_niveau is None:
        return VetoOutcome(
            final_niveau="différée",
            orientation=llm_orientation,
            source="abort",
            nlp_niveau=nlp_niveau,
            nlp_confiance=nlp_confiance,
            llm_niveau=llm_niveau,
            rationale="LLM n'a pas classé le cas, sortie conversationnelle.",
        )

    if nlp_confiance < CONF_THRESHOLD or nlp_niveau is None:
        return VetoOutcome(
            final_niveau=llm_niveau,
            orientation=llm_orientation,
            source="llm",
            nlp_niveau=nlp_niveau,
            nlp_confiance=nlp_confiance,
            llm_niveau=llm_niveau,
            rationale="NLP pas assez confiant, on garde le LLM.",
        )

    # --- Cas 2 : Accord → LLM garde la main (raison + niveau) ---
    if nlp_niveau == llm_niveau:
        return VetoOutcome(
            final_niveau=llm_niveau,
            orientation=llm_orientation,
            source="llm",
            nlp_niveau=nlp_niveau,
            nlp_confiance=nlp_confiance,
            llm_niveau=llm_niveau,
            rationale="Accord LLM/NLP, sortie LLM conservée.",
        )

    # --- Cas 3 : Désaccord ---
    llm_pri = PRIORITY.get(llm_niveau, 0)
    nlp_pri = PRIORITY.get(nlp_niveau, 0)

    # NLP promeut (maximale côté NLP, plus bas côté LLM) → on monte
    if nlp_pri > llm_pri and nlp_niveau == "maximale":
        new_orientation = (
            f"{llm_orientation} [Override sécurité : NLP détecte une urgence "
            f"maximale (confiance={nlp_confiance:.2f}).]"
        )
        logger.warning(
            "🚨 VETO NLP→MAXIMALE : LLM=%s, NLP=%s, conf=%.2f",
            llm_niveau,
            nlp_niveau,
            nlp_confiance,
        )
        return VetoOutcome(
            final_niveau="maximale",
            orientation=new_orientation,
            source="llm_vetoed_by_nlp",
            nlp_niveau=nlp_niveau,
            nlp_confiance=nlp_confiance,
            llm_niveau=llm_niveau,
            rationale="NLP a promu le cas vers maximale (urgence manquée).",
        )

    # NLP dégrade mais LLM voit maximale → on garde maximale (sécurité)
    if llm_pri > nlp_pri and llm_niveau == "maximale":
        logger.warning(
            "🛡️  VETO LLM→MAXIMALE : LLM=%s, NLP=%s, conf=%.2f (NLP rétrogradé)",
            llm_niveau,
            nlp_niveau,
            nlp_confiance,
        )
        return VetoOutcome(
            final_niveau="maximale",
            orientation=llm_orientation,
            source="nlp_vetoed_by_llm",
            nlp_niveau=nlp_niveau,
            nlp_confiance=nlp_confiance,
            llm_niveau=llm_niveau,
            rationale="LLM garde maximale (veto défensif contre la rétrogradation NLP).",
        )

    # Désaccord mod-vs-diff : LLM l'emporte (modérée reste investiguée)
    return VetoOutcome(
        final_niveau=llm_niveau,
        orientation=llm_orientation,
        source="llm",
        nlp_niveau=nlp_niveau,
        nlp_confiance=nlp_confiance,
        llm_niveau=llm_niveau,
        rationale="Désaccord mod/diff, LLM l'emporte (raison plus riche).",
    )
