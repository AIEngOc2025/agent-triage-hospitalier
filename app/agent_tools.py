from app.api_utils import anonymize_text
from app.nlp_triage import triage_classifier


def anonymize_clinical_data(raw_text: str) -> str:
    """
    @definition : Nettoie et anonymise les données cliniques brutes du patient (RGPD).
    @args/params : raw_text (str) - Texte brut contenant potentiellement des PII.
    @return : str - Texte nettoyé des informations personnelles identifiables.
    """
    return anonymize_text(raw_text)


def classify_triage_urgency(text: str) -> dict:
    """
    @definition : Analyse le texte pour donner une première estimation du niveau d'urgence.
    @args/params : text (str) - Texte clinique anonymisé.
    @return : dict - Niveau d'urgence (maximale, modérée, différée) et score de confiance.
    """
    return triage_classifier.predict(text)


def is_vital_emergency_suspected(text: str) -> dict:
    """
    @definition : Détecte les mots-clés d'urgence vitale critique (drapeaux rouges).
    @args/params : text (str) - Texte clinique à analyser.
    @return : dict - Indication de suspicion d'urgence vitale critique et motif.
    """
    red_flag_keywords = [
        "douleur thoracique",
        "oppression",
        "bras gauche",
        "étouffement",
        "détresse respiratoire",
        "inconscient",
        "perte de connaissance",
        "paralysie",
        "avc",
        "hémorragie",
        "coma",
        "convulsion",
    ]
    lower_text = text.lower()
    detected = [kw for kw in red_flag_keywords if kw in lower_text]
    return {
        "vital_emergency": len(detected) > 0,
        "matched_keywords": detected,
    }


def clinical_veto_tool(veto_decision: bool, comment: str) -> dict:
    """
    @definition : Enregistre le veto ou l'approbation d'un soignant sur la décision de l'IA.
    @args/params : veto_decision (bool) - True si validé, False si refusé (veto).
    @args/params : comment (str) - Justification de la décision clinique.
    @return : dict - Statut de l'enregistrement du veto.
    """
    # Enregistrement du veto pour auditabilité clinique
    return {
        "status": "veto_recorded",
        "decision": veto_decision,
        "comment": comment,
    }
