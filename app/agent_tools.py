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
