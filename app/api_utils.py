import asyncio
import json
import re
import uuid
from time import perf_counter, strftime
from typing import Dict

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from app.core.settings import settings

# --- ANONYMIZATION ---


class MedicalAnonymizer:
    def __init__(self):
        """
        @definition : Initialise l'anonymiseur médical avec Presidio
        pour le français et l'anglais.
        @args/params : Aucun.
        @return : Aucun.
        """
        # Configuration correcte pour charger plusieurs modèles SpaCy avec Presidio.
        # Utilisation des modèles larges pour une meilleure détection d'entités.
        provider_config = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "fr", "model_name": "fr_core_news_lg"},
                {"lang_code": "en", "model_name": "en_core_web_lg"},
            ],
        }

        # Explicitly configure NLP provider with language information
        provider = NlpEngineProvider(nlp_configuration=provider_config)
        nlp_engine = provider.create_engine()

        # Instantiate AnalyzerEngine with explicit nlp_engine
        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, default_score_threshold=0.4
        )
        self.anonymizer = AnonymizerEngine()
        self.last_latency = 0.0

    def anonymize_text(self, text: str, lang: str = "fr") -> str:
        """
        @definition : Anonymise les informations sensibles (noms, lieux,
        téléphones) dans un texte.
        @args/params :
            - text (str): Le texte à anonymiser.
            - lang (str): Le code langue ('fr' ou 'en') pour choisir le bon modèle.
        @return : Le texte anonymisé (str).
        """
        if not isinstance(text, str):
            return text

        start = perf_counter()

        # 1. Analyse du texte
        results = self.analyzer.analyze(
            text=text,
            entities=["PERSON", "LOCATION", "PHONE_NUMBER", "US_POSTAL_CODE"],
            language=lang,
        )

        # 2. Application de l'anonymisation
        operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "<PATIENT>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<ADRESSE>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<TELEPHONE>"}),
            "US_POSTAL_CODE": OperatorConfig("replace", {"new_value": "<CODE POSTAL>"}),
        }

        anonymized = self.anonymizer.anonymize(
            text=text, analyzer_results=results, operators=operators
        )

        self.last_latency = perf_counter() - start
        return anonymized.text


# Global instance for shared use
medical_anonymizer = MedicalAnonymizer()


def anonymize_text(text: str) -> str:
    """
    @definition: Anonymise les entités sensibles dans le texte en utilisant
    MedicalAnonymizer.
    @args/params: text (str)
    @return: str (texte anonymisé)
    """
    return medical_anonymizer.anonymize_text(text)


# --- UTILITIES ---


def clean_response(text: str) -> str:
    """
    @definition: Removes specific tags (like <think>, <tool_call>, <tool_response>) 
    and any other HTML-like tags from the model's output.
    @args/params:
        - text (str): The raw text from the model.
    @return: The cleaned text string.
    """
    patterns = [
        r"<think>.*?</think>",
        r"<tool_call>.*?</tool_call>",
        r"<tool_response>.*?</tool_response>",
        r"<[^>]+>",  # Balises génériques restantes
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    return text.strip()


async def log_audit(entry: dict) -> float:
    """
    @definition: Writes an audit log entry in JSONL format, anonymized.
    @args/params:
        - entry (dict): The log entry to record.
    @return: float (latency in seconds of the logging process).
    """
    start = perf_counter()
    try:
        # Anonymiser la décision ET l'input avant de loguer
        entry["decision"] = anonymize_text(entry["decision"])
        entry["input"] = anonymize_text(entry.get("input", ""))

        def write_log():
            with open(settings.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        await asyncio.to_thread(write_log)
    except Exception as e:
        print(f"❌ Audit logging failed: {e}")

    return perf_counter() - start


def create_log_entry(
    patient_id: str, user_input: str, decision: str, latency: float, stream: bool
) -> Dict:
    """
    @definition: Creates a standardized dictionary for an audit log entry.
    @args/params:
        - patient_id (str): The patient's identifier.
        - user_input (str): The user's query.
        - decision (str): The final model response.
        - latency (float): The request processing time in seconds.
        - stream (bool): Whether the request was streaming.
    @return: A dictionary representing the log entry.
    """
    return {
        "audit_id": str(uuid.uuid4()),
        "patient_id": anonymize_text(patient_id),
        "input": user_input,
        "decision": decision,
        "latency_sec": round(latency, 3),
        "timestamp": strftime("%Y-%m-%d %H:%M:%S"),
        "stream": stream,
    }
