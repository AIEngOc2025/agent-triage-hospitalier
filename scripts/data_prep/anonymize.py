from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


class MedicalAnonymizer:
    def __init__(self):
        # Configuration pour le français
        self.analyzer = AnalyzerEngine(default_score_threshold=0.4)
        self.anonymizer = AnonymizerEngine()

    def anonymize_text(self, text):
        if not isinstance(text, str):
            return text

        # 1. Analyse du texte (recherche de noms, téléphones, lieux)
        # Note: 'fr' doit être installé via spacy
        results = self.analyzer.analyze(text=text, entities=["PERSON", "LOCATION", "PHONE_NUMBER"], language='fr')

        # 2. Application de l'anonymisation (remplacement par <NOM>, etc.)
        operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "<PATIENT>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<ADRESSE>"}),
            "PHONE_NUMBER": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 10, "from_end": True}),
        }

        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators
        )
        return anonymized.text

# --- Test rapide ---
if __name__ == "__main__":
    cleaner = MedicalAnonymizer()
    test_msg = "Le patient Jean Dupont, habitant à Paris, a appelé au 0601020304."
    print(f"Original : {test_msg}")
    print(f"Anonymisé : {cleaner.anonymize_text(test_msg)}")
