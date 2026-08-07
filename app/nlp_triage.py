import logging
from typing import Dict
from transformers import pipeline

logger = logging.getLogger(__name__)

class TriageClassifier:
    def __init__(self):
        """
        @definition: Initialise le modèle NLP Zero-Shot de triage.
        """
        self.is_ready = False
        self._load_model()

    def _load_model(self):
        """Charge le modèle Zero-Shot."""
        logger.info("Loading Zero-Shot NLP Triage Classifier...")
        # Modèle léger et performant pour le zero-shot
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli"
        )
        self.is_ready = True
        logger.info("NLP Triage Classifier ready.")

    def predict(self, text: str) -> Dict[str, str]:
        """
        @definition: Prédit le niveau de triage sans entraînement.
        @args: text (str)
        @return: Dict avec 'niveau' et 'confiance'
        """
        if not self.is_ready:
            return {"niveau": "différée", "confiance": 0.0}
        
        # Labels en anglais pour un meilleur alignement avec le modèle BART
        labels = ["maximal vital emergency", "moderate medical consultation", "deferred medical advice"]
        
        result = self.classifier(text, labels)
        
        # Mapping corrigé
        label_map = {
            "maximal vital emergency": "maximale",
            "moderate medical consultation": "modérée",
            "deferred medical advice": "différée"
        }
        
        top_label = result['labels'][0]
        confidence = result['scores'][0]
        
        return {
            "niveau": label_map.get(top_label, "différée"),
            "confiance": round(confidence, 2)
        }

# Instance globale
triage_classifier = TriageClassifier()
