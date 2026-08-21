"""Classifieur NLP de triage (guardrail déterministe).

Deux modes :
- **fine-tuné** (par défaut) : charge un modèle `distil-xlm-roberta-base`
  fine-tuné localement depuis `models/triage_nlp_model/`.
- **zero-shot fallback** : si le modèle fine-tuné est absent, utilise
  `facebook/bart-large-mnli` (de l'implémentation d'origine).

L'objectif est de fournir une prédiction rapide du niveau de triage
(maximale / modérée / différée) utilisable comme garde-fou dans
l'endpoint `/triage` (veto bidirectionnel).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

# --- Constantes ---
DEFAULT_FINE_TUNED_DIR = "models/triage_nlp_model"
FALLBACK_MODEL = "facebook/bart-large-mnli"
CONFIDENCE_MIN = 0.7  # seuil de fiabilité pour activer le veto

# Mapping fine-tuné (label -> libellé métier)
LABEL_ORDER = ["différée", "modérée", "maximale"]
Id2Label = {idx: label for idx, label in enumerate(LABEL_ORDER)}


class TriageClassifier:
    """Classifieur NLP de triage avec deux modes."""

    def __init__(self, model_dir: str = DEFAULT_FINE_TUNED_DIR):
        self.is_ready = False
        self.mode = "uninitialized"
        self.model_dir = Path(model_dir)
        self.classifier = None
        self._tokenizer = None
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Charge le modèle fine-tuné, sinon bascule en zero-shot."""
        if self.model_dir.exists() and (self.model_dir / "config.json").exists():
            self._load_fine_tuned()
        else:
            self._load_zero_shot()

    def _load_fine_tuned(self) -> None:
        """Charge le classifieur fine-tuné local via transformers."""
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )

            logger.info(
                "Loading fine-tuned NLP Triage Classifier from %s", self.model_dir
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir
            )
            self.classifier = pipeline(
                "text-classification",
                model=self._model,
                tokenizer=self._tokenizer,
            )
            self.mode = "fine_tuned"
            self.is_ready = True
            logger.info("✅ Fine-tuned NLP Triage Classifier ready.")
        except Exception as exc:
            logger.warning(
                "⚠️ Échec du chargement fine-tuné (%s) — fallback zero-shot", exc
            )
            self._load_zero_shot()

    def _load_zero_shot(self) -> None:
        """Fallback zero-shot si le modèle fine-tuné est indisponible."""
        try:
            from transformers import pipeline

            logger.info("Loading zero-shot fallback (bart-large-mnli)...")
            self.classifier = pipeline("zero-shot-classification", model=FALLBACK_MODEL)
            self.mode = "zero_shot"
            self.is_ready = True
            logger.info("✅ Zero-shot fallback ready.")
        except Exception as exc:
            logger.error("❌ Aucun classifieur disponible : %s", exc)
            self.is_ready = False

    def predict(self, text: str) -> Dict[str, object]:
        """Prédit le niveau de triage avec score de confiance.

        @return: {"niveau": str, "confiance": float, "mode": str, "actif": bool}
        """
        if not self.is_ready or not self.classifier:
            return {
                "niveau": "différée",
                "confiance": 0.0,
                "mode": self.mode,
                "actif": False,
            }

        try:
            if self.mode == "fine_tuned":
                result = self._predict_fine_tuned(text)
            else:
                result = self._predict_zero_shot(text)
            result["mode"] = self.mode
            result["actif"] = result["confiance"] >= CONFIDENCE_MIN
            return result
        except Exception as exc:
            logger.exception("Erreur de prédiction : %s", exc)
            return {
                "niveau": "différée",
                "confiance": 0.0,
                "mode": self.mode,
                "actif": False,
            }

    def _predict_fine_tuned(self, text: str) -> Dict[str, object]:
        """Prédiction avec le modèle fine-tuné."""
        result = self.classifier(text, truncation=True, max_length=256)
        top = result[0]
        raw_label = top["label"]
        # Le label peut être LABEL_0, LABEL_1, LABEL_2 ou directement le nom
        try:
            idx = int(raw_label.split("_")[-1])
            niveau = Id2Label.get(idx, raw_label)
        except (ValueError, IndexError):
            niveau = raw_label
        return {
            "niveau": niveau,
            "confiance": round(float(top["score"]), 4),
        }

    def _predict_zero_shot(self, text: str) -> Dict[str, object]:
        """Prédiction zero-shot."""
        labels = [
            "maximal vital emergency",
            "moderate medical consultation",
            "deferred medical advice",
        ]
        label_map = {
            "maximal vital emergency": "maximale",
            "moderate medical consultation": "modérée",
            "deferred medical advice": "différée",
        }
        result = self.classifier(text, labels)
        return {
            "niveau": label_map.get(result["labels"][0], "différée"),
            "confiance": round(float(result["scores"][0]), 4),
        }


# Instance globale (singleton)
triage_classifier = TriageClassifier()
