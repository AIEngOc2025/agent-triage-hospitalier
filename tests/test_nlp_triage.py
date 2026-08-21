"""Tests unitaires du classifieur NLP de triage.

Couvre les deux modes (fine-tuné si dispo, zero-shot sinon) et la
robustesse aux cas dégradés (modèle absent, texte vide).
"""

from __future__ import annotations

import pytest

from app.nlp_triage import (
    CONFIDENCE_MIN,
    LABEL_ORDER,
    TriageClassifier,
)


class _StubClassifier:
    """Stub minimal mimant un pipeline de classification."""

    def __init__(self, label: str, score: float):
        self._label = label
        self._score = score

    def __call__(self, text, **kwargs):
        return [{"label": self._label, "score": self._score}]


def test_predict_returns_default_when_not_ready():
    """Si le modèle n'a pas pu être chargé, renvoie un payload dégradé."""
    classifier = TriageClassifier.__new__(TriageClassifier)
    classifier.is_ready = False
    classifier.mode = "uninitialized"
    classifier.classifier = None

    out = classifier.predict("douleur thoracique aiguë")

    assert out["niveau"] == "différée"
    assert out["confiance"] == 0.0
    assert out["actif"] is False


def test_predict_fine_tuned_label_decoding():
    """Déchiffre correctement les labels LABEL_0/LABEL_1/LABEL_2."""
    classifier = TriageClassifier.__new__(TriageClassifier)
    classifier.is_ready = True
    classifier.mode = "fine_tuned"
    # index 2 -> maximale
    classifier.classifier = _StubClassifier("LABEL_2", 0.91)

    out = classifier.predict("essoufflement sévère")

    assert out["niveau"] == "maximale"
    assert out["confiance"] == pytest.approx(0.91)
    assert out["actif"] is True  # 0.91 >= CONFIDENCE_MIN


def test_predict_handles_empty_text():
    """Texte vide : renvoie un payload sûr sans planter."""
    classifier = TriageClassifier.__new__(TriageClassifier)
    classifier.is_ready = True
    classifier.mode = "zero_shot"
    classifier.classifier = _StubClassifier("LABEL_0", 0.5)

    out = classifier.predict("")

    # Même avec un stub, le résultat doit être bien formé
    assert "niveau" in out
    assert "confiance" in out
    assert "mode" in out
    assert "actif" in out


def test_predict_exception_returns_safe_default():
    """Si le classifieur lève, on retombe sur la valeur sûre."""

    class _Boom:
        def __call__(self, text, **kwargs):
            raise RuntimeError("model exploded")

    classifier = TriageClassifier.__new__(TriageClassifier)
    classifier.is_ready = True
    classifier.mode = "fine_tuned"
    classifier.classifier = _Boom()

    out = classifier.predict("malaise")
    assert out["niveau"] == "différée"
    assert out["actif"] is False


def test_confidence_threshold_is_documented():
    """Garde-fou : le seuil de confiance est exposé et constant."""
    assert CONFIDENCE_MIN == 0.7
    assert set(LABEL_ORDER) == {"maximale", "modérée", "différée"}


def test_predict_zero_shot_label_mapping():
    """Mode zero-shot : mappe les labels anglais vers les libellés métier."""
    classifier = TriageClassifier.__new__(TriageClassifier)
    classifier.is_ready = True
    classifier.mode = "zero_shot"

    closed = {"labels": ["moderate medical consultation"], "scores": [0.8]}
    classifier.classifier = lambda text, labels: closed

    out = classifier.predict("douleur modérée depuis 3 jours")
    assert out["niveau"] == "modérée"
    assert out["confiance"] == pytest.approx(0.8)
