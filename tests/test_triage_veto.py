"""Tests unitaires du veto bidirectionnel NLP ↔ LLM.

Couvre la matrice 5 cas définie dans `app/triage_veto.py` :
1. LLM n'a pas classé → source "abort"
2. NLP peu confiant → source "llm"
3. Accord → source "llm"
4. NLP promeut vers maximale → source "llm_vetoed_by_nlp"
5. NLP dégrade maximale → source "nlp_vetoed_by_llm"
"""

from __future__ import annotations

import pytest

from app.triage_veto import CONF_THRESHOLD, decide_veto

# --- 1. LLM n'a pas classé ---


def test_abort_when_llm_did_not_classify():
    out = decide_veto(
        llm_niveau=None,
        llm_orientation="Pouvez-vous préciser ?",
        nlp_niveau="maximale",
        nlp_confiance=0.95,
    )
    assert out.source == "abort"
    assert out.final_niveau == "différée"


# --- 2. NLP peu confiant ---


def test_llm_wins_when_nlp_low_confidence():
    out = decide_veto(
        llm_niveau="modérée",
        llm_orientation="Surveillez et consultez.",
        nlp_niveau="différée",
        nlp_confiance=CONF_THRESHOLD - 0.05,
    )
    assert out.source == "llm"
    assert out.final_niveau == "modérée"


def test_llm_wins_when_nlp_niveau_none():
    out = decide_veto(
        llm_niveau="modérée",
        llm_orientation="…",
        nlp_niveau=None,
        nlp_confiance=0.0,
    )
    assert out.source == "llm"


# --- 3. Accord ---


def test_agreement_keeps_llm():
    out = decide_veto(
        llm_niveau="maximale",
        llm_orientation="Appelez le 15.",
        nlp_niveau="maximale",
        nlp_confiance=0.9,
    )
    assert out.source == "llm"
    assert out.final_niveau == "maximale"
    assert out.orientation == "Appelez le 15."


# --- 4. NLP promeut vers "maximale" (urgence manquée) ---


def test_nlp_veto_promotes_to_maximale():
    out = decide_veto(
        llm_niveau="modérée",
        llm_orientation="Surveillez.",
        nlp_niveau="maximale",
        nlp_confiance=0.85,
    )
    assert out.source == "llm_vetoed_by_nlp"
    assert out.final_niveau == "maximale"
    assert "Override sécurité" in out.orientation


def test_nlp_veto_does_not_promote_to_modere():
    """Le NLP peut promouvoir seulement vers 'maximale', pas 'modérée'."""
    out = decide_veto(
        llm_niveau="différée",
        llm_orientation="Attendre.",
        nlp_niveau="modérée",
        nlp_confiance=0.85,
    )
    assert out.source == "llm"
    assert out.final_niveau == "différée"


# --- 5. NLP dégrade une maximale LLM (veto défensif) ---


def test_llm_keeps_maximale_when_nlp_demotes():
    out = decide_veto(
        llm_niveau="maximale",
        llm_orientation="Urgence vitale.",
        nlp_niveau="modérée",
        nlp_confiance=0.9,
    )
    assert out.source == "nlp_vetoed_by_llm"
    assert out.final_niveau == "maximale"
    assert out.llm_niveau == "maximale"


# --- 6. Désaccord mod vs diff ---


def test_llm_wins_on_mod_vs_diff():
    out = decide_veto(
        llm_niveau="modérée",
        llm_orientation="Consultation rapide.",
        nlp_niveau="différée",
        nlp_confiance=0.75,
    )
    assert out.source == "llm"
    assert out.final_niveau == "modérée"


# --- Aubaine : payload toujours bien formé ---


@pytest.mark.parametrize(
    "scenario",
    [
        {"llm_niveau": "maximale", "nlp_niveau": "maximale", "nlp_confiance": 0.9},
        {"llm_niveau": "modérée", "nlp_niveau": "maximale", "nlp_confiance": 0.85},
        {"llm_niveau": "maximale", "nlp_niveau": "modérée", "nlp_confiance": 0.9},
        {"llm_niveau": None, "nlp_niveau": "maximale", "nlp_confiance": 0.9},
        {"llm_niveau": "différée", "nlp_niveau": "différée", "nlp_confiance": 0.0},
    ],
)
def test_outcome_payload_shape(scenario):
    out = decide_veto(
        llm_niveau=scenario["llm_niveau"],
        llm_orientation="…",
        nlp_niveau=scenario["nlp_niveau"],
        nlp_confiance=scenario["nlp_confiance"],
    )
    assert out.final_niveau in {"différée", "modérée", "maximale"}
    assert out.source in {"llm", "llm_vetoed_by_nlp", "nlp_vetoed_by_llm", "abort"}
    assert out.rationale
