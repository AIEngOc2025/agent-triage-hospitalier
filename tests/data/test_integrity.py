import json
import os
import pytest

SFT_PATH = "data/processed/Mpaga_Christophe_1_Dataset_Train_SFT_052026.jsonl"
DPO_PATH = "data/processed/Mpaga_Christophe_1_Dataset_Train_DPO_052026.jsonl"


def load_jsonl(path):
    if not os.path.exists(path):
        pytest.skip(f"Dataset file {path} not found.")
    
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): # Skip empty lines
                data.append(json.loads(line))
    return data


# --- TESTS VOLUME ---
def test_sft_volume():
    """Vérifie que le dataset contient au moins 5000 exemples (ou vos paliers)"""
    data = load_jsonl(SFT_PATH)
    assert len(data) >= 4000  # Train (4000) + Val(500) + Test(500) = 5000 total


# --- TESTS BILINGUISME ---
def test_bilingual_parity():
    """Vérifie l'équilibre 50/50"""
    data = load_jsonl(SFT_PATH)
    fr_count = sum(1 for x in data if x["clinical_metadata"]["language"] == "fr")
    en_count = sum(1 for x in data if x["clinical_metadata"]["language"] == "en")

    # On autorise une marge d'erreur de 5%
    total = len(data)
    assert abs(fr_count - en_count) < (total * 0.05)
    print(f"\nRatio FR/EN : {fr_count}/{en_count}")


# --- TESTS ANONYMISATION ---
def test_anonymization_presence():
    """Vérifie que les tags d'anonymisation sont présents"""
    data = load_jsonl(SFT_PATH)
    anonymized = any("<PATIENT>" in str(x) for x in data)
    assert anonymized is True


# --- TESTS STRUCTURE DPO ---
def test_dpo_structure():
    """Vérifie le format spécifique Prompt/Chosen/Rejected"""
    data = load_jsonl(DPO_PATH)
    sample = data[0]
    assert "prompt" in sample
    assert "chosen" in sample
    assert "rejected" in sample
    assert "<|im_start|>" in sample["prompt"]
