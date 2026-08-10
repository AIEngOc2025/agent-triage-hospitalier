import json

import pandas as pd
import torch
from langdetect import LangDetectException, detect
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen3-1.7B-Base"
ADAPTERS = "models/dpo_final_chsa"
TEST_FILE = "data/golden_set.jsonl"


def calculate_matrix():
    """
    @definition : Calculates and prints the performance metrics matrix for the
                  model.
    @args/params : None.
    @return : A dictionary containing performance metrics (language precision,
              triage precision, and safety rate).
    """
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map={"": device}
    )
    model = PeftModel.from_pretrained(base_model, ADAPTERS)
    model.eval()

    with open(TEST_FILE, "r") as f:
        test_samples = [json.loads(line) for line in f][
            :50
        ]  # On teste 50 cas pour la matrice

    results = []

    print(f"📊 Génération de la matrice quantitative sur {len(test_samples)} cas...")

    for item in tqdm(test_samples):
        # Prompt renforcé pour la concision et la prise de décision
        prompt = (
            f"<|im_start|>system\nTu es l'infirmier du CHSA. "
            f"Répondez avec concision (max 50 tokens). "
            "Format strict : [Niveau: <maximale|modérée|différée>] - "
            "Orientation : <orientation>.\n"
            f"<|im_start|>user\n{item['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.01,
                do_sample=False,
                repetition_penalty=1.5,
                pad_token_id=tokenizer.pad_token_id,
            )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
        # Sécurisation de l'extraction du tag
        if "[niveau:" in output_text and "]" in output_text.split("[niveau:")[1]:
            prediction_tag = (
                output_text.split("[niveau:")[1].split("]")[0].strip().lower()
            )
        else:
            prediction_tag = None

        ground_truth = item["response"].lower()
        # Gestion sécurisée de la langue
        metadata = item.get("clinical_metadata", {})
        lang = metadata.get("language", "fr")

        # --- LOGIQUE DE LA MATRICE ---
        # 1. Vérification de la langue
        try:
            detected_lang = detect(output_text)
            lang_ok = 1 if detected_lang == lang else 0
        except LangDetectException:
            lang_ok = 0

        # 2. Vérification de l'urgence (Priorité)
        urgence_levels = {
            "maximale": ["maximale", "emergency", "immediate"],
            "modérée": ["modérée", "urgency"],
            "différée": ["différée", "deferred"],
        }
        expected_level = None
        for level in urgence_levels.keys():
            if level in ground_truth:
                expected_level = level
                break

        urgence_match = 1 if expected_level and prediction_tag == expected_level else 0

        # 3. Détection d'hallucination technique
        hallucination = (
            1
            if ("ui" in output_text or "!!!" in output_text or "self." in output_text)
            else 0
        )

        results.append(
            {
                "lang_ok": lang_ok,
                "urgence_match": urgence_match,
                "hallucination": hallucination,
            }
        )

        # DEBUG: Afficher quelques sorties
        if len(results) <= 3:
            print(f"\n--- DEBUG SAMPLE {len(results)} ---")
            print(f"PROMPT: {item['instruction'][:100]}...")
            print(f"TRUTH: {ground_truth[:100]}...")
            print(f"PRED: {output_text[:100]}...")

    # --- CALCUL DES SCORES FINAUX ---
    df = pd.DataFrame(results)
    matrix = {
        "Précision Linguistique": f"{(df['lang_ok'].mean() * 100):.2f}%",
        "Précision Triage (Mots-clés)": f"{(df['urgence_match'].mean() * 100):.2f}%",
        "Taux de Sécurité (Sans Hallucination)": (
            f"{((1 - df['hallucination'].mean()) * 100):.2f}%"
        ),
    }

    print("\n" + "═" * 45)
    print("📈 MATRICE DE PERFORMANCE QUANTITATIVE (SFT)")
    print("═" * 45)
    for k, v in matrix.items():
        print(f"{k:<35} : {v}")
    print("═" * 45)

    return matrix


if __name__ == "__main__":
    calculate_matrix()
