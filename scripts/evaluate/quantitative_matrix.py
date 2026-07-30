import json

import pandas as pd
import torch
from langdetect import LangDetectException, detect
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen3-1.7B-Base"
ADAPTERS = "models/sft_final_chsa"
TEST_FILE = "data/processed/Mpaga_Christophe_1_Dataset_Test_SFT_052026.jsonl"


def calculate_matrix():
    """
    @definition : Calculates and prints the performance metrics matrix for the
                  model.
    @args/params : None.
    @return : A dictionary containing performance metrics (language precision,
              triage precision, and safety rate).
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
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
        prompt = (
            f"<|im_start|>system\nTu es l'infirmier du CHSA.<|im_end|>\n"
            f"<|im_start|>user\n{item['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=100, do_sample=False, repetition_penalty=1.2
            )

        output_text = (
            tokenizer.decode(outputs[0], skip_special_tokens=True)
            .split("assistant")[-1]
            .lower()
        )
        ground_truth = item["response"].lower()
        lang = item["clinical_metadata"]["language"]

        # --- LOGIQUE DE LA MATRICE ---
        # 1. Vérification de la langue
        try:
            detected_lang = detect(output_text)
            lang_ok = 1 if detected_lang == lang else 0
        except LangDetectException:
            lang_ok = 0  # Mark as incorrect if detection fails (e.g., text too short)

        # 2. Vérification de l'urgence (Priorité)
        # On cherche si les mots clés d'urgence (maximale/modérée) correspondent
        # On vérifie si le niveau d'urgence spécifique est correctement prédit.
        urgence_levels = {
            "maximale": ["maximale", "emergency", "immediate"],
            "modérée": ["modérée", "urgency"],
            "différée": ["différée", "deferred"],
        }

        # On détermine d'abord le niveau d'urgence attendu de la vérité terrain
        expected_level = None
        for level, keywords in urgence_levels.items():
            if any(kw in ground_truth for kw in keywords):
                expected_level = level
                break

        # Ensuite, on vérifie si la prédiction correspond à ce niveau attendu
        urgence_match = 0
        if expected_level and any(
            kw in output_text for kw in urgence_levels[expected_level]
        ):
            urgence_match = 1

        # 3. Détection d'hallucination technique (Code Swift/Points d'exclamation)
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
