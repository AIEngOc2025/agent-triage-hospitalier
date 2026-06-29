import json

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-0.5B"
ADAPTERS = "models/sft_final_chsa"
TEST_FILE = "data/processed/Mpaga_Christophe_1_Dataset_Test_SFT_052026.jsonl"

def calculate_matrix():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map={"": device})
    model = PeftModel.from_pretrained(base_model, ADAPTERS)
    model.eval()

    with open(TEST_FILE, 'r') as f:
        test_samples = [json.loads(line) for line in f][:50] # On teste 50 cas pour la matrice

    results = []

    print(f"📊 Génération de la matrice quantitative sur {len(test_samples)} cas...")

    for item in tqdm(test_samples):
        prompt = f"<|im_start|>system\nTu es l'infirmier du CHSA.<|im_end|>\n<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, repetition_penalty=1.2)

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].lower()
        ground_truth = item['response'].lower()
        lang = item['clinical_metadata']['language']

        # --- LOGIQUE DE LA MATRICE ---
        # 1. Vérification de la langue
        lang_ok = 1 if (lang == 'fr' and any(w in output_text for w in ['le', 'la', 'est'])) or \
                       (lang == 'en' and any(w in output_text for w in ['the', 'is', 'with'])) else 0

        # 2. Vérification de l'urgence (Priorité)
        # On cherche si les mots clés d'urgence (maximale/modérée) correspondent
        urgence_keywords = ['maximale', 'modérée', 'différée', 'urgency', 'emergency', 'immediate']
        urgence_match = 1 if any(w in output_text for w in urgence_keywords if w in ground_truth) else 0

        # 3. Détection d'hallucination technique (Code Swift/Points d'exclamation)
        hallucination = 1 if ("ui" in output_text or "!!!" in output_text or "self." in output_text) else 0

        results.append({
            "lang_ok": lang_ok,
            "urgence_match": urgence_match,
            "hallucination": hallucination
        })

    # --- CALCUL DES SCORES FINAUX ---
    df = pd.DataFrame(results)
    matrix = {
        "Précision Linguistique": f"{(df['lang_ok'].mean() * 100):.2f}%",
        "Précision Triage (Mots-clés)": f"{(df['urgence_match'].mean() * 100):.2f}%",
        "Taux de Sécurité (Sans Hallucination)": f"{((1 - df['hallucination'].mean()) * 100):.2f}%"
    }

    print("\n" + "═"*45)
    print("📈 MATRICE DE PERFORMANCE QUANTITATIVE (SFT)")
    print("═"*45)
    for k, v in matrix.items():
        print(f"{k:<35} : {v}")
    print("═"*45)

    return matrix

if __name__ == "__main__":
    calculate_matrix()
