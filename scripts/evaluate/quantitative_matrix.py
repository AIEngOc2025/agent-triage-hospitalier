import argparse
import json
import re

import pandas as pd
import torch
from langdetect import LangDetectException, detect
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION DEFAULTS ---
DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B-Base"
DEFAULT_ADAPTERS = "models/dpo_final_chsa"
DEFAULT_TEST_FILE = "data/golden_set.jsonl"


def calculate_matrix(model_id, adapters, test_file):
    """
    @definition : Calculates and prints the performance metrics matrix for the
                  model with improved robustness and device selection.
    @args/params :
        - model_id (str): ID of the base model.
        - adapters (str): Path to the PEFT adapters.
        - test_file (str): Path to the test dataset.
    @return : A dictionary containing performance metrics.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"🚀 Utilisation du périphérique : {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Use bfloat16 for stability on Apple Silicon (MPS)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    # Try 4-bit loading for memory efficiency
    try:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            adapters, quantization_config=quant_config, device_map={"": device}
        )
    except Exception:
        print("⚠️ 4-bit quantization failed, falling back to full precision.")
        model = AutoModelForCausalLM.from_pretrained(
            adapters, torch_dtype=dtype, device_map={"": device}
        )

    model.eval()

    with open(test_file, "r") as f:
        test_samples = [json.loads(line) for line in f][:2]

    results = []

    print(f"📊 Génération de la matrice quantitative sur {len(test_samples)} cas...")

    for item in tqdm(test_samples):
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

        # Robust extraction using Regex (case-insensitive and tolerant)
        match = re.search(
            r"\[niveau:\s*(maximale|modérée|différée)\s*\]", output_text, re.IGNORECASE
        )
        prediction_tag = match.group(1).lower() if match else None

        ground_truth = item["response"].lower()
        metadata = item.get("clinical_metadata", {})
        lang = metadata.get("language", "fr")

        # --- LOGIQUE DE LA MATRICE ---
        try:
            detected_lang = detect(output_text)
            lang_ok = 1 if detected_lang == lang else 0
        except LangDetectException:
            lang_ok = 0

        urgence_levels = {
            "maximale": ["maximale", "emergency", "immediate"],
            "modérée": ["modérée", "urgency"],
            "différée": ["différée", "deferred"],
        }
        expected_level = None
        for level, keywords in urgence_levels.items():
            if any(kw in ground_truth for kw in keywords):
                expected_level = level
                break

        urgence_match = 1 if expected_level and prediction_tag == expected_level else 0

        # Revised Hallucination detection: Focus on structural markers
        hallucination = (
            1
            if (
                "!!!" in output_text or "self." in output_text or prediction_tag is None
            )
            else 0
        )

        results.append(
            {
                "lang_ok": lang_ok,
                "urgence_match": urgence_match,
                "hallucination": hallucination,
            }
        )

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
    parser = argparse.ArgumentParser(description="Évaluation quantitative du modèle")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="ID du modèle")
    parser.add_argument(
        "--adapters", default=DEFAULT_ADAPTERS, help="Chemin des adaptateurs"
    )
    parser.add_argument(
        "--test_file", default=DEFAULT_TEST_FILE, help="Fichier de test"
    )
    args = parser.parse_args()

    calculate_matrix(args.model_id, args.adapters, args.test_file)
