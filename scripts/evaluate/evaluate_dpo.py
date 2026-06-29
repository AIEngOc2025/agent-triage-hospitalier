import gc
import json
import os

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-0.5B"
DPO_ADAPTERS = "models/dpo_final_chsa"
# On utilise le fichier TEST DPO que nous avons créé sur Mac
TEST_FILE = "data/processed/Mpaga_Christophe_1_Dataset_Test_DPO_052026.jsonl"

def evaluate_dpo_safe():
    print("🧹 Nettoyage de la mémoire...")
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # 1. Chargement du Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Chargement du modèle de base (FP16 pour Mac M1)
    print("📥 Chargement du modèle de base...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map={"": "mps"}
    )

    # 3. Chargement des adaptateurs DPO
    print("🧠 Application des adaptateurs DPO finaux...")
    model = PeftModel.from_pretrained(base_model, DPO_ADAPTERS)
    model.eval()

    # 4. Chargement des données TEST DPO
    if not os.path.exists(TEST_FILE):
        print(f"❌ Erreur : Le fichier {TEST_FILE} est introuvable.")
        return

    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    total_loss = 0
    count = 0

    print(f"🔬 Évaluation sur {len(dataset)} exemples DPO...")

    with torch.no_grad():
        for item in tqdm(dataset):
            try:
                # --- CORRECTION DES CLÉS ICI ---
                # Dans le DPO, on utilise 'prompt' et 'chosen' (la réponse de référence)
                prompt_text = item['prompt']
                chosen_answer = item['chosen']

                # On construit le texte complet tel que le modèle doit le voir
                # Note : 'prompt' contient déjà les balises <|im_start|> d'après nos scripts précédents
                full_text = f"{prompt_text}{chosen_answer}<|im_end|>"

                inputs = tokenizer(full_text, return_tensors="pt").to("mps")
                labels = inputs["input_ids"].clone()

                outputs = model(**inputs, labels=labels)
                total_loss += outputs.loss.item()
                count += 1

                # Nettoyage mémoire vive toutes les 10 itérations
                if count % 10 == 0:
                    gc.collect()
                    torch.mps.empty_cache()

            except KeyError as e:
                print(f"⚠️ Ligne ignorée (Clé manquante) : {e}")
                continue

    if count > 0:
        avg_loss = total_loss / count
        print("\n" + "="*40)
        print("📊 RÉSULTAT TEST DPO (Mpaga_Christophe)")
        print(f"Validation Loss : {avg_loss:.4f}")
        print("="*40)

        # Sauvegarde du résultat pour le rapport technique
        os.makedirs("reports/metrics", exist_ok=True)
        with open("reports/metrics/final_dpo_eval.json", "w") as f:
            json.dump({"test_loss": avg_loss, "samples": count}, f)
    else:
        print("❌ Aucune donnée n'a pu être évaluée.")

if __name__ == "__main__":
    evaluate_dpo_safe()
