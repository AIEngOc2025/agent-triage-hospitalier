import json
import os
import random

# Chemins
INPUT_PATH = "data/processed/Mpaga_Christophe_1_Dataset_Train_SFT_052026.jsonl"
OUTPUT_PATH = "data/processed/Mpaga_Christophe_1_Dataset_Train_SFT_Final_5k.jsonl"

def balance_and_limit():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Erreur : Fichier source {INPUT_PATH} introuvable.")
        return

    fr_pool = []
    en_pool = []

    print("📖 Lecture du dataset global...")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                lang = data.get('clinical_metadata', {}).get('language', 'unknown')

                if lang == 'fr':
                    fr_pool.append(data)
                elif lang == 'en':
                    en_pool.append(data)
            except:
                continue

    print(f"📊 État initial : FR={len(fr_pool)} | EN={len(en_pool)}")

    # Vérification des quotas
    if len(fr_pool) < 2500 or len(en_pool) < 2500:
        print("⚠️ Attention : Un des pools est inférieur à 2500. On prendra le maximum possible.")

    # Échantillonnage aléatoire pour la diversité
    random.seed(42) # Pour la reproductibilité
    final_fr = random.sample(fr_pool, min(2500, len(fr_pool)))
    final_en = random.sample(en_pool, min(2500, len(en_pool)))

    final_dataset = final_fr + final_en
    random.shuffle(final_dataset) # Mélange pour l'entraînement

    # Sauvegarde
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for entry in final_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n" + "="*40)
    print("✅ DATASET SFT FINALISÉ (5 000 entrées)")
    print("="*40)
    print(f"🇫🇷 Français : {len(final_fr)} (50%)")
    print(f"🇺🇸 Anglais  : {len(final_en)} (50%)")
    print(f"📍 Chemin   : {OUTPUT_PATH}")
    print("="*40)

if __name__ == "__main__":
    balance_and_limit()
