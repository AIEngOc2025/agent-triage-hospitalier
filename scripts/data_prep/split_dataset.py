import json
import os
import random

# Chemins
INPUT_PATH = "data/processed/Mpaga_Christophe_1_Dataset_Train_SFT_Final_5k.jsonl"
OUTPUT_DIR = "data/processed/"

def split_dataset():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Erreur : Fichier {INPUT_PATH} introuvable.")
        return

    # 1. Charger les données par langue pour garder l'équilibre dans les splits
    fr_data = []
    en_data = []

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item['clinical_metadata']['language'] == 'fr':
                fr_data.append(item)
            else:
                en_data.append(item)

    # 2. Mélanger
    random.seed(42)
    random.shuffle(fr_data)
    random.shuffle(en_data)

    # 3. Définir les indices de découpe (80% Train, 10% Val, 10% Test)
    # Pour 2500 de chaque langue : 2000 Train, 250 Val, 250 Test
    train_fr, val_fr, test_fr = fr_data[:2000], fr_data[2000:2250], fr_data[2250:]
    train_en, val_en, test_en = en_data[:2000], en_data[2000:2250], en_data[2250:]

    # 4. Fusionner et mélanger les groupes
    splits = {
        "Train": train_fr + train_en,
        "Val": val_fr + val_en,
        "Test": test_fr + test_en
    }

    # 5. Sauvegarde des 3 fichiers
    print("\n📦 --- COMPARTIMENTAGE DU DATASET ---")
    for name, data in splits.items():
        random.shuffle(data) # Mélange final
        file_name = f"Mpaga_Christophe_1_Dataset_{name}_SFT_052026.jsonl"
        path = os.path.join(OUTPUT_DIR, file_name)

        with open(path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Calcul du bilinguisme par split pour vérification
        fr_count = sum(1 for x in data if x['clinical_metadata']['language'] == 'fr')
        en_count = sum(1 for x in data if x['clinical_metadata']['language'] == 'en')

        print(f"✅ {name:<5} : {len(data):>4} exemples (FR: {fr_count} | EN: {en_count})")
        print(f"      📍 {path}")

if __name__ == "__main__":
    split_dataset()
