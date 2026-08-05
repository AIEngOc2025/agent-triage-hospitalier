import json
import os
import random

# Chemins
INPUT_PATH = "data/processed/train_sft.jsonl"
OUTPUT_DIR = "data/processed/"
TRAIN_RATIO = 0.8


def split_dataset():
    """
    @definition : Divise le dataset SFT nettoyé (instruction/response) en jeux de Train et Val.
    @args/params : Aucun.
    @return : None.
    """
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Erreur : Fichier {INPUT_PATH} introuvable.")
        return

    data = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    # Mélanger
    random.seed(42)
    random.shuffle(data)

    # Découper
    split_idx = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Sauvegarde
    train_path = os.path.join(OUTPUT_DIR, "train_sft_split.jsonl")
    val_path = os.path.join(OUTPUT_DIR, "val_sft_split.jsonl")

    for path, dataset in [(train_path, train_data), (val_path, val_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"✅ {len(dataset)} exemples sauvegardés dans {path}")


if __name__ == "__main__":
    split_dataset()
