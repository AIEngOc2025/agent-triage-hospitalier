import json
import random
from pathlib import Path

# Configuration
INPUT_FILE = Path("data/processed/train_sft.jsonl")
TRAIN_OUTPUT = Path("data/processed/train_sft_split.jsonl")
VAL_OUTPUT = Path("data/processed/val_sft_split.jsonl")
VAL_RATIO = 0.1


def split_dataset():
    """
    @definition : Divise le dataset SFT fusionné en train (90%) et validation (10%).
    @args/params : Aucune.
    @return : Aucun résultat retourné (fichiers sauvegardés).
    """
    print(f"🔄 Lecture de {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)

    split_idx = int(len(data) * (1 - VAL_RATIO))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"📊 Taille totale : {len(data)}")
    print(f"📈 Train : {len(train_data)} | Validation : {len(val_data)}")

    with open(TRAIN_OUTPUT, "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(VAL_OUTPUT, "w", encoding="utf-8") as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Splits créés : {TRAIN_OUTPUT} et {VAL_OUTPUT}")


if __name__ == "__main__":
    split_dataset()
