import json
import random
from pathlib import Path

# Configuration
INPUT_FILE = Path("data/processed/train_sft_triage_only.jsonl")
OUTPUT_FILE = Path("data/processed/train_sft_balanced_50_50.jsonl")


def detect_lang(text):
    fr_words = {" le ", " la ", " les ", " est ", " vous ", " dans ", " pour "}
    text_lower = " " + text.lower() + " "
    return "fr" if any(word in text_lower for word in fr_words) else "en"


def balance_dataset():
    """
    @definition : Filtre et rééchantillonne pour obtenir un ratio 50/50 FR/EN.
    @args/params : Aucune.
    @return : Aucun résultat retourné (fichier sauvegardé).
    """
    fr_examples = []
    en_examples = []

    print(f"🔄 Lecture de {INPUT_FILE}...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            lang = detect_lang(data.get("instruction", ""))
            if lang == "fr":
                fr_examples.append(data)
            else:
                en_examples.append(data)

    # Équilibrage
    min_count = min(len(fr_examples), len(en_examples))
    print(f"📊 Stats avant équilibrage : FR={len(fr_examples)}, EN={len(en_examples)}")

    balanced_fr = random.sample(fr_examples, min_count)
    balanced_en = random.sample(en_examples, min_count)
    balanced_data = balanced_fr + balanced_en
    random.shuffle(balanced_data)

    print(
        f"✅ Équilibrage effectué : {min_count} exemples de chaque langue (Total: {len(balanced_data)})"
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in balanced_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"📁 Dataset 50/50 sauvegardé dans {OUTPUT_FILE}")


if __name__ == "__main__":
    balance_dataset()
