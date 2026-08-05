import json
import os
import sys
from collections import Counter

# Ajout pour permettre l'import de app.api_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from app.api_utils import MedicalAnonymizer

print("✅ Initialisation de l'anonymiseur médical...")
anonymizer = MedicalAnonymizer()


def fix_and_audit():
    """
    @definition : Nettoie et anonymise le dataset d'entraînement SFT et
                  génère un rapport de bilinguisme.
    @args/params : Aucun.
    @return : None.
    """
    input_path = "data/processed/train_sft.jsonl"
    output_path = "data/processed/Mpaga_Christophe_1_Dataset_Train_SFT_052026.jsonl"

    stats = Counter()
    final_data = []

    print("🚀 Début du nettoyage et de l'anonymisation...")

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        try:
            data = json.loads(line)
            source = data.get("source", "").lower()

            # Détection précise de la langue
            if "fr" in source or "french" in source:
                lang = "fr"
            else:
                lang = "en"

            # Anonymisation (Correction de ton problème précédent)
            clean_inst = anonymizer.anonymize_text(data["instruction"], lang)
            clean_resp = anonymizer.anonymize_text(data["response"], lang)

            # Formatage final conforme au POC
            final_data.append(
                {
                    "instruction": clean_inst,
                    "response": clean_resp,
                    "clinical_metadata": {
                        "language": lang,
                        "source": source,
                        "anonymized": True,
                    },
                }
            )
            stats[lang] += 1

        except Exception:
            continue

    # Sauvegarde du fichier final
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in final_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n" + "=" * 40)
    print("📊 RÉSULTAT FINAL DU BILINGUISME")
    print("=" * 40)
    total = sum(stats.values())
    for lang, count in stats.items():
        pct = (count / total) * 100
        flag = "🇫🇷" if lang == "fr" else "🇺🇸"
        print(f"{flag} {lang.upper()} : {count} exemples ({pct:.2f}%)")
    print("-" * 40)
    print(f"✅ Fichier anonymisé créé : {output_path}")
    print("=" * 40)


if __name__ == "__main__":
    fix_and_audit()
