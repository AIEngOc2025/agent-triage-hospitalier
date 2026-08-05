import json
import os
import sys
from collections import Counter
from pathlib import Path

# Ajout pour permettre l'import de app.api_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from app.api_utils import MedicalAnonymizer

# Constantes pour la configuration
BASE_DATA_PATH = Path("data/processed")
INPUT_FILE = BASE_DATA_PATH / "train_sft.jsonl"
OUTPUT_FILE = BASE_DATA_PATH / "Mpaga_Christophe_1_Dataset_Train_SFT_052026.jsonl"

print("✅ Initialisation de l'anonymiseur médical...")
anonymizer = MedicalAnonymizer()


def detect_lang_by_content(text):
    """
    @definition : Détecte si un texte est en français ou en
    anglais basé sur des mots outils fréquents.
    @args/params : text (str).
    @return : Code langue ('fr' ou 'en').
    """
    # Mots outils très fréquents en français
    fr_words = {
        " le ",
        " la ",
        " les ",
        " est ",
        " vous ",
        " dans ",
        " pour ",
        " avec ",
        " une ",
    }
    text_lower = " " + text.lower() + " "
    if any(word in text_lower for word in fr_words):
        return "fr"
    return "en"


def process():
    """
    @definition : Analyse le fichier SFT, anonymise les textes
    et sauvegarde les résultats en ajoutant des métadonnées.
    Utilise maintenant le MedicalAnonymizer centralisé pour la cohérence.
    @args/params : Aucune.
    @return : Aucun résultat retourné (fichiers sauvegardés).
    """
    stats = Counter()
    final_data = []

    print("🚀 Analyse du contenu pour 21 007 lignes...")

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    instr = data["instruction"]
                    resp = data["response"]

                    # 1. Détection de langue par les mots du texte
                    lang = detect_lang_by_content(instr)

                    # 2. Anonymisation (Critère RGPD de la mission)
                    clean_inst = anonymizer.anonymize_text(instr, lang)
                    clean_resp = anonymizer.anonymize_text(resp, lang)

                    final_data.append(
                        {
                            "instruction": clean_inst,
                            "response": clean_resp,
                            "clinical_metadata": {
                                "language": lang,
                                "anonymized": True,
                            },
                        }
                    )
                    stats[lang] += 1
                except json.JSONDecodeError:
                    print(f"⚠️ Ligne {i} ignorée : JSON invalide.")
                except KeyError as e:
                    print(f"⚠️ Ligne {i} ignorée : Clé manquante {e}.")
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier d'entrée `{INPUT_FILE}` n'a pas été trouvé.")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in final_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n" + "=" * 40)
    print("🌍 AUDIT FINAL DU BILINGUISME")
    print("=" * 40)
    total = sum(stats.values())
    if total > 0:
        for lang, count in stats.items():
            pct = (count / total) * 100
            label = "🇫🇷 FR" if lang == "fr" else "🇺🇸 EN"
            print(f"{label:<10} : {count:>6} exemples ({pct:>6.2f}%)")
    else:
        print("Aucune donnée n'a été traitée.")
    print("=" * 40)
    print(f"✅ Livrable créé : {OUTPUT_FILE}")


if __name__ == "__main__":
    process()
