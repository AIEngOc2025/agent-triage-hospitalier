import json
from collections import Counter

import spacy

# 1. Chargement de l'anonymiseur léger pour Mac
try:
    nlp_fr = spacy.load("fr_core_news_lg")
    nlp_en = spacy.load("en_core_web_lg")
    print("✅ Modèles SpaCy chargés.")
except:
    print("❌ Erreur : Modèles SpaCy manquants. Lance : python -m spacy download fr_core_news_lg")
    exit()

def simple_anonymize(text, lang):
    nlp = nlp_fr if lang == "fr" else nlp_en
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["PER", "PERSON"]:
            text = text.replace(ent.text, "<PATIENT>")
        elif ent.label_ in ["LOC", "GPE"]:
            text = text.replace(ent.text, "<LIEU>")
    return text

def fix_and_audit():
    input_path = "data/processed/train_sft.jsonl"
    output_path = "data/processed/Mpaga_Christophe_1_Dataset_Train_SFT_052026.jsonl"

    stats = Counter()
    final_data = []

    print("🚀 Début du nettoyage et de l'anonymisation...")

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        try:
            data = json.loads(line)
            source = data.get('source', '').lower()

            # Détection précise de la langue
            if 'fr' in source or 'french' in source:
                lang = 'fr'
            else:
                lang = 'en'

            # Anonymisation (Correction de ton problème précédent)
            clean_inst = simple_anonymize(data['instruction'], lang)
            clean_resp = simple_anonymize(data['response'], lang)

            # Formatage final conforme au POC
            final_data.append({
                "instruction": clean_inst,
                "response": clean_resp,
                "clinical_metadata": {
                    "language": lang,
                    "source": source,
                    "anonymized": True
                }
            })
            stats[lang] += 1

        except Exception:
            continue

    # Sauvegarde du fichier final
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in final_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n" + "="*40)
    print("📊 RÉSULTAT FINAL DU BILINGUISME")
    print("="*40)
    total = sum(stats.values())
    for lang, count in stats.items():
        pct = (count / total) * 100
        flag = "🇫🇷" if lang == "fr" else "🇺🇸"
        print(f"{flag} {lang.upper()} : {count} exemples ({pct:.2f}%)")
    print("-" * 40)
    print(f"✅ Fichier anonymisé créé : {output_path}")
    print("="*40)

if __name__ == "__main__":
    fix_and_audit()
