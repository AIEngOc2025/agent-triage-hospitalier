import json
from collections import Counter

import spacy

# Chargement de SpaCy (nécessaire pour l'anonymisation demandée par le CHSA)
try:
    nlp_fr = spacy.load("fr_core_news_lg")
    nlp_en = spacy.load("en_core_web_lg")
    print("✅ Modèles SpaCy chargés.")
except:
    print("❌ Erreur : Modèles SpaCy manquants.")
    exit()

def detect_lang_by_content(text):
    # Mots outils très fréquents en français
    fr_words = {" le ", " la ", " les ", " est ", " vous ", " dans ", " pour ", " avec ", " une "}
    text_lower = " " + text.lower() + " "
    if any(word in text_lower for word in fr_words):
        return "fr"
    return "en"

def anonymize_text(text, lang):
    nlp = nlp_fr if lang == "fr" else nlp_en
    doc = nlp(text)
    new_text = text
    # On remplace les noms de personnes détectés par SpaCy
    for ent in doc.ents:
        if ent.label_ in ["PER", "PERSON"]:
            new_text = new_text.replace(ent.text, "<PATIENT>")
    return new_text

def process():
    input_path = "data/processed/train_sft.jsonl"
    output_path = "data/processed/Mpaga_Christophe_1_Dataset_Train_SFT_052026.jsonl"

    stats = Counter()
    final_data = []

    print("🚀 Analyse du contenu pour 21 007 lignes...")

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                instr = data['instruction']
                resp = data['response']

                # 1. Détection de langue par les mots du texte
                lang = detect_lang_by_content(instr)

                # 2. Anonymisation (Critère RGPD de la mission)
                clean_inst = anonymize_text(instr, lang)
                clean_resp = anonymize_text(resp, lang)

                final_data.append({
                    "instruction": clean_inst,
                    "response": clean_resp,
                    "clinical_metadata": {
                        "language": lang,
                        "anonymized": True
                    }
                })
                stats[lang] += 1
            except:
                continue

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in final_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n" + "="*40)
    print("🌍 AUDIT FINAL DU BILINGUISME")
    print("="*40)
    total = sum(stats.values())
    for lang, count in stats.items():
        pct = (count/total)*100
        label = "🇫🇷 FR" if lang == "fr" else "🇺🇸 EN"
        print(f"{label:<10} : {count:>6} exemples ({pct:>6.2f}%)")
    print("="*40)
    print(f"✅ Livrable créé : {output_path}")

if __name__ == "__main__":
    process()
