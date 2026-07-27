import json
from collections import Counter

import spacy

# Chargement de SpaCy (nécessaire pour l'anonymisation demandée par le CHSA)
try:
    nlp_fr = spacy.load("fr_core_news_lg")
    nlp_en = spacy.load("en_core_web_lg")
    print("✅ Modèles SpaCy chargés.")
except Exception:
    print("❌ Erreur : Modèles SpaCy manquants.")
    exit()


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


def anonymize_text(text, lang):
    """
    @definition : Anonymise les noms des personnes dans un
    texte en utilisant SpaCy.
    @args/params : text (str), lang (str).
    @return : Texte anonymisé (str).
    """
    nlp = nlp_fr if lang == "fr" else nlp_en
    doc = nlp(text)
    new_text = text
    # On remplace les noms de personnes détectés par SpaCy
    for ent in doc.ents:
        if ent.label_ in ["PER", "PERSON"]:
            new_text = new_text.replace(ent.text, "<PATIENT>")
    return new_text


def process():
    """
    @definition : Analyse le fichier SFT, anonymise les textes
    et sauvegarde les résultats en ajoutant des métadonnées.
    @args/params : Aucune.
    @return : Aucun résultat retourné (fichiers sauvegardés).
    """
    input_path = "data/processed/train_sft.jsonl"
    output_path = "data/processed/Mpaga_Christophe_1_Dataset_Train_SFT_052026.jsonl"

    stats = Counter()
    final_data = []

    print("🚀 Analyse du contenu pour 21 007 lignes...")

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                instr = data["instruction"]
                resp = data["response"]

                # 1. Détection de langue par les mots du texte
                lang = detect_lang_by_content(instr)

                # 2. Anonymisation (Critère RGPD de la mission)
                clean_inst = anonymize_text(instr, lang)
                clean_resp = anonymize_text(resp, lang)

                final_data.append(
                    {
                        "instruction": clean_inst,
                        "response": clean_resp,
                        "clinical_metadata": {"language": lang, "anonymized": True},
                    }
                )
                stats[lang] += 1
            except Exception:
                continue

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in final_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n" + "=" * 40)
    print("🌍 AUDIT FINAL DU BILINGUISME")
    print("=" * 40)
    total = sum(stats.values())
    for lang, count in stats.items():
        pct = (count / total) * 100
        label = "🇫🇷 FR" if lang == "fr" else "🇺🇸 EN"
        print(f"{label:<10} : {count:>6} exemples ({pct:>6.2f}%)")
    print("=" * 40)
    print(f"✅ Livrable créé : {output_path}")


if __name__ == "__main__":
    process()
