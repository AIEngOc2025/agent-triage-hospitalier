import json
import os
import random

import spacy

# --- CONFIGURATION OFFICIELLE ---
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SEED = 42
TRAIN_RATIO = 0.8

FILE_TRAIN = "Mpaga_Christophe_1_Dataset_Train_DPO_052026.jsonl"
FILE_TEST = "Mpaga_Christophe_1_Dataset_Test_DPO_052026.jsonl"

# --- CHARGEMENT ANONYMISEUR ---
print("📥 Chargement des modèles linguistiques...")
try:  # Ruff: E722 - Do not use bare 'except'
    nlp_fr = spacy.load("fr_core_news_lg")
    nlp_en = spacy.load("en_core_web_lg")
except Exception:
    print("❌ Erreur : Modèles SpaCy manquants.")
    exit()


def anonymize(text, lang):
    """
    @definition : Anonymise les noms des personnes dans un texte
    en utilisant SpaCy.
    @args/params : text (str), lang (str).
    @return : Texte anonymisé (str).
    """
    if not text or not isinstance(text, str):
        return ""
    nlp = nlp_fr if lang == "fr" else nlp_en
    doc = nlp(text)
    new_text = text
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "PER"]:
            new_text = new_text.replace(ent.text, "<PATIENT>")
    return new_text


def format_prompt_chatml(text):
    """
    @definition : Formate un texte au format ChatML pour
    le modèle d'IA.
    @args/params : text (str).
    @return : Texte formaté (str).
    """
    # Format ChatML requis pour éviter les "!!!!" lors du test final
    return (
        f"<|im_start|>system\nTu es l'infirmier d'accueil bienveillant du CHSA."
        f"<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    )


def process_dpo_data():
    """
    @definition : Traite les données d'entraînement DPO,
    anonymise les informations et sauvegarde les sets
    d'entraînement et de test.
    @args/params : Aucune.
    @return : Aucun résultat retourné (fichiers sauvegardés).
    """
    final_pool = []

    # 1. TRAITEMENT DES DONNÉES FRANÇAISES (Sécurisé contre les KeyError)
    path_fr = os.path.join(RAW_DIR, "frenchmedmcqa_fr_train.jsonl")
    if os.path.exists(path_fr):
        print("🇫🇷 Traitement des données FR (Validation médicale)...")
        with open(path_fr, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 1000:
                    break
                try:
                    data = json.loads(line)

                    # Détection robuste de la réponse correcte
                    correct_ans = data.get("correct_answers")
                    if correct_ans is None or not isinstance(correct_ans, list):
                        continue

                    # Normalisation de l'index
                    ans_idx = correct_ans[0]

                    # Mapping des clés
                    key_map = {
                        0: "answer_a",
                        1: "answer_b",
                        2: "answer_c",
                        3: "answer_d",
                        4: "answer_e",
                    }

                    chosen_key = key_map.get(ans_idx)
                    chosen = data.get(chosen_key)

                    # Choix d'une mauvaise réponse
                    all_options = [data.get(k) for k in key_map.values() if data.get(k)]
                    wrong_options = [opt for opt in all_options if opt != chosen]

                    if chosen and wrong_options:
                        rejected = random.choice(wrong_options)
                        final_pool.append(
                            {
                                "prompt": format_prompt_chatml(
                                    anonymize(data.get("question"), "fr")
                                ),
                                "chosen": anonymize(chosen, "fr"),
                                "rejected": anonymize(rejected, "fr"),
                            }
                        )
                # Cible les erreurs de parsing JSON, de clés ou de types
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue

    # 3. MÉLANGE, SPLIT ET SAUVEGARDE
    if not final_pool:
        print("❌ Aucune donnée n'a pu être extraite. Vérifiez les noms des fichiers.")
        return

    random.seed(SEED)
    random.shuffle(final_pool)

    split_idx = int(len(final_pool) * TRAIN_RATIO)
    train_set = final_pool[:split_idx]
    test_set = final_pool[split_idx:]

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for name, data, filename in [
        ("TRAIN", train_set, FILE_TRAIN),
        ("TEST", test_set, FILE_TEST),
    ]:
        path = os.path.join(PROCESSED_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"✅ {name} DPO créé : {len(data)} paires -> {filename}")


if __name__ == "__main__":
    process_dpo_data()
