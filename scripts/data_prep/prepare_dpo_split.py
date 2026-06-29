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
try:
    nlp_fr = spacy.load("fr_core_news_lg")
    nlp_en = spacy.load("en_core_web_lg")
except:
    os.system("python -m spacy download fr_core_news_lg")
    os.system("python -m spacy download en_core_web_lg")
    nlp_fr = spacy.load("fr_core_news_lg")
    nlp_en = spacy.load("en_core_web_lg")

def anonymize(text, lang):
    if not text or not isinstance(text, str): return ""
    nlp = nlp_fr if lang == "fr" else nlp_en
    doc = nlp(text)
    new_text = text
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "PER"]:
            new_text = new_text.replace(ent.text, "<PATIENT>")
    return new_text

def format_prompt_chatml(text):
    # Format ChatML requis pour éviter les "!!!!" lors du test final
    return f"<|im_start|>system\nTu es l'infirmier d'accueil bienveillant du CHSA.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

def process_dpo_data():
    final_pool = []

    # 1. TRAITEMENT DES DONNÉES ANGLAISES (Format Argilla détecté)
    path_en = os.path.join(RAW_DIR, "dpo_mix_en_train.jsonl")
    if os.path.exists(path_en):
        print("🇬🇧 Traitement des données EN (Format Argilla)...")
        with open(path_en, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 1500: break # Échantillon solide
                try:
                    data = json.loads(line)
                    # Extraction basée sur ton extrait JSON
                    prompt_raw = data['chosen'][0]['content']
                    chosen_raw = data['chosen'][1]['content']
                    rejected_raw = data['rejected'][1]['content']

                    final_pool.append({
                        "prompt": format_prompt_chatml(anonymize(prompt_raw, "en")),
                        "chosen": anonymize(chosen_raw, "en"),
                        "rejected": anonymize(rejected_raw, "en")
                    })
                except (KeyError, IndexError): continue

    # 2. TRAITEMENT DES DONNÉES FRANÇAISES (Sécurisé contre les KeyError)
    path_fr = os.path.join(RAW_DIR, "frenchmedmcqa_fr_train.jsonl")
    if os.path.exists(path_fr):
        print("🇫🇷 Traitement des données FR (Validation médicale)...")
        with open(path_fr, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 1000: break
                try:
                    data = json.loads(line)

                    # Détection robuste de la réponse correcte
                    correct_ans = data.get('correct_answers') or data.get('cop')
                    if correct_ans is None: continue

                    # Normalisation de l'index (si c'est une liste ["a"] ou un int 0)
                    ans_idx = correct_ans[0] if isinstance(correct_ans, list) else correct_ans
                    ans_char = str(ans_idx).lower()

                    # Mapping des clés possibles
                    key_map = {'a': 'answer_a', 'b': 'answer_b', 'c': 'answer_c', 'd': 'answer_d', 'e': 'answer_e',
                               '0': 'opa', '1': 'opb', '2': 'opc', '3': 'opd'}

                    chosen_key = key_map.get(ans_char, f"answer_{ans_char}")
                    chosen = data.get(chosen_key) or data.get(f"op{ans_char}")

                    # Choix d'une mauvaise réponse
                    wrong_options = [data.get(k) for k in ['answer_a', 'answer_b', 'answer_c', 'answer_d', 'opa', 'opb']
                                     if data.get(k) and data.get(k) != chosen]

                    if chosen and wrong_options:
                        final_pool.append({
                            "prompt": format_prompt_chatml(anonymize(data.get('question'), "fr")),
                            "chosen": anonymize(chosen, "fr"),
                            "rejected": anonymize(random.choice(wrong_options), "fr")
                        })
                except Exception: continue

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
    for name, data, filename in [("TRAIN", train_set, FILE_TRAIN), ("TEST", test_set, FILE_TEST)]:
        path = os.path.join(PROCESSED_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"✅ {name} DPO créé : {len(data)} paires -> {filename}")

if __name__ == "__main__":
    process_dpo_data()
