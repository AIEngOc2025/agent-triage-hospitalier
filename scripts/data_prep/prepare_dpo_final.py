import json
import os

import spacy

# Chargement de SpaCy pour l'anonymisation demandée par le CHSA
try:
    nlp_en = spacy.load("en_core_web_lg")
except:
    os.system("python -m spacy download en_core_web_lg")
    nlp_en = spacy.load("en_core_web_lg")

def anonymize(text):
    doc = nlp_en(text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "PER"]:
            text = text.replace(ent.text, "<PATIENT>")
    return text

def prepare_dpo():
    input_path = "data/raw/dpo_mix_en_train.jsonl"
    output_path = "data/processed/Mpaga_Christophe_1_Dataset_DPO_Final_052026.jsonl"

    final_dpo = []
    print("🧪 Préparation du dataset d'alignement (DPO)...")

    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000: break # On limite pour la vitesse du POC

            data = json.loads(line)
            # On extrait le triplet prompt/chosen/rejected
            prompt = data['chosen'][0]['content']
            chosen = data['chosen'][1]['content']
            rejected = data['rejected'][1]['content']

            final_dpo.append({
                "prompt": f"<|im_start|>system\nTu es l'infirmier du CHSA.<|im_end|>\n<|im_start|>user\n{anonymize(prompt)}<|im_end|>\n<|im_start|>assistant\n",
                "chosen": anonymize(chosen),
                "rejected": anonymize(rejected)
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in final_dpo:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Dataset DPO créé : {len(final_dpo)} paires dans {output_path}")

if __name__ == "__main__":
    prepare_dpo()
