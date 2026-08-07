import json
import pandas as pd
from app.api_utils import MedicalAnonymizer

def process_raw_jsonl(url, output_path):
    """
    @definition : Lit un JSONL directement depuis une URL, anonymise et formate.
    @args/params : url (str), output_path (str)
    @return : None
    """
    print(f"⬇️ Lecture directe depuis : {url}...")
    df = pd.read_json(url, lines=True)
    anonymizer = MedicalAnonymizer()
    
    processed_data = []
    
    for _, row in df.iterrows():
        try:
            # Extraction basée sur la structure vue dans l'erreur
            symptoms = ", ".join(row["presentation"]["symptoms"])
            urgency = row["triage_classification"]["urgency_category"]
            reasoning = row["triage_classification"]["urgency_reasoning"]
            
            # Formatage
            instruction = f"Patient avec les symptômes suivants : {symptoms}. Quel est le niveau d'urgence ?"
            response = f"[Niveau: {urgency}] - Orientation : {reasoning}"
            
            # Anonymisation
            clean_inst = anonymizer.anonymize_text(instruction)
            clean_resp = anonymizer.anonymize_text(response)
            
            processed_data.append({"instruction": clean_inst, "response": clean_resp})
            
        except Exception as e:
            continue
            
    # Sauvegarde
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in processed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"✅ {len(processed_data)} exemples traités et sauvegardés dans {output_path}")

if __name__ == "__main__":
    url = "https://huggingface.co/datasets/syntech-ai/medical-triage-500/resolve/main/medical_triage_500.jsonl"
    process_raw_jsonl(url, "data/processed/new_triage_data.jsonl")
