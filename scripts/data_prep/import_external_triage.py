import json
import os
from datasets import load_dataset
from app.api_utils import MedicalAnonymizer

def process_external_triage_dataset(dataset_id, output_path):
    """
    @definition : Charge, anonymise et formate le dataset de triage externe.
    @args/params : dataset_id (str), output_path (str)
    @return : None
    """
    print(f"⬇️ Chargement du dataset {dataset_id}...")
    dataset = load_dataset(dataset_id, split="train")
    anonymizer = MedicalAnonymizer()
    
    processed_data = []
    
    for row in dataset:
        try:
            # Extraction des données
            symptoms = ", ".join(row["presentation"]["symptoms"])
            urgency = row["triage_classification"]["urgency_category"]
            reasoning = row["triage_classification"]["urgency_reasoning"]
            
            # Formatage pour l'agent
            instruction = f"Patient avec les symptômes suivants : {symptoms}. Quel est le niveau d'urgence ?"
            response = f"[Niveau: {urgency}] - Orientation : {reasoning}"
            
            # Anonymisation
            clean_inst = anonymizer.anonymize_text(instruction)
            clean_resp = anonymizer.anonymize_text(response)
            
            processed_data.append({"instruction": clean_inst, "response": clean_resp})
            
        except KeyError as e:
            continue
            
    # Sauvegarde
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in processed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"✅ {len(processed_data)} exemples traités et sauvegardés dans {output_path}")

if __name__ == "__main__":
    process_external_triage_dataset("syntech-ai/medical-triage-500", "data/processed/new_triage_data.jsonl")
