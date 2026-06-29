import json

from anonymize import MedicalAnonymizer  # On importe ton script précédent
from datasets import load_dataset


class MedicalDataProcessor:
    def __init__(self):
        self.anonymizer = MedicalAnonymizer()
        self.final_data = []

    def format_french_med_mcqa(self):
        print("Chargement de FrenchMedMCQA...")
        # Dataset de QCM médicaux en Français
        ds = load_dataset("frenchmedmcqa", trust_remote_code=True)

        for item in ds['train']:
            # On transforme le QCM en une question/réponse simple
            instruction = f"Question médicale : {item['question']}"
            # On récupère la réponse correcte parmi les options
            options = [item['opa'], item['opb'], item['opc'], item['opd'], item['ope']]
            correct_idx = ord(item['cop'].lower()) - ord('a')
            response = f"La réponse correcte est : {options[correct_idx]}"

            self.add_to_final(instruction, response)

    def format_mediqa(self):
        print("Chargement de MediQA (English)...")
        # Questions/Réponses médicales (Anglais)
        ds = load_dataset("lavis-nlp/MediQA-QA", trust_remote_code=True)

        for item in ds['train']:
            instruction = f"Medical Question: {item['Question']}"
            response = item['Answer']
            self.add_to_final(instruction, response)

    def add_to_final(self, instruction, response):
        # On anonymise avant d'ajouter
        clean_instruction = self.anonymizer.anonymize_text(instruction)
        clean_response = self.anonymizer.anonymize_text(response)

        self.final_data.append({
            "instruction": clean_instruction,
            "response": clean_response
        })

    def save_to_jsonl(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in self.final_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"✅ Dataset sauvegardé : {len(self.final_data)} exemples dans {filepath}")

if __name__ == "__main__":
    processor = MedicalDataProcessor()

    # Exécuter le chargement (on limite pour le test sur Mac)
    processor.format_french_med_mcqa()
    processor.format_mediqa()

    # Sauvegarde dans le dossier processed créé par ton script zsh
    processor.save_to_jsonl("../../data/processed/train_sft.jsonl")
