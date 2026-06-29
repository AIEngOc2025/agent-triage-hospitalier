import json

import pandas as pd
from anonymize import MedicalAnonymizer


class LocalDataProcessor:
    def __init__(self):
        self.anonymizer = MedicalAnonymizer()
        self.final_data = []

    def process_csv(self, file_path, col_question, col_reponse):
        print(f"Traitement du CSV : {file_path}")
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            self.add_to_final(row[col_question], row[col_reponse])

    def process_json(self, file_path, key_question, key_reponse):
        print(f"Traitement du JSON : {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                self.add_to_final(item[key_question], item[key_reponse])

    def add_to_final(self, instruction, response):
        # On s'assure que ce sont des strings
        instruction = str(instruction)
        response = str(response)

        # ANONYMISATION (Obligatoire pour le livrable CHSA)
        clean_inst = self.anonymizer.anonymize_text(instruction)
        clean_resp = self.anonymizer.anonymize_text(response)

        self.final_data.append({
            "instruction": clean_inst,
            "response": clean_resp
        })

    def save(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in self.final_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"✅ Terminé ! {len(self.final_data)} exemples prêts dans {output_path}")

# --- CONFIGURATION ---
if __name__ == "__main__":
    processor = LocalDataProcessor()

    # EXEMPLE : Si tu as un fichier CSV nommé 'mes_donnees.csv' dans data/raw
    # remplace 'question' et 'reponse' par les vrais noms de tes colonnes
    # path_csv = "data/raw/mes_donnees.csv"
    # if os.path.exists(path_csv):
    #    processor.process_csv(path_csv, col_question="question", col_reponse="reponse")

    # Sauvegarde finale
    processor.save("data/processed/train_sft.jsonl")
