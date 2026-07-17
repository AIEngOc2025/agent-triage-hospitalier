import argparse
import json
import os

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
        with open(file_path, "r", encoding="utf-8") as f:
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

        self.final_data.append({"instruction": clean_inst, "response": clean_resp})

    def save(self, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in self.final_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"✅ Terminé ! {len(self.final_data)} exemples prêts dans {output_path}")


# --- CONFIGURATION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process local data for SFT training.")
    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to the input CSV file (e.g., data/raw/mes_donnees.csv).",
    )
    parser.add_argument(
        "--csv_col_question",
        type=str,
        default="question",
        help="Column name for questions in CSV.",
    )
    parser.add_argument(
        "--csv_col_reponse",
        type=str,
        default="reponse",
        help="Column name for responses in CSV.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/train_sft.jsonl",
        help="Path to save the processed JSONL file.",
    )
    args = parser.parse_args()

    processor = LocalDataProcessor()
    if args.csv_path and os.path.exists(args.csv_path):
        processor.process_csv(
            args.csv_path,
            col_question=args.csv_col_question,
            col_reponse=args.csv_col_reponse,
        )
    processor.save(args.output_file)
