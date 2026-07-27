import argparse
import json

from anonymize import MedicalAnonymizer  # On importe ton script précédent
from datasets import load_dataset


class MedicalDataProcessor:
    def __init__(self):
        """
        @definition : Initialise le processeur de données médicales avec un anonymiseur.
        @args/params : Aucune.
        @return : Aucun résultat retourné.
        """
        self.anonymizer = MedicalAnonymizer()
        self.final_data = []

    def format_french_med_mcqa(self):
        """
        @definition : Charge et transforme le dataset FrenchMedMCQA
        en format instruction/réponse.
        @args/params : Aucune.
        @return : Aucun résultat retourné (données stockées dans
        self.final_data).
        """
        print("Chargement de FrenchMedMCQA...")
        # Dataset de QCM médicaux en Français (limité pour le test)
        ds = load_dataset(
            "frenchmedmcqa",
            trust_remote_code=True,
            split=f"train[:{self.limit_samples}]",
        )

        for item in ds["train"]:
            # On transforme le QCM en une question/réponse simple
            instruction = f"Question médicale : {item['question']}"
            # On récupère la réponse correcte parmi les options
            options = [item["opa"], item["opb"], item["opc"], item["opd"], item["ope"]]
            # Assurez-vous que 'cop' existe et est valide
            if (
                "cop" in item
                and item["cop"]
                and item["cop"].lower() in ["a", "b", "c", "d", "e"]
            ):
                correct_idx = ord(item["cop"].lower()) - ord("a")
                response = f"La réponse correcte est : {options[correct_idx]}"
            else:
                # Gérer les cas où 'cop' est manquant ou invalide
                print(
                    f"Avertissement: 'cop' manquant ou invalide pour l'élément: {item}"
                )
                continue

            self.add_to_final(instruction, response)

    def format_mediqa(self):
        """
        @definition : Charge et transforme le dataset MediQA
        (Anglais) en format instruction/réponse.
        @args/params : Aucune.
        @return : Aucun résultat retourné (données stockées dans
        self.final_data).
        """
        print("Chargement de MediQA (English)...")
        # Questions/Réponses médicales (Anglais)
        # Limité pour le test
        ds = load_dataset(
            "lavis-nlp/MediQA-QA",
            trust_remote_code=True,
            split=f"train[:{self.limit_samples}]",
        )

        for item in ds["train"]:
            instruction = f"Medical Question: {item['Question']}"
            response = item["Answer"]
            self.add_to_final(instruction, response)

    def add_to_final(self, instruction, response):
        """
        @definition : Anonymise l'instruction et la réponse,
        puis les ajoute à la liste finale.
        @args/params : instruction (str), response (str).
        @return : Aucun résultat retourné (données stockées dans
        self.final_data).
        """
        # On anonymise avant d'ajouter
        clean_instruction = self.anonymizer.anonymize_text(instruction)
        clean_response = self.anonymizer.anonymize_text(response)

        self.final_data.append(
            {"instruction": clean_instruction, "response": clean_response}
        )

    def save_to_jsonl(self, filepath):
        """
        @definition : Sauvegarde les données traitées dans un fichier au format JSONL.
        @args/params : filepath (str).
        @return : Aucun résultat retourné.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            for entry in self.final_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(
            f"✅ Dataset sauvegardé : {len(self.final_data)} exemples dans {filepath}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process medical datasets for SFT training."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="../../data/processed/train_sft.jsonl",
        help="Path to save the processed JSONL file.",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=50,
        help="Limit the number of samples loaded from each dataset for testing/POC.",
    )
    args = parser.parse_args()

    processor = MedicalDataProcessor()
    processor.limit_samples = args.limit_samples  # Inject the limit

    # Exécuter le chargement (on limite pour le test sur Mac)
    processor.format_french_med_mcqa()
    processor.format_mediqa()

    # Sauvegarde dans le dossier processed créé par ton script zsh
    processor.save_to_jsonl(args.output_file)
