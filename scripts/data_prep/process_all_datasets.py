import json
import os
import sys

# Ajout pour permettre l'import de app.api_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from datasets import load_dataset
from app.api_utils import MedicalAnonymizer


class UniversalMedicalProcessor:
    def __init__(self):
        """
        @definition : Initialise le processeur de données médicales avec un
                      anonymiseur et une liste de données.
        @args/params : Aucun.
        @return : Aucun.
        """
        self.anonymizer = MedicalAnonymizer()
        self.final_data = []

    def download_and_save_dataset(self, dataset_id, local_path, split="train"):
        """
        @definition : Télécharge un dataset depuis Hugging Face et le sauvegarde
                      localement en JSONL.
        @args/params :
            - dataset_id (str): L'identifiant du dataset sur Hugging Face.
            - local_path (str): Le chemin où sauvegarder le fichier JSONL.
            - split (str): La partition du dataset à télécharger (ex: 'train').
        @return : Aucun.
        """
        print(f"⬇️ {dataset_id} non trouvé localement. Téléchargement...")
        try:
            # Certains datasets nécessitent de faire confiance au code distant
            dataset = load_dataset(dataset_id, split=split, trust_remote_code=True)

            # S'assurer que le dossier de destination existe
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Sauvegarder au format JSONL
            dataset.to_json(local_path, orient="records", lines=True)

            print(f"✅ Dataset sauvegardé dans {local_path}")

        except Exception as e:
            print(f"❌ Erreur lors du téléchargement de {dataset_id}: {e}")
            # Créer un fichier vide pour ne pas retenter le téléchargement à chaque fois
            with open(local_path, "w"):
                pass  # Fichier vide

    def is_triage_relevant(self, text):
        """
        @definition : Vérifie si le texte est pertinent pour le triage médical.
        @args/params : text (str)
        @return : bool
        """
        # Vocabulaire attendu pour le triage
        triage_vocab = [
            "maximale",
            "modérée",
            "différée",
            "urgence",
            "triage",
            "priorité",
        ]
        text_lower = text.lower()
        return any(word in text_lower for word in triage_vocab)

    def process_file(self, file_path, source_name):
        """
        @definition : Traite un fichier, extrait les instructions/réponses,
                      anonymise et filtre pour ne garder que le triage.
                      Ajoute les métadonnées de source dynamiquement.
        @args/params : file_path (str), source_name (str)
        @return : Aucun.
        """
        filename = os.path.basename(file_path)
        print(f"--- 📂 Lecture et filtrage de : {filename} (Source: {source_name}) ---")

        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    instruction, response = None, None

                    # 1. Cas spécial : DPO (dpo_mix_en_train.jsonl)
                    if "chosen" in data and "prompt" in data:
                        instruction = data["prompt"]
                        response = data["chosen"]

                    # 2. Cas spécial : QCM (medmcqa, frenchmedmcqa, medical_mqca)
                    elif "question" in data and "cop" in data:
                        instruction = data["question"]
                        options = {0: "opa", 1: "opb", 2: "opc", 3: "opd", 4: "ope"}
                        cop = data["cop"]
                        if isinstance(cop, str):
                            idx = (
                                ord(cop.lower()) - ord("a")
                                if cop.isalpha()
                                else int(cop) - 1
                            )
                        else:
                            idx = cop
                        key_opt = options.get(idx, "opa")
                        response = data.get(key_opt, "Réponse non disponible")

                    # 3. Cas standard : Medical QA / MedQuAD
                    else:
                        instruction = (
                            data.get("instruction")
                            or data.get("question")
                            or data.get("Question")
                            or data.get("prompt")
                        )
                        response = (
                            data.get("response")
                            or data.get("answer")
                            or data.get("Answer")
                            or data.get("output")
                        )

                    # Logique de filtrage ajoutée
                    if (
                        instruction
                        and response
                        and self.is_triage_relevant(str(instruction) + str(response))
                    ):
                        # Anonymisation
                        clean_inst = self.anonymizer.anonymize_text(str(instruction))
                        clean_resp = self.anonymizer.anonymize_text(str(response))

                        self.final_data.append(
                            {
                                "instruction": clean_inst,
                                "response": clean_resp,
                                "clinical_metadata": {
                                    "source": source_name,
                                    "anonymized": True,
                                },
                            }
                        )
                        count += 1
                except (json.JSONDecodeError, KeyError):
                    continue

        print(f"✅ Terminé : {count} exemples de triage extraits de {source_name}.")

    def save(self, output_path):
        """
        @definition : Sauvegarde les données traitées dans un fichier JSONL.
        @args/params : output_path (str): Chemin vers le fichier de sortie.
        @return : Aucun.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in self.final_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(
            f"\n🚀 TOTAL GÉNÉRAL : {len(self.final_data)} exemples dans {output_path}"
        )


if __name__ == "__main__":
    processor = UniversalMedicalProcessor()

    # Mapping entre les noms de fichiers locaux et les identifiants Hugging Face
    DATASET_MAP = {
        "medical_qa_en_train.jsonl": ("pubmed_qa", "train", "PubMedQA"),
        "medical_qa_shared_task_en_train.jsonl": ("bioasq", "train", "BioASQ"),
        "medmcqa_en_train.jsonl": ("medmcqa", "train", "MedMCQA"),
        "frenchmedmcqa_fr_train.jsonl": (
            "Dr-BERT/FrenchMedMCQA",
            "train",
            "FrenchMedMCQA",
        ),
        "medquad_en_train.jsonl": ("kevinma/medquad", "train", "MedQuAD"),
        "medical_mqca_fr_train.jsonl": (
            "fids-lab/medical_mqca_fr",
            "train",
            "MedicalMQCA_FR",
        ),
    }

    base_path = "data/raw/"
    os.makedirs(base_path, exist_ok=True)

    for local_filename, (hf_id, split, source_name) in DATASET_MAP.items():
        full_path = os.path.join(base_path, local_filename)

        # Si le fichier n'existe pas ou est vide, on le télécharge
        if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
            processor.download_and_save_dataset(hf_id, full_path, split)

        # On traite le fichier (qu'il ait été téléchargé ou qu'il existait déjà)
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            processor.process_file(full_path, source_name)
