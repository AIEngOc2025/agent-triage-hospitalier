import json
import os

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
        @definition : Télécharge un dataset depuis Hugging Face et le sauvegarde localement en JSONL.
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
            with open(local_path, "w") as f:
                pass  # Fichier vide

    def process_file(self, file_path):
        """
        @definition : Traite un fichier de données (JSONL), extrait les
                      instructions et les réponses, et anonymise le contenu.
        @args/params : file_path (str): Chemin vers le fichier à traiter.
        @return : Aucun.
        """
        filename = os.path.basename(file_path)
        print(f"--- 📂 Lecture de : {filename} ---")

        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    instruction, response = None, None

                    # 1. Cas spécial : DPO (dpo_mix_en_train.jsonl)
                    # Pour le SFT, on prend le prompt et la réponse 'chosen'
                    if "chosen" in data and "prompt" in data:
                        instruction = data["prompt"]
                        response = data["chosen"]

                    # 2. Cas spécial : QCM (medmcqa, frenchmedmcqa, medical_mqca)
                    # Ces fichiers ont souvent 'question' + 'opa', 'opb'...
                    # et 'cop' (index de la réponse)
                    elif "question" in data and "cop" in data:
                        instruction = data["question"]
                        # On essaie de reconstruire la réponse textuelle
                        # à partir de l'option correcte
                        options = {0: "opa", 1: "opb", 2: "opc", 3: "opd", 4: "ope"}
                        # Parfois cop est un int (0,1,2) ou un str ('A','B','C')
                        cop = data["cop"]
                        if isinstance(cop, str):
                            # Convertit 'A' ou '1' en index
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

                    if instruction and response:
                        # Anonymisation
                        clean_inst = self.anonymizer.anonymize_text(str(instruction))
                        clean_resp = self.anonymizer.anonymize_text(str(response))

                        self.final_data.append(
                            {"instruction": clean_inst, "response": clean_resp}
                        )
                        count += 1
                # Cible les erreurs attendues (JSON malformé, clé manquante)
                # pour ne pas masquer d'autres problèmes.
                except (json.JSONDecodeError, KeyError):
                    continue

        print(f"✅ Terminé : {count} exemples extraits.")

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
    # Cela permet de télécharger les datasets s'ils sont manquants.
    DATASET_MAP = {
        "medical_qa_en_train.jsonl": ("pubmed_qa", "train"),
        "medical_qa_shared_task_en_train.jsonl": ("bioasq", "train"),
        "medmcqa_en_train.jsonl": ("medmcqa", "train"),
        "frenchmedmcqa_fr_train.jsonl": ("Dr-BERT/FrenchMedMCQA", "train"),
        "medquad_en_train.jsonl": ("kevinma/medquad", "train"),
        "medical_mqca_fr_train.jsonl": ("fids-lab/medical_mqca_fr", "train"),
    }

    base_path = "data/raw/"
    os.makedirs(base_path, exist_ok=True)

    for local_filename, (hf_id, split) in DATASET_MAP.items():
        full_path = os.path.join(base_path, local_filename)

        # Si le fichier n'existe pas ou est vide, on le télécharge
        if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
            processor.download_and_save_dataset(hf_id, full_path, split)

        # On traite le fichier (qu'il ait été téléchargé ou qu'il existait déjà)
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            processor.process_file(full_path)

    # Sauvegarde du résultat final combiné
    processor.save("data/processed/train_sft.jsonl")
