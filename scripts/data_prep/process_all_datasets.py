import json
import os

from anonymize import MedicalAnonymizer


class UniversalMedicalProcessor:
    def __init__(self):
        self.anonymizer = MedicalAnonymizer()
        self.final_data = []

    def process_file(self, file_path):
        filename = os.path.basename(file_path)
        print(f"--- 📂 Lecture de : {filename} ---")

        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    instruction, response = None, None

                    # 1. Cas spécial : DPO (dpo_mix_en_train.jsonl)
                    # Pour le SFT, on prend le prompt et la réponse 'chosen'
                    if 'chosen' in data and 'prompt' in data:
                        instruction = data['prompt']
                        response = data['chosen']

                    # 2. Cas spécial : QCM (medmcqa, frenchmedmcqa, medical_mqca)
                    # Ces fichiers ont souvent 'question' + 'opa', 'opb'... et 'cop' (index de la réponse)
                    elif 'question' in data and 'cop' in data:
                        instruction = data['question']
                        # On essaie de reconstruire la réponse textuelle à partir de l'option correcte
                        options = {0: 'opa', 1: 'opb', 2: 'opc', 3: 'opd', 4: 'ope'}
                        # Parfois cop est un int (0,1,2) ou un str ('A','B','C')
                        cop = data['cop']
                        if isinstance(cop, str):
                            # Convertit 'A' ou '1' en index
                            idx = ord(cop.lower()) - ord('a') if cop.isalpha() else int(cop) - 1
                        else:
                            idx = cop

                        key_opt = options.get(idx, 'opa')
                        response = data.get(key_opt, "Réponse non disponible")

                    # 3. Cas standard : Medical QA / MedQuAD
                    else:
                        instruction = data.get('instruction') or data.get('question') or data.get('Question') or data.get('prompt')
                        response = data.get('response') or data.get('answer') or data.get('Answer') or data.get('output')

                    if instruction and response:
                        # Anonymisation
                        clean_inst = self.anonymizer.anonymize_text(str(instruction))
                        clean_resp = self.anonymizer.anonymize_text(str(response))

                        self.final_data.append({
                            "instruction": clean_inst,
                            "response": clean_resp
                        })
                        count += 1
                except Exception:
                    continue

        print(f"✅ Terminé : {count} exemples extraits.")

    def save(self, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in self.final_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"\n🚀 TOTAL GÉNÉRAL : {len(self.final_data)} exemples dans {output_path}")

if __name__ == "__main__":
    processor = UniversalMedicalProcessor()

    # Liste exacte de tes fichiers
    fichiers = [
        'dpo_mix_en_train.jsonl',
        'medical_qa_en_train.jsonl',
        'medical_qa_shared_task_en_train.jsonl',
        'medmcqa_en_train.jsonl',
        'frenchmedmcqa_fr_train.jsonl',
        'medquad_en_train.jsonl',
        'medical_mqca_fr_train.jsonl'
    ]

    base_path = "data/raw/" # Vérifie que tes fichiers sont bien là

    for f in fichiers:
        full_path = os.path.join(base_path, f)
        if os.path.exists(full_path):
            processor.process_file(full_path)
        else:
            print(f"⚠️ Fichier introuvable : {full_path}")

    processor.save("data/processed/train_sft.jsonl")
