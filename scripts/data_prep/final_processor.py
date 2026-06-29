import json
import os

from anonymize import MedicalAnonymizer


class MedicalDataFinalProcessor:
    def __init__(self):
        self.anonymizer = MedicalAnonymizer()
        self.final_data = []

    def process_file(self, file_path):
        filename = os.path.basename(file_path)
        print(f"🔄 Traitement de : {filename}")
        count = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    inst, resp = None, None

                    # 1. medmcqa_en_train.jsonl (Options: opa, opb, opc, opd)
                    if 'cop' in data and 'opa' in data:
                        inst = data['question']
                        options = {0: 'opa', 1: 'opb', 2: 'opc', 3: 'opd'}
                        # 'cop' est l'index 0-3 de la bonne réponse
                        idx = int(data['cop'])
                        resp = data.get(options.get(idx, 'opa'))

                    # 2. frenchmedmcqa_fr_train.jsonl (Options: answer_a, answer_b...)
                    elif 'correct_answers' in data and 'answer_a' in data:
                        inst = data['question']
                        # correct_answers est souvent une liste ['a'] ou ['a', 'b']
                        ans_key = data['correct_answers'][0] if isinstance(data['correct_answers'], list) else data['correct_answers']
                        key = f"answer_{ans_key.lower()}"
                        resp = data.get(key)

                    # 3. medquad_en_train.jsonl (Standard question/answer)
                    elif 'answer' in data and 'question' in data:
                        inst = data['question']
                        resp = data['answer']

                    # 4. medical_mqca_fr_train.jsonl (Déjà formatté instruction/output)
                    elif 'instruction' in data and 'output' in data:
                        inst = data['instruction']
                        resp = data['output']

                    # 5. dpo_mix_en_train.jsonl (Format DPO : prompt/chosen)
                    elif 'prompt' in data and 'chosen' in data:
                        inst = data['prompt']
                        resp = data['chosen']

                    # 6. Fallback pour les autres (medical_qa_en...)
                    else:
                        inst = data.get('question') or data.get('instruction')
                        resp = data.get('answer') or data.get('output') or data.get('response')

                    if inst and resp:
                        # Anonymisation et ajout
                        self.final_data.append({
                            "instruction": self.anonymizer.anonymize_text(str(inst)),
                            "response": self.anonymizer.anonymize_text(str(resp))
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
        print(f"\n🚀 SUCCESS : {len(self.final_data)} exemples bilingues anonymisés créés.")

if __name__ == "__main__":
    processor = MedicalDataFinalProcessor()

    fichiers = [
        'dpo_mix_en_train.jsonl',
        'medical_qa_en_train.jsonl',
        'medical_qa_shared_task_en_train.jsonl',
        'medmcqa_en_train.jsonl',
        'frenchmedmcqa_fr_train.jsonl',
        'medquad_en_train.jsonl',
        'medical_mqca_fr_train.jsonl'
    ]

    for f in fichiers:
        path = os.path.join("data/raw", f)
        if os.path.exists(path):
            processor.process_file(path)

    processor.save("data/processed/train_sft.jsonl")
