import gc

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
ADAPTERS = "models/sft_final_chsa"
VAL_FILE = "data/processed/Mpaga_Christophe_1_Dataset_Val_SFT_052026.jsonl"

def evaluate_validation_light():
    # Nettoyage initial
    gc.collect()
    torch.mps.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Chargement en float32 pour la stabilité sur M1
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                    dtype=torch.float32,
                                                    device_map={"": "mps"}
                                                    )
    model = PeftModel.from_pretrained(base_model, ADAPTERS)

    val_dataset = load_dataset("json", data_files=VAL_FILE, split="train")

    def format_chatml(ex):
        return {"text": f"<|im_start|>system\nTu es l'infirmier du CHSA.<|im_end|>\n<|im_start|>user\n{ex['instruction']}<|im_end|>\n<|im_start|>assistant\n{ex['response']}<|im_end|>"}

    val_dataset = val_dataset.map(format_chatml)

    # Arguments ultra-légers
    eval_args = TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=1, # Crucial : 1 par 1
        dataloader_pin_memory=False, # Désactivé pour MPS
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=val_dataset,
        eval_dataset=val_dataset,
        #dataset_text_field="text",
        #max_seq_length=256,
        args=eval_args
    )

    print("🔬 Calcul de la Validation Loss (Mode Sécurisé)...")
    metrics = trainer.evaluate()

    print(f"\n📊 RÉSULTAT VALIDATION SFT : {metrics['eval_loss']:.4f}")
    return metrics['eval_loss']

if __name__ == "__main__":
    evaluate_validation_light()
