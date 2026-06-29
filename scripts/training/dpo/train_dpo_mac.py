import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

MODEL_ID = "Qwen/Qwen2.5-1.5B"
SFT_ADAPTERS = "./models/qwen3-medical-sft" # Là où tu as tes fichiers .safetensors
DPO_DATA = "data/processed/Mpaga_Christophe_1_Dataset_DPO_Final_052026.jsonl"

def train_dpo():
    # 1. Charger le modèle SFT (Base + Tes adaptateurs)
    print("🔄 Chargement du modèle SFT pour alignement...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map={"": "mps"}
    )
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTERS, is_trainable=True)

    # 2. Dataset DPO
    dataset = load_dataset("json", data_files=DPO_DATA, split="train[:100]") # Test sur 100

    # 3. Config DPO (Phase 3 de la mission)
    training_args = TrainingArguments(
        output_dir="./models/qwen3-medical-dpo",
        per_device_train_batch_size=1,
        max_steps=10, # Test local
        learning_rate=5e-7, # LR très bas pour le DPO
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none"
    )

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        beta=0.1, # Force de l'alignement
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_prompt_length=256,
        max_length=512,
    )

    print("⚡ Lancement de l'alignement DPO (Semaine 3)...")
    dpo_trainer.train()
    print("✅ POC DPO terminé !")

if __name__ == "__main__":
    train_dpo()
