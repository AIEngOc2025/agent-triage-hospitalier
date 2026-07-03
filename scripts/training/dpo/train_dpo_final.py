# --- 1. INSTALLATION ---

import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import DPOTrainer

parser = argparse.ArgumentParser(description="DPO Training Script")
parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model ID from Hugging Face.")
parser.add_argument("--sft_adapters_path", type=str, default="/kaggle/working/chsa_model_final", help="Path to the SFT adapters.")
parser.add_argument("--dpo_dataset_path", type=str, default="/kaggle/input/ton-dataset-dpo/Mpaga_Christophe_1_Dataset_Train_DPO_052026.jsonl", help="Path to the DPO training data.")
parser.add_argument("--output_dir", type=str, default="./qwen-chsa-dpo-final", help="Directory to save the DPO model.")
args, _ = parser.parse_known_args() # Use parse_known_args in notebooks to avoid conflicts

# --- 2. CONFIGURATION ---
MODEL_ID = args.model_id
SFT_ADAPTERS = args.sft_adapters_path
DPO_DATA = args.dpo_dataset_path
OUTPUT_DIR = args.output_dir

# --- 3. CHARGEMENT DU MODÈLE SFT ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32, # Stabilité
)

print("📥 Chargement du modèle SFT pour alignement...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# On injecte les poids du SFT (tes 4 époques)
model = PeftModel.from_pretrained(base_model, SFT_ADAPTERS, is_trainable=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# --- 4. CHARGEMENT DU DATASET DPO ---
dataset = load_dataset("json", data_files=DPO_DATA, split="train")

# --- 5. ARGUMENTS D'ENTRAÎNEMENT DPO (SÉCURISÉS) ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,     # DPO est lourd, on reste à 1
    gradient_accumulation_steps=8,     # Batch effectif de 8
    max_steps=100,                     # 100 à 200 étapes suffisent pour le DPO
    learning_rate=5e-7,                # LR très faible pour ne pas "casser" le SFT
    fp16=True,
    logging_steps=5,
    save_strategy="no",
    remove_unused_columns=False,       # Requis pour DPOTrainer
    report_to="none"
)

# --- 6. LE TRAINER DPO ---
dpo_trainer = DPOTrainer(
    model,
    ref_model=None,             # En QLoRA, laisser à None pour gagner de la RAM
    args=training_args,
    beta=0.1,                   # Paramètre de force de l'alignement (standard)
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_prompt_length=256,
    max_length=512,
)

print("⚡ Lancement de l'alignement DPO (Semaine 3)...")
dpo_trainer.train()

# --- 7. SAUVEGARDE FINALE ---
dpo_trainer.save_model(OUTPUT_DIR)
print(f"✅ POC DPO TERMINÉ ! Modèle prêt pour la Phase 4.")