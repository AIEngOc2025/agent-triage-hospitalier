
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# --- 1. CONFIGURATION ---
# Qwen2.5-1.5B est le choix parfait pour la dimension 1536 (le 1.7B du PDF)
MODEL_ID = "Qwen/Qwen2.5-1.5B"
DATA_PATH = "data/processed/Mpaga_Christophe_1_Dataset_Train_SFT_052026.jsonl"
OUTPUT_DIR = "models/sft_final_chsa"

# --- 2. ACCÉLÉRATION MAC M1 (MPS) ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Entraînement sur : {device.upper()}")

# --- 3. CHARGEMENT DU DATASET ---
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# --- 4. TOKENIZER & MODÈLE DE BASE ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# On charge le modèle brut (SANS get_peft_model ici)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32, # Crucial pour la stabilité sur puce M1
    device_map={"": device},
    trust_remote_code=True
)

# --- 5. CONFIGURATION LORA (Semaine 2) ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# --- 6. FORMATAGE CHATML (Infaillible pour Qwen) ---
def formatting_prompts_func(example):
    # Formatage propre respectant la structure apprise par Qwen
    return f"<|im_start|>system\nTu es l'infirmier d'accueil du CHSA.<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"

# --- 7. ARGUMENTS D'ENTRAÎNEMENT ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-2,            # LR légèrement baissé pour éviter les instabilités
    num_train_epochs=3,
    logging_steps=5,
    logging_dir="./logs",
    save_strategy="epoch",
    # --- SPÉCIFIQUE MAC M1 ---
    fp16=False,                    # On désactive fp16 (incompatible MPS GradScaler)
    bf16=False,                    # On désactive bf16 (non supporté sur M1)
    max_grad_norm=0.3,             # Empêche l'explosion des gradients (évite les "!!!!")
    report_to="tensorboard"
)

# --- 8. TRAINER (C'est lui qui applique LoRA maintenant) ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,       # Le Trainer applique LoRA automatiquement
    formatting_func=formatting_prompts_func,
    args=training_args,
)

print("⚡ Démarrage de l'entraînement SFT CHSA...")
trainer.train()

# Sauvegarde finale (Adaptateurs + Tokenizer)
trainer.save_model(OUTPUT_DIR)
print(f"✅ Modèle sauvegardé avec succès dans {OUTPUT_DIR}")
