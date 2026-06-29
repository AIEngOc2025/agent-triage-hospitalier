
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# 1. Configuration
MODEL_ID = "Qwen/Qwen2.5-1.5B"
DATA_PATH = "data/processed/Mpaga_Christophe_1_Dataset_Train_SFT_Final_5k.jsonl"

# 2. Chargement de l'échantillon (50 lignes)
print("📥 Chargement du dataset...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train[:50]")

# 3. Tokenizer & Modèle (Optimisé MPS)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Device détecté : {device.upper()}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
    device_map={"": device}
)

# 4. Fonction de formatage CORRIGÉE (Traitement ligne par ligne)
def formatting_prompts_func(example):
    # On retourne simplement la chaîne de caractères formatée pour UNE ligne
    return f"### Instruction: {example['instruction']}\n### Response: {example['response']}"

# 5. Configuration LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

# 6. Arguments d'entraînement (SFTConfig pour éviter les warnings)
sft_config = SFTConfig(
    output_dir="models/test_local",
    max_steps=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="tensorboard",
   # max_seq_length=512,
    dataset_text_field="text", # Sera créé automatiquement par formatting_func
    packing=False
)

# 7. Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    #tokenizer=tokenizer,
    args=sft_config,
)

print("⚡ Lancement du test technique...")
trainer.train()
print("\n" + "="*40)
print("✅ TEST RÉUSSI : Le pipeline est 100% opérationnel !")
print("="*40)
