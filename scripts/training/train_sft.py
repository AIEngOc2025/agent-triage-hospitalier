import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# 1. CONFIGURATION
MODEL_NAME = "Qwen/Qwen2.5-1.5B" # Version stable recommandée
DATA_PATH = "data/processed/train_sft.jsonl"
OUTPUT_DIR = "models/checkpoints/qwen-sft-medical"

# Détection de l'accélérateur (Mac vs Cloud)
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"🚀 Utilisation du device : {device}")

# 2. CHARGEMENT DU DATASET
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
# Pour le test sur Mac, on ne prend que 50 exemples pour ne pas tout bloquer
dataset = dataset.select(range(50))

# 3. TOKENIZER & MODÈLE
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    device_map={"": device} if device != "cpu" else None,
    trust_remote_code=True
)

# 4. CONFIGURATION LORA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Cibles pour Qwen
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 5. ARGUMENTS D'ENTRAÎNEMENT
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=10, # On fait 10 étapes pour tester sur Mac
    logging_steps=1,
    save_strategy="no",
    bf16=False, # MPS ne supporte pas toujours bien le bf16
    fp16=True if device == "cuda" else False,
    push_to_hub=False,
    report_to="none"
)

# 6. LANCEMENT DU TRAINER
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
    dataset_text_field="instruction", # Champ source
    max_seq_length=512,
)

print("⚡ Début de l'entraînement de test...")
trainer.train()
print("✅ Test terminé ! Modèle prêt à être entraîné sur le Cloud.")
