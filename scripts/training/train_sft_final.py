import argparse
import glob
import os
import matplotlib.pyplot as plt

# Désactiver le parallélisme des tokenizers pour éviter les deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig


def format_chatml(ex):
    # Format ChatML strict pour Qwen
    return {
        "text": f"<|im_start|>system\nTu es l'infirmier de triage du CHSA.<|im_end|>\n"
        f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{ex['response']}<|im_end|>"
    }


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/hf_dataset_triage",
        help="Répertoire contenant le dataset au format Arrow (via load_from_disk)",
    )
    parser.add_argument("--output_dir", type=str, default="models/sft_model_final")
    args, _ = parser.parse_known_args()

    # --- 1. CHARGEMENT ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Configuration 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Initialisation de la config SFT
    sft_config = SFTConfig(
        dataset_text_field="text",
        #max_seq_length=512,
        loss_type="nll",
    )

    # --- 2. DATASETS ---
    print(f"📂 Chargement du dataset depuis {args.data_dir}")
    full_dataset = load_from_disk(args.data_dir)

    # Partitionnement interne (Train/Val)
    print("✂️  Partitionnement du dataset (Train 90% / Val 10%)...")
    split_ds = full_dataset.train_test_split(test_size=0.1)

    # Mapping avec num_proc=1 pour éviter les crashs de multiprocessing
    train_ds = split_ds["train"].map(format_chatml, num_proc=1)
    val_ds = split_ds["test"].map(format_chatml, num_proc=1)

    # --- 3. LORA ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- 4. TRAINING ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
    )

    # --- 5. INITIALISATION DU TRAINER ---
    trainer = SFTTrainer(
        model=args.model_id,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        processing_class=tokenizer,
        peft_config=lora_config,
        config=sft_config,
        quantization_config=bnb_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"✅ Entraînement terminé, modèle sauvegardé dans : {args.output_dir}")

    # --- 6. GÉNÉRATION DES COURBES TRAIN/VAL LOSS ---
    print("📊 Génération des graphiques de performance...")
    history = trainer.state.log_history

    train_loss = [x['loss'] for x in history if 'loss' in x]
    train_steps = [x['step'] for x in history if 'loss' in x]
    val_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
    val_steps = [x['step'] for x in history if 'eval_loss' in x]

    plt.figure(figsize=(12, 6))
    plt.plot(train_steps, train_loss, label='Train Loss (SFT)', color='#1f77b4', linewidth=2)
    if val_loss:
        plt.plot(val_steps, val_loss, label='Validation Loss (SFT)', color='#e31a1c', marker='s', linestyle='--')

    plt.title('Convergence de l\'Alignement SFT - POC CHSA', fontsize=14)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sft_convergence.png', dpi=300)
    print("✅ Graphique de convergence sauvegardé sous 'sft_convergence.png'")


if __name__ == "__main__":
    train()
