import argparse
import glob
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


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
        default="data/processed/",
        help="Répertoire contenant les fichiers .jsonl de données",
    )
    parser.add_argument("--output_dir", type=str, default="models/sft_model_final")
    args, _ = parser.parse_known_args()

    # --- 1. CHARGEMENT ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # --- 2. DATASETS ---
    # Charger les fichiers .jsonl : soit un répertoire, soit un fichier unique
    if os.path.isdir(args.data_dir):
        data_files = glob.glob(os.path.join(args.data_dir, "*.jsonl"))
    else:
        data_files = [args.data_dir]

    print(f"📂 Chargement de {len(data_files)} fichiers depuis {args.data_dir}")

    full_dataset = load_dataset("json", data_files=data_files, split="train")

    split_ds = full_dataset.train_test_split(test_size=0.1)

    train_ds = split_ds["train"].map(format_chatml)
    val_ds = split_ds["test"].map(format_chatml)

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

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"✅ Entraînement terminé, modèle sauvegardé dans : {args.output_dir}")


if __name__ == "__main__":
    train()
