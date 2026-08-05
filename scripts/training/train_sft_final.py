import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def format_chatml(ex):
    # Format ChatML strict pour Qwen
    # L'instruction est utilisée comme système ou utilisateur selon le cas.
    # Ici, on suit la structure du notebook Kaggle pour la cohérence.
    return {
        "text": f"<|im_start|>system\nTu es l'infirmier de triage du CHSA.<|im_end|>\n"
        f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{ex['response']}<|im_end|>"
    }


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/processed/train_sft_balanced_50_50.jsonl",
    )
    parser.add_argument(
        "--val_path", type=str, default="data/processed/val_sft_split.jsonl"
    )
    parser.add_argument("--output_dir", type=str, default="models/sft_model_final")
    args = parser.parse_args()

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
    train_ds = load_dataset("json", data_files=args.train_path, split="train").map(
        format_chatml
    )
    val_ds = load_dataset("json", data_files=args.val_path, split="train").map(
        format_chatml
    )

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
