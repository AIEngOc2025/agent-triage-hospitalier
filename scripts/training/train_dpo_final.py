# --- 1. INSTALLATION ---

import argparse

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer


def run_dpo_training(model_id, sft_adapters_path, dpo_dataset_path, output_dir):
    """
    @definition : Runs the Direct Preference Optimization (DPO) training
                  process to align the model.
    @args/params : model_id (str), sft_adapters_path (str),
                   dpo_dataset_path (str), output_dir (str).
    @return : None. Saves the trained model to output_dir.
    """
    # --- 2. CONFIGURATION ---
    # --- 3. CHARGEMENT DU MODÈLE SFT ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,  # Stabilité
    )

    print("📥 Chargement du modèle SFT pour alignement...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # On injecte les poids du SFT (tes 4 époques)
    model = PeftModel.from_pretrained(base_model, sft_adapters_path, is_trainable=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 4. CHARGEMENT DU DATASET DPO ---
    dataset = load_dataset("json", data_files=dpo_dataset_path, split="train")

    # --- 5. ARGUMENTS D'ENTRAÎNEMENT DPO (SÉCURISÉS) ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # DPO est lourd, on reste à 1
        gradient_accumulation_steps=8,  # Batch effectif de 8
        max_steps=100,  # 100 à 200 étapes suffisent pour le DPO
        learning_rate=5e-7,  # LR très faible pour ne pas "casser" le SFT
        fp16=True,
        logging_steps=5,
        save_strategy="no",
        remove_unused_columns=False,  # Requis pour DPOTrainer
        report_to="none",
    )

    # --- 6. LE TRAINER DPO ---
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,  # En QLoRA, laisser à None pour gagner de la RAM
        args=training_args,
        beta=0.1,  # Paramètre de force de l'alignement (standard)
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_prompt_length=256,
        max_length=512,
    )

    print("⚡ Lancement de l'alignement DPO (Semaine 3)...")
    dpo_trainer.train()

    # --- 7. SAUVEGARDE FINALE ---
    dpo_trainer.save_model(output_dir)
    print("✅ POC DPO TERMINÉ ! Modèle prêt pour la Phase 4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-1.7B-Base",
        help="Base model ID from Hugging Face.",
    )
    parser.add_argument(
        "--sft_adapters_path",
        type=str,
        default="/kaggle/working/chsa_model_final",
        help="Path to the SFT adapters.",
    )
    parser.add_argument(
        "--dpo_dataset_path",
        type=str,
        default="/kaggle/input/ton-dataset-dpo/Mpaga_Christophe_1_Dataset_Train_DPO_052026.jsonl",
        help="Path to the DPO training data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen-chsa-dpo-final",
        help="Directory to save the DPO model.",
    )
    args = parser.parse_args()
    run_dpo_training(
        args.model_id, args.sft_adapters_path, args.dpo_dataset_path, args.output_dir
    )
