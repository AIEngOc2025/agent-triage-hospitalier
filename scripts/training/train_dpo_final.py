# --- 1. INSTALLATION ---

import argparse
import gc

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


def format_dpo_dataset(ex):
    """
    @definition: Formats a single example from the dataset into the ChatML
                 structure required for DPO training.
    @args/params: ex (dict): A dictionary containing 'instruction', 'chosen',
                             and 'rejected' keys.
    @return: A dictionary with 'prompt', 'chosen', and 'rejected' keys formatted
             in ChatML.
    """
    ex["prompt"] = (
        f"<|im_start|>system\nTu es l'infirmier de triage du CHSA.<|im_end|>\n<|im_start|>user\n{ex['instruction']}<|im_end|>"
    )
    ex["chosen"] = f"<|im_start|>assistant\n{ex['chosen']}<|im_end|>"
    ex["rejected"] = f"<|im_start|>assistant\n{ex['rejected']}<|im_end|>"
    return ex


def run_dpo_training(model_id, sft_adapters_path, dpo_dataset_path, output_dir):
    """
    @definition : Runs the Direct Preference Optimization (DPO) training
                  process to align the model.
    @args/params : model_id (str), sft_adapters_path (str),
                   dpo_dataset_path (str), output_dir (str).
    @return : None. Saves the trained model to output_dir.
    """
    # --- 2. CONFIGURATION ---
    # Configuration de la quantification pour un usage mémoire optimisé (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Utiliser float16 pour la compatibilité GPU Kaggle
    )

    print("📥 Chargement du modèle SFT pour alignement...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # On injecte les poids du SFT (fine-tuning supervisé)
    model = PeftModel.from_pretrained(base_model, sft_adapters_path, is_trainable=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    # Crucial pour les modèles Qwen pour éviter les problèmes de génération
    tokenizer.padding_side = "right"

    # --- 4. CHARGEMENT DU DATASET DPO ---
    print(f"📂 Chargement et formatage du dataset DPO depuis {dpo_dataset_path}...")
    dataset = load_dataset("json", data_files=dpo_dataset_path, split="train")
    # Formater le dataset pour qu'il corresponde au format attendu par DPOTrainer
    # avec les colonnes 'prompt', 'chosen', 'rejected'
    dataset = dataset.map(format_dpo_dataset)

    # --- 5. ARGUMENTS D'ENTRAÎNEMENT DPO (SÉCURISÉS) ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # DPO est lourd, on reste à 1
        gradient_accumulation_steps=8,  # Batch effectif de 8
        max_steps=100,  # 100 à 200 étapes suffisent pour le DPO
        learning_rate=5e-7,  # LR très faible pour ne pas "casser" le SFT
        fp16=True,  # Activer la précision mixte pour l'entraînement sur GPU
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

    print("💾 Sauvegarde du modèle DPO final...")
    # --- 7. SAUVEGARDE FINALE (AVANT NETTOYAGE) ---
    # Il est crucial de sauvegarder AVANT de supprimer l'objet trainer.
    dpo_trainer.save_model(output_dir)

    # --- 8. NETTOYAGE MÉMOIRE ---
    # Libérer la mémoire GPU après la sauvegarde pour terminer proprement.
    del dpo_trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ POC DPO TERMINÉ ! Modèle prêt pour la Phase 4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training Script")
    # --- Arguments pour Kaggle / Entraînement sur le Cloud ---
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-1.7B-Base",
        help="ID du modèle de base sur le Hub Hugging Face (ex: 'Qwen/Qwen3-1.7B-Base').",
    )
    parser.add_argument(
        "--sft_adapters_path",
        type=str,
        required=True,
        help="Chemin vers les adaptateurs SFT (ex: '/kaggle/input/mes-adaptateurs-sft/') ou ID sur le Hub.",
    )
    parser.add_argument(
        "--dpo_dataset_path",
        type=str,
        required=True,
        help="Chemin vers le fichier de données DPO (ex: '/kaggle/input/mon-dataset/dpo.jsonl').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen-chsa-dpo-final",
        help="Répertoire de sortie pour sauvegarder les adaptateurs DPO (ex: '/kaggle/working/dpo_final').",
    )
    args = parser.parse_args()

    # Note: Pour utiliser des modèles/datasets privés sur Kaggle,
    # ajoutez votre token HF dans les "Secrets" du notebook.
    # from huggingface_hub import login
    # from kaggle_secrets import UserSecretsClient
    # login(token=UserSecretsClient().get_secret("HF_TOKEN"))

    run_dpo_training(
        args.model_id, args.sft_adapters_path, args.dpo_dataset_path, args.output_dir
    )
