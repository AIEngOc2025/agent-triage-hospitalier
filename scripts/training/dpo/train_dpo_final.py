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


def setup_dpo_training(args):
    """
    @definition: Configures and initializes models, tokenizers, and
    arguments for DPO training.
    @args/params:
        - args (argparse.Namespace): Command-line arguments.
    @return: A tuple containing the model, tokenizer, dataset, and training arguments.
    """
    # --- 2. SFT MODEL LOADING ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,  # Stability
    )

    print("📥 Loading SFT model for alignment...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Injecting SFT weights
    model = PeftModel.from_pretrained(
        base_model, args.sft_adapters_path, is_trainable=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 3. DPO DATASET LOADING ---
    print(f"📂 Loading DPO dataset from: {args.dpo_dataset_path}")
    dataset = load_dataset("json", data_files=args.dpo_dataset_path, split="train")

    # --- 4. DPO TRAINING ARGUMENTS (SECURED) ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,  # DPO is memory-intensive
        gradient_accumulation_steps=8,  # Effective batch size of 8
        max_steps=args.max_steps,  # 100-200 steps are often sufficient for DPO
        learning_rate=5e-7,  # Very low LR to avoid "breaking" the SFT
        fp16=True,
        logging_steps=5,
        save_strategy="no",
        remove_unused_columns=False,  # Required for DPOTrainer
        report_to="none",
    )

    return model, tokenizer, dataset, training_args


def main(args):
    """
    @definition: Main function to run the DPO training process.
    @args/params:
        - args (argparse.Namespace): Command-line arguments.
    """
    model, tokenizer, dataset, training_args = setup_dpo_training(args)

    # --- 5. DPO TRAINER ---
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,  # In QLoRA, set to None to save RAM
        args=training_args,
        beta=0.1,  # Standard alignment strength parameter
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_prompt_length=256,
        max_length=512,
    )

    # --- 6. TRAINING EXECUTION ---
    print("⚡ Starting DPO alignment (Week 3)...")
    dpo_trainer.train()

    # --- 7. FINAL SAVE ---
    print(f"💾 Saving final model to {args.output_dir}")
    dpo_trainer.save_model(args.output_dir)
    print("✅ DPO POC COMPLETE! Model ready for Phase 4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Base model ID from Hugging Face.",
    )
    parser.add_argument(
        "--sft_adapters_path",
        type=str,
        required=True,
        help="Path to the SFT adapters.",
    )
    parser.add_argument(
        "--dpo_dataset_path",
        type=str,
        required=True,
        help="Path to the DPO training data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen-chsa-dpo-final",
        help="Directory to save the DPO model.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=100, help="Number of training steps."
    )
    cli_args = parser.parse_args()
    main(cli_args)
