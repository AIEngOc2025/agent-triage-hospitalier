import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def setup_training(args):
    """
    @definition: Configure and initialize models, tokenizers, and training arguments.
    @args/params:
        - args (argparse.Namespace): Command-line arguments.
    @return: A tuple containing the model, tokenizer, dataset, and training arguments.
    """
    # --- 1. ACCELERATOR DETECTION ---
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"🚀 Using device: {device}")

    # --- 2. DATASET LOADING ---
    print(f"📂 Loading dataset from: {args.data_path}")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    if args.max_samples:
        print(f"✂️  Limiting dataset to {args.max_samples} samples for testing.")
        dataset = dataset.select(range(args.max_samples))

    # --- 3. TOKENIZER & MODEL ---
    print(f"🤖 Loading model and tokenizer for: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map={"": device} if device != "cpu" else None,
        trust_remote_code=True,
    )

    # --- 4. LORA CONFIGURATION ---
    print("🛠️  Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Qwen targets
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # --- 5. TRAINING ARGUMENTS ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="no",
        bf16=False,  # MPS does not fully support bf16
        fp16=True if device == "cuda" else False,
        push_to_hub=False,
        report_to="none",
    )

    return model, tokenizer, dataset, training_args


def main(args):
    """
    @definition: Main function to run the SFT training process.
    @args/params:
        - args (argparse.Namespace): Command-line arguments.
    """
    model, tokenizer, dataset, training_args = setup_training(args)

    # --- 6. TRAINER INITIALIZATION ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="instruction",  # Source field
        max_seq_length=512,
    )

    # --- 7. TRAINING EXECUTION ---
    print("⚡ Starting test training...")
    trainer.train()
    print("✅ Test finished! Model is ready for full training on the Cloud.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Training Script")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Base model ID.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/train_sft.jsonl",
        help="Path to training data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/sft_model",
        help="Output directory for the model.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="Number of training steps for local testing.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="Number of samples for local testing.",
    )

    cli_args = parser.parse_args()
    main(cli_args)
