"""Entraînement du classifieur NLP de triage.

Fine-tuner distil-xlm-roberta-base pour classer le niveau de triage.
"""

from __future__ import annotations

import json
import logging

import evaluate
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Configuration
INPUT_FILE = "data/processed/labeled_triage.jsonl"
OUTPUT_DIR = "models/triage_nlp_model"
MODEL_NAME = "distil-xlm-roberta-base"
LABEL_MAP = {"maximale": 0, "modérée": 1, "différée": 2}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset() -> Dataset:
    """
    @definition : Charge et prépare le dataset pour l'entraînement.
    @args/params : None
    @return : Dataset Hugging Face préparé (split train/val/test).
    """
    data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append(
                {
                    "text": item["text"],
                    "label": LABEL_MAP[item["niveau"]],
                }
            )

    # Split: 70% train, 15% val, 15% test
    train_val, test = train_test_split(
        data, test_size=0.15, stratify=[d["label"] for d in data]
    )
    train, val = train_test_split(
        train_val, test_size=0.176, stratify=[d["label"] for d in train_val]
    )

    # Convert to HF Dataset
    return (
        Dataset.from_dict({k: [d[k] for d in train] for k in train[0]}),
        Dataset.from_dict({k: [d[k] for d in val] for k in val[0]}),
        Dataset.from_dict({k: [d[k] for d in test] for k in test[0]}),
    )


def train():
    """
    @definition : Exécute le processus d'entraînement du modèle.
    @args/params : None
    @return : None
    """
    train_ds, val_ds, test_ds = load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        return tokenizer(
            examples["text"], truncation=True, padding=True, max_length=128
        )

    train_ds = train_ds.map(preprocess, batched=True)
    val_ds = val_ds.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_MAP)
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
