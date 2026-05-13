"""
Fine-tuning BERT-base-uncased and RoBERTa-base on the fake news dataset
using the HuggingFace Trainer API.

NOTE: Training on CPU is extremely slow (days for full dataset).
Recommended: run this on Google Colab (GPU) or Kaggle (GPU T4).
Upload the saved checkpoint folders back to models/ for use in the app.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import evaluate as hf_evaluate

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MAX_LENGTH = 256  # 512 gives marginally better accuracy but 2× slower


def tokenize_dataset(tokenizer, df: pd.DataFrame, max_length: int = MAX_LENGTH) -> Dataset:
    ds = Dataset.from_pandas(df[["combined_text", "label"]].rename(columns={"combined_text": "text"}))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,          # DataCollatorWithPadding handles dynamic padding
            max_length=max_length,
        )

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_weighted": f1}


def fine_tune(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    output_subdir: str = None,
) -> str:
    """Fine-tune model_name and return the checkpoint directory path."""

    safe_name = model_name.replace("/", "_").replace("-", "_")
    output_dir = os.path.join(MODELS_DIR, output_subdir or safe_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fine-tuning: {model_name}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU (slow!)'}")
    print(f"Epochs: {num_epochs}  |  Batch: {batch_size}  |  LR: {learning_rate}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_ds = tokenize_dataset(tokenizer, train_df)
    val_ds   = tokenize_dataset(tokenizer, val_df)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=100,
        report_to="none",           # disable wandb/tensorboard
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"\nTraining complete in {train_time/60:.1f} minutes")

    # Save final model + tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    return output_dir


def evaluate_transformer(checkpoint_dir: str, test_df: pd.DataFrame, model_label: str) -> dict:
    print(f"\nEvaluating {model_label} from {checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    texts  = test_df["combined_text"].tolist()
    labels = test_df["label"].tolist()
    preds  = []

    t0 = time.time()
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())

    infer_time_ms = (time.time() - t0) / len(texts) * 1000
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted")

    print(classification_report(labels, preds, target_names=["Fake", "Real"]))
    return {
        "model": model_label,
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "infer_ms_per_sample": round(infer_time_ms, 3),
    }


def predict_single(text: str, checkpoint_dir: str):
    """Return (label_str, confidence) for a single article."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()

    enc = tokenizer(text, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    label = "Real" if probs[1] > probs[0] else "Fake"
    confidence = float(max(probs))
    return label, confidence, probs.tolist()


if __name__ == "__main__":
    from src.data_prep import load_processed

    train_df, val_df, test_df = load_processed()

    # Fine-tune BERT
    bert_dir = fine_tune(
        "bert-base-uncased",
        train_df, val_df,
        num_epochs=3,
        batch_size=16,
        output_subdir="bert_base_uncased",
    )
    bert_metrics = evaluate_transformer(bert_dir, test_df, "BERT-base-uncased")

    # Fine-tune RoBERTa
    roberta_dir = fine_tune(
        "roberta-base",
        train_df, val_df,
        num_epochs=3,
        batch_size=16,
        output_subdir="roberta_base",
    )
    roberta_metrics = evaluate_transformer(roberta_dir, test_df, "RoBERTa-base")

    print("\nTransformer results:")
    print(bert_metrics)
    print(roberta_metrics)
