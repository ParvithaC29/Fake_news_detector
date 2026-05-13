"""
Google Colab training script for BERT-base-uncased and RoBERTa-base.

HOW TO USE IN COLAB:
--------------------
Option A — Upload processed CSVs (fastest):
  1. Upload train.csv, val.csv, test.csv from your local data/processed/ folder
     using the Colab file panel (left sidebar -> Files -> Upload)
  2. Run this script as-is.

Option B — Upload raw ISOT files (self-contained):
  1. Upload True.csv and Fake.csv from your local data/raw/ folder
  2. Set USE_RAW = True below
  3. Run this script.

After training:
  - Two zip files will be created: bert_base_uncased.zip and roberta_base.zip
  - Download them from the Colab file panel
  - Unzip into your local models/ folder
"""

# ── 0. Install packages ───────────────────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "-q", "transformers", "datasets", "accelerate", "scikit-learn"], check=True)

import os
import re
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback,
)

# ── 1. Configuration ──────────────────────────────────────────────────────────
USE_RAW = False   # Set True if you uploaded True.csv / Fake.csv instead of the processed CSVs

MAX_LEN    = 256
EPOCHS     = 3
BATCH_SIZE = 16   # reduce to 8 if you get CUDA out-of-memory

# ── 2. Data loading ───────────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if " (Reuters)" in text and " - " in text:
        text = text.split(" - ", 1)[-1]
    return text

def load_from_raw():
    print("Loading raw ISOT files...")
    df_true = pd.read_csv("True.csv")
    df_fake = pd.read_csv("Fake.csv")
    df_true["label"] = 1
    df_fake["label"] = 0
    df = pd.concat([df_true, df_fake], ignore_index=True)

    title_col = "title" if "title" in df.columns else None
    text_col  = "text"  if "text"  in df.columns else "content"

    if title_col:
        df["combined_text"] = df[title_col].fillna("").apply(clean_text) + " " + df[text_col].fillna("").apply(clean_text)
    else:
        df["combined_text"] = df[text_col].fillna("").apply(clean_text)

    df = df[df["combined_text"].str.strip().str.len() > 20].reset_index(drop=True)
    print(f"  Total: {len(df):,}  Real: {df.label.sum():,}  Fake: {(df.label==0).sum():,}")

    train_val, test = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])
    train, val      = train_test_split(train_val, test_size=0.15/0.85, random_state=42, stratify=train_val["label"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def load_from_processed():
    print("Loading processed CSVs...")
    train = pd.read_csv("train.csv")
    val   = pd.read_csv("val.csv")
    test  = pd.read_csv("test.csv")
    return train, val, test

if USE_RAW:
    train_df, val_df, test_df = load_from_raw()
else:
    train_df, val_df, test_df = load_from_processed()

print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
print(f"GPU available: {torch.cuda.is_available()}")

# ── 3. Tokenisation helper ────────────────────────────────────────────────────
def tokenize_df(tokenizer, df):
    ds = Dataset.from_pandas(
        df[["combined_text", "label"]].rename(columns={"combined_text": "text"})
    )
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=MAX_LEN)
    ds = ds.map(tok, batched=True, remove_columns=["text"])
    return ds.rename_column("label", "labels")

# ── 4. Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":     accuracy_score(labels, preds),
        "f1_weighted":  f1_score(labels, preds, average="weighted"),
    }

# ── 5. Training function ──────────────────────────────────────────────────────
def train_model(model_name, output_dir):
    print(f"\n{'='*60}")
    print(f"Fine-tuning: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_ds = tokenize_df(tokenizer, train_df)
    val_ds   = tokenize_df(tokenizer, val_df)
    test_ds  = tokenize_df(tokenizer, test_df)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=200,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Save best checkpoint
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate on test set
    print(f"\n--- Test set results for {model_name} ---")
    results = trainer.evaluate(test_ds)
    print(results)

    # Run full classification report
    preds_out = trainer.predict(test_ds)
    preds = np.argmax(preds_out.predictions, axis=-1)
    print(classification_report(test_df["label"].tolist(), preds, target_names=["Fake", "Real"]))

    return output_dir

# ── 6. Train both models ──────────────────────────────────────────────────────
bert_dir    = train_model("bert-base-uncased", "bert_base_uncased")
roberta_dir = train_model("roberta-base",      "roberta_base")

# ── 7. Zip for download ───────────────────────────────────────────────────────
print("\nPackaging checkpoints for download...")
shutil.make_archive("bert_base_uncased", "zip", ".", "bert_base_uncased")
shutil.make_archive("roberta_base",      "zip", ".", "roberta_base")
print("Done! Download bert_base_uncased.zip and roberta_base.zip from the Files panel.")
print("Unzip both into your local models/ folder.")
