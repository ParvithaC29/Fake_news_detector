"""
Data loading and preprocessing for the ISOT Fake News dataset.
Expects the Kaggle ISOT dataset CSVs placed in data/raw/:
  - data/raw/True.csv
  - data/raw/Fake.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def load_isot() -> pd.DataFrame:
    true_path = os.path.join(RAW_DIR, "True.csv")
    fake_path = os.path.join(RAW_DIR, "Fake.csv")

    if not (os.path.exists(true_path) and os.path.exists(fake_path)):
        raise FileNotFoundError(
            "Place True.csv and Fake.csv from the ISOT Kaggle dataset into data/raw/"
        )

    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    df_true["label"] = 1  # Real
    df_fake["label"] = 0  # Fake

    df = pd.concat([df_true, df_fake], ignore_index=True)
    return df


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Remove Reuters-style datelines that leak source identity
    if " (Reuters)" in text:
        text = text.split("-", 1)[-1] if " - " in text else text
    return text


def build_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """Combine title + body into a single text field."""
    df = df.copy()
    title_col = "title" if "title" in df.columns else None
    text_col = "text" if "text" in df.columns else "content"

    if title_col:
        df["combined_text"] = df[title_col].fillna("").apply(clean_text) + " " + df[text_col].fillna("").apply(clean_text)
    else:
        df["combined_text"] = df[text_col].fillna("").apply(clean_text)

    df["combined_text"] = df["combined_text"].str.strip()
    # Drop rows with empty text
    df = df[df["combined_text"].str.len() > 20].reset_index(drop=True)
    return df


def split_dataset(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42):
    """Returns (train_df, val_df, test_df)."""
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=seed, stratify=train_val["label"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def prepare_and_save():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("Loading ISOT dataset...")
    df = load_isot()
    print(f"  Total records: {len(df):,}  |  Real: {df.label.sum():,}  |  Fake: {(df.label==0).sum():,}")

    df = build_combined_text(df)
    train, val, test = split_dataset(df)

    train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    print(f"  Train: {len(train):,}  |  Val: {len(val):,}  |  Test: {len(test):,}")
    print(f"Saved to {PROCESSED_DIR}/")
    return train, val, test


def load_processed():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val   = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test  = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    return train, val, test


if __name__ == "__main__":
    prepare_and_save()
