"""
One-shot script: prepare data → train TF-IDF baselines → evaluate.
Run this locally (fast, no GPU needed).

Usage:
  python train_baselines.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.data_prep import prepare_and_save, load_processed
from src.baseline_models import train_baselines
from src.evaluate_all import run_all_evaluations

if __name__ == "__main__":
    # Step 1 — prepare data (skip if processed files already exist)
    processed_train = os.path.join("data", "processed", "train.csv")
    if os.path.exists(processed_train):
        print("Processed data found — skipping data prep step.")
        train_df, val_df, test_df = load_processed()
    else:
        train_df, val_df, test_df = prepare_and_save()

    # Step 2 — train baselines
    results = train_baselines(train_df, test_df)

    # Step 3 — compare
    comparison = run_all_evaluations(test_df)

    print("\nDone! Launch the app with:  python app.py")
