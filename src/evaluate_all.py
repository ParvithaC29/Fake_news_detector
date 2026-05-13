"""
Aggregate evaluation: loads all trained models, runs them on the held-out
test set, and produces a comparison table + bar chart saved as results/comparison.png.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_all_evaluations(test_df: pd.DataFrame) -> pd.DataFrame:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    # ── Baseline models ──────────────────────────────────────────────────────
    import joblib
    from sklearn.metrics import accuracy_score, f1_score
    import time

    for fname, label in [
        ("tf-idf_plus_logistic_regression.joblib", "TF-IDF + Logistic Reg."),
        ("tf-idf_plus_linearsvc.joblib",           "TF-IDF + LinearSVC"),
    ]:
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            print(f"  Skipping {label}: file not found ({fname})")
            continue
        pipe = joblib.load(path)
        X = test_df["combined_text"].tolist()
        y = test_df["label"].tolist()
        t0 = time.time()
        preds = pipe.predict(X)
        infer_ms = (time.time() - t0) / len(X) * 1000
        rows.append({
            "Model": label,
            "Accuracy": round(accuracy_score(y, preds), 4),
            "F1 (weighted)": round(f1_score(y, preds, average="weighted"), 4),
            "Infer ms/sample": round(infer_ms, 3),
        })

    # ── Transformer models ────────────────────────────────────────────────────
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from sklearn.metrics import accuracy_score, f1_score

        for subdir, label in [
            ("bert_base_uncased", "BERT-base-uncased"),
            ("roberta_base",      "RoBERTa-base"),
        ]:
            ckpt = os.path.join(MODELS_DIR, subdir)
            if not os.path.isdir(ckpt):
                print(f"  Skipping {label}: checkpoint not found ({subdir})")
                continue

            tokenizer = AutoTokenizer.from_pretrained(ckpt)
            model     = AutoModelForSequenceClassification.from_pretrained(ckpt)
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            texts  = test_df["combined_text"].tolist()
            y_true = test_df["label"].tolist()
            preds  = []

            t0 = time.time()
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                enc = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    logits = model(**enc).logits
                preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())

            infer_ms = (time.time() - t0) / len(texts) * 1000
            rows.append({
                "Model": label,
                "Accuracy": round(accuracy_score(y_true, preds), 4),
                "F1 (weighted)": round(f1_score(y_true, preds, average="weighted"), 4),
                "Infer ms/sample": round(infer_ms, 3),
            })
    except ImportError:
        print("transformers not installed; skipping transformer evaluation")

    results_df = pd.DataFrame(rows)
    print("\n" + results_df.to_string(index=False))

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV -> {csv_path}")

    # Bar chart
    if not results_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(results_df))
        width = 0.35
        bars1 = ax.bar([xi - width/2 for xi in x], results_df["Accuracy"],   width, label="Accuracy",      color="#3b82f6")
        bars2 = ax.bar([xi + width/2 for xi in x], results_df["F1 (weighted)"], width, label="F1 (weighted)", color="#10b981")
        ax.set_xticks(list(x))
        ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
        ax.set_ylim(0.8, 1.01)
        ax.set_ylabel("Score")
        ax.set_title("Fake News Detection — Model Comparison")
        ax.legend()
        for bar in list(bars1) + list(bars2):
            ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        plt.tight_layout()
        chart_path = os.path.join(RESULTS_DIR, "comparison.png")
        plt.savefig(chart_path, dpi=150)
        print(f"Saved chart -> {chart_path}")
        plt.close()

    return results_df


if __name__ == "__main__":
    from src.data_prep import load_processed
    _, _, test_df = load_processed()
    run_all_evaluations(test_df)
