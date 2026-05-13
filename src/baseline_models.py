"""
Baseline models: TF-IDF + Logistic Regression and TF-IDF + LinearSVC.
Saves trained pipelines to models/ and prints evaluation metrics.
"""

import os
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def build_tfidf_lr() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=100_000,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1,
        )),
    ])


def build_tfidf_svm() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=100_000,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            min_df=2,
        )),
        ("clf", LinearSVC(C=1.0, max_iter=2000)),
    ])


def evaluate(model, X_test, y_test, name: str, train_time: float) -> dict:
    t0 = time.time()
    preds = model.predict(X_test)
    infer_time_ms = (time.time() - t0) / len(X_test) * 1000

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="weighted")

    print(f"\n{'='*50}")
    print(f"Model : {name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Train time : {train_time:.1f}s")
    print(f"Infer/sample: {infer_time_ms:.3f} ms")
    print(classification_report(y_test, preds, target_names=["Fake", "Real"]))

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "train_time_s": round(train_time, 1),
        "infer_ms_per_sample": round(infer_time_ms, 3),
    }


def train_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[dict]:
    os.makedirs(MODELS_DIR, exist_ok=True)
    results = []

    X_train = train_df["combined_text"].tolist()
    y_train = train_df["label"].tolist()
    X_test  = test_df["combined_text"].tolist()
    y_test  = test_df["label"].tolist()

    for name, pipeline in [
        ("TF-IDF + Logistic Regression", build_tfidf_lr()),
        ("TF-IDF + LinearSVC",           build_tfidf_svm()),
    ]:
        print(f"\nTraining {name}...")
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        metrics = evaluate(pipeline, X_test, y_test, name, train_time)
        results.append(metrics)

        safe_name = name.lower().replace(" ", "_").replace("+", "plus")
        save_path = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
        joblib.dump(pipeline, save_path)
        print(f"Saved -> {save_path}")

    return results


def load_baseline(model_name: str):
    """model_name: 'lr' or 'svm'"""
    filename_map = {
        "lr":  "tf-idf_+_logistic_regression.joblib",
        "svm": "tf-idf_+_linearsvc.joblib",
    }
    # Try both naming conventions
    for fname in [
        filename_map.get(model_name, model_name),
        f"tf-idf_plus_logistic_regression.joblib" if model_name == "lr" else f"tf-idf_plus_linearsvc.joblib",
    ]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(f"No saved model found for '{model_name}' in {MODELS_DIR}")


if __name__ == "__main__":
    from src.data_prep import load_processed
    train_df, val_df, test_df = load_processed()
    results = train_baselines(train_df, test_df)
    print("\nBaseline results summary:")
    for r in results:
        print(r)
