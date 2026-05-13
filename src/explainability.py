"""
LIME-based explainability for both baseline (TF-IDF) and transformer models.
Returns token-level attribution scores for highlighting in the Gradio UI.
"""

import numpy as np
import re
from lime.lime_text import LimeTextExplainer

LABEL_NAMES = ["Fake", "Real"]
NUM_FEATURES = 20   # top tokens to highlight
NUM_SAMPLES  = 500  # LIME perturbation samples (higher = slower but more stable)


def _make_tfidf_predictor(pipeline):
    """Wrap a sklearn Pipeline to return a [n_samples, 2] probability array."""
    def predict_proba(texts):
        clf = pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            return pipeline.predict_proba(texts)
        # LinearSVC: use decision function and convert to pseudo-probabilities via sigmoid
        scores = pipeline.decision_function(texts)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        probs = 1 / (1 + np.exp(-scores))
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs
    return predict_proba


def _make_transformer_predictor(checkpoint_dir: str):
    """Lazy-load transformer and return a batched predictor."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()

    def predict_proba(texts):
        results = []
        for text in texts:
            enc = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0].numpy()
            results.append(probs)
        return np.array(results)

    return predict_proba


def explain_tfidf(pipeline, text: str, label_idx: int = None) -> list[tuple[str, float]]:
    """
    Returns a list of (word, importance) tuples sorted by absolute importance.
    label_idx: 0=Fake, 1=Real. If None, uses the predicted class.
    """
    explainer  = LimeTextExplainer(class_names=LABEL_NAMES)
    predictor  = _make_tfidf_predictor(pipeline)
    predicted  = int(np.argmax(predictor([text])[0]))
    target_idx = label_idx if label_idx is not None else predicted

    exp = explainer.explain_instance(
        text,
        predictor,
        num_features=NUM_FEATURES,
        num_samples=NUM_SAMPLES,
        labels=[target_idx],
    )
    return exp.as_list(label=target_idx)


def explain_transformer(checkpoint_dir: str, text: str, label_idx: int = None) -> list[tuple[str, float]]:
    """Same interface as explain_tfidf but uses a transformer model."""
    explainer = LimeTextExplainer(class_names=LABEL_NAMES)
    predictor = _make_transformer_predictor(checkpoint_dir)
    predicted = int(np.argmax(predictor([text])[0]))
    target_idx = label_idx if label_idx is not None else predicted

    exp = explainer.explain_instance(
        text,
        predictor,
        num_features=NUM_FEATURES,
        num_samples=NUM_SAMPLES,
        labels=[target_idx],
    )
    return exp.as_list(label=target_idx)


def build_highlighted_html(text: str, lime_pairs: list[tuple[str, float]]) -> str:
    """
    Wrap LIME-flagged words in the original text with colour spans.
    Red  (negative weight) = pushes toward Fake
    Green (positive weight) = pushes toward Real
    """
    word_scores = {word: score for word, score in lime_pairs}

    # Normalize opacity relative to the max absolute score
    max_abs = max(abs(s) for s in word_scores.values()) if word_scores else 1.0

    tokens = re.split(r"(\s+)", text)
    html_parts = []
    for token in tokens:
        clean = re.sub(r"[^\w]", "", token).lower()
        if clean in word_scores:
            score = word_scores[clean]
            opacity = min(0.9, abs(score) / max_abs)
            if score < 0:
                # negative for predicted class → deceptive signal
                colour = f"rgba(220, 38, 38, {opacity:.2f})"
                title  = f"Deception signal: {score:.3f}"
            else:
                colour = f"rgba(22, 163, 74, {opacity:.2f})"
                title  = f"Credibility signal: {score:.3f}"
            html_parts.append(
                f'<mark style="background:{colour};padding:1px 3px;border-radius:3px;" '
                f'title="{title}">{token}</mark>'
            )
        else:
            html_parts.append(token)

    return "".join(html_parts)
