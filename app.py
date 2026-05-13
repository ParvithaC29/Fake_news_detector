"""
Gradio web app: paste or fetch a news article → Real/Fake verdict +
LIME-highlighted text + model comparison table.

Run: python app.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
import pandas as pd

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

BERT_CKPT    = os.path.join(MODELS_DIR, "bert_base_uncased")
ROBERTA_CKPT = os.path.join(MODELS_DIR, "roberta_base")
LR_MODEL     = os.path.join(MODELS_DIR, "tf-idf_plus_logistic_regression.joblib")
SVM_MODEL    = os.path.join(MODELS_DIR, "tf-idf_plus_linearsvc.joblib")


# ── Lazy-loaded model cache ───────────────────────────────────────────────────
_model_cache: dict = {}


def _load_baseline(path: str):
    if path not in _model_cache:
        import joblib
        _model_cache[path] = joblib.load(path)
    return _model_cache[path]


def _load_transformer(ckpt: str):
    if ckpt not in _model_cache:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model     = AutoModelForSequenceClassification.from_pretrained(ckpt)
        model.eval()
        _model_cache[ckpt] = (tokenizer, model)
    return _model_cache[ckpt]


def _predict_baseline(pipeline, text: str):
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        probs = pipeline.predict_proba([text])[0]
    else:
        import numpy as np
        score = pipeline.decision_function([text])[0]
        prob_real = 1 / (1 + 1e-7 + (1 / max(1e-7, (1 / (1 + float('inf' if score < -500 else (1 / (1 + 2.718281828 ** (-score))))))) ))
        probs_raw = 1 / (1 + 2.718281828 ** (-float(score)))
        probs = [1 - probs_raw, probs_raw]
    label = "Real" if probs[1] > probs[0] else "Fake"
    return label, float(max(probs)), list(probs)


def _predict_transformer(ckpt: str, text: str):
    import torch
    tokenizer, model = _load_transformer(ckpt)
    enc = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    import torch.nn.functional as F
    probs = F.softmax(logits, dim=-1)[0].tolist()
    label = "Real" if probs[1] > probs[0] else "Fake"
    return label, float(max(probs)), probs


# ── URL scraping ──────────────────────────────────────────────────────────────
def scrape_article(url: str) -> str:
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=10, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        return text[:5000]
    except Exception as e:
        return f"Error fetching URL: {e}"


# ── LIME explanation ──────────────────────────────────────────────────────────
def get_lime_html(text: str, model_choice: str) -> str:
    from src.explainability import explain_tfidf, explain_transformer, build_highlighted_html

    try:
        if model_choice == "TF-IDF + Logistic Regression" and os.path.exists(LR_MODEL):
            pipe   = _load_baseline(LR_MODEL)
            pairs  = explain_tfidf(pipe, text)
        elif model_choice == "TF-IDF + LinearSVC" and os.path.exists(SVM_MODEL):
            pipe   = _load_baseline(SVM_MODEL)
            pairs  = explain_tfidf(pipe, text)
        elif model_choice == "BERT-base-uncased" and os.path.isdir(BERT_CKPT):
            pairs  = explain_transformer(BERT_CKPT, text)
        elif model_choice == "RoBERTa-base" and os.path.isdir(ROBERTA_CKPT):
            pairs  = explain_transformer(ROBERTA_CKPT, text)
        else:
            return "<p><em>Model not available for LIME explanation.</em></p>"

        return build_highlighted_html(text, pairs)
    except Exception as e:
        return f"<p><em>LIME error: {e}</em></p>"


# ── Main prediction function ──────────────────────────────────────────────────
def predict(text_input: str, url_input: str, model_choice: str, run_lime: bool):
    # Resolve text source
    text = text_input.strip()
    if not text and url_input.strip():
        text = scrape_article(url_input.strip())

    if not text or len(text) < 30:
        return (
            "⚠️ Please paste an article or provide a valid URL.",
            "",
            gr.update(value=None),
            "",
        )

    # Run prediction
    label, confidence, probs = None, 0.0, [0.5, 0.5]

    if model_choice in ("TF-IDF + Logistic Regression",) and os.path.exists(LR_MODEL):
        label, confidence, probs = _predict_baseline(_load_baseline(LR_MODEL), text)
    elif model_choice == "TF-IDF + LinearSVC" and os.path.exists(SVM_MODEL):
        label, confidence, probs = _predict_baseline(_load_baseline(SVM_MODEL), text)
    elif model_choice == "BERT-base-uncased" and os.path.isdir(BERT_CKPT):
        label, confidence, probs = _predict_transformer(BERT_CKPT, text)
    elif model_choice == "RoBERTa-base" and os.path.isdir(ROBERTA_CKPT):
        label, confidence, probs = _predict_transformer(ROBERTA_CKPT, text)
    else:
        return (
            f"⚠️ Model '{model_choice}' not found. Train it first.",
            "",
            gr.update(value=None),
            "",
        )

    # Verdict display
    icon    = "✅" if label == "Real" else "🚨"
    colour  = "#16a34a" if label == "Real" else "#dc2626"
    verdict = (
        f"<div style='font-size:2rem;font-weight:700;color:{colour};'>"
        f"{icon} {label}</div>"
        f"<div style='font-size:1.1rem;margin-top:4px;'>Confidence: {confidence*100:.1f}%</div>"
        f"<div style='color:#6b7280;font-size:0.9rem;margin-top:4px;'>"
        f"Fake prob: {probs[0]*100:.1f}%  |  Real prob: {probs[1]*100:.1f}%"
        f"</div>"
    )

    # Confidence bar (Gradio Label component dict format)
    bar_data = {"Fake": probs[0], "Real": probs[1]}

    # LIME
    lime_html = ""
    if run_lime:
        lime_html = (
            "<div style='font-family:sans-serif;line-height:1.8;'>"
            + get_lime_html(text[:2000], model_choice)
            + "<hr style='margin-top:12px;'/>"
            "<span style='background:rgba(220,38,38,0.4);padding:1px 6px;border-radius:3px;'>Red</span> = deception signal &nbsp;"
            "<span style='background:rgba(22,163,74,0.4);padding:1px 6px;border-radius:3px;'>Green</span> = credibility signal"
            "</div>"
        )
    else:
        lime_html = "<p><em>Enable 'Show LIME explanation' to see highlighted phrases.</em></p>"

    return verdict, lime_html, bar_data, text[:300] + "..." if len(text) > 300 else text


# ── Comparison table loader ───────────────────────────────────────────────────
def load_comparison_table():
    csv_path = os.path.join(RESULTS_DIR, "comparison.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    # Placeholder
    return pd.DataFrame({
        "Model": ["TF-IDF + Logistic Reg.", "TF-IDF + LinearSVC", "BERT-base-uncased", "RoBERTa-base"],
        "Accuracy": ["Run evaluate_all.py"] * 4,
        "F1 (weighted)": ["—"] * 4,
        "Infer ms/sample": ["—"] * 4,
    })


# ── Gradio UI ─────────────────────────────────────────────────────────────────
AVAILABLE_MODELS = []
if os.path.exists(LR_MODEL):      AVAILABLE_MODELS.append("TF-IDF + Logistic Regression")
if os.path.exists(SVM_MODEL):     AVAILABLE_MODELS.append("TF-IDF + LinearSVC")
if os.path.isdir(BERT_CKPT):      AVAILABLE_MODELS.append("BERT-base-uncased")
if os.path.isdir(ROBERTA_CKPT):   AVAILABLE_MODELS.append("RoBERTa-base")

if not AVAILABLE_MODELS:
    AVAILABLE_MODELS = ["TF-IDF + Logistic Regression", "TF-IDF + LinearSVC",
                        "BERT-base-uncased", "RoBERTa-base"]

CSS = """
#verdict-box { border: 2px solid #e5e7eb; border-radius: 10px; padding: 16px; }
#lime-box    { border: 1px solid #e5e7eb; border-radius: 8px;  padding: 12px; max-height: 400px; overflow-y: auto; }
footer { display: none !important; }
"""

with gr.Blocks(title="Fake News Detector") as demo:
    gr.Markdown("## 🔍 Fake News Detector\n*NLP-powered detection with LIME explainability*")

    with gr.Tabs():
        # ─── Tab 1: Classify ──────────────────────────────────────────────────
        with gr.Tab("Classify Article"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input  = gr.Textbox(label="Paste article text", lines=10, placeholder="Paste the full article body here…")
                    url_input   = gr.Textbox(label="Or enter article URL", placeholder="https://example.com/news-article")
                    model_dd    = gr.Dropdown(choices=AVAILABLE_MODELS, value=AVAILABLE_MODELS[0], label="Model")
                    lime_toggle = gr.Checkbox(label="Show LIME explanation (slower)", value=False)
                    submit_btn  = gr.Button("Analyse", variant="primary")

                with gr.Column(scale=3):
                    verdict_html  = gr.HTML(label="Verdict", elem_id="verdict-box")
                    confidence_bar = gr.Label(label="Confidence Scores")
                    preview_text  = gr.Textbox(label="Text preview (first 300 chars)", interactive=False)

            gr.Markdown("### Phrase-level Explanation (LIME)")
            lime_html_out = gr.HTML(elem_id="lime-box", value="<p><em>Run the classifier first.</em></p>")

            submit_btn.click(
                fn=predict,
                inputs=[text_input, url_input, model_dd, lime_toggle],
                outputs=[verdict_html, lime_html_out, confidence_bar, preview_text],
            )

            gr.Examples(
                examples=[
                    ["Breaking: Scientists discover a cure for all cancers using household items found in your kitchen. Government suppressing this information.", "", "TF-IDF + Logistic Regression", False],
                    ["The Federal Reserve raised interest rates by 25 basis points, citing persistent inflation concerns and a resilient labour market.", "", "TF-IDF + Logistic Regression", False],
                ],
                inputs=[text_input, url_input, model_dd, lime_toggle],
            )

        # ─── Tab 2: Model Comparison ──────────────────────────────────────────
        with gr.Tab("Model Comparison"):
            gr.Markdown("Performance on held-out ISOT test set. Run `python -m src.evaluate_all` to populate.")
            refresh_btn  = gr.Button("Refresh Table")
            comparison_table = gr.Dataframe(value=load_comparison_table, interactive=False)
            chart_img = gr.Image(
                value=os.path.join(RESULTS_DIR, "comparison.png") if os.path.exists(os.path.join(RESULTS_DIR, "comparison.png")) else None,
                label="Accuracy & F1 Bar Chart",
            )
            refresh_btn.click(fn=load_comparison_table, outputs=comparison_table)

        # ─── Tab 3: How it works ──────────────────────────────────────────────
        with gr.Tab("How It Works"):
            gr.Markdown("""
### Architecture

| Stage | What happens |
|-------|-------------|
| **Data** | 44K ISOT articles (Real + Fake), cleaned, split 70/15/15 |
| **Baseline** | TF-IDF unigrams+bigrams → Logistic Regression / LinearSVC |
| **Deep model** | BERT-base-uncased / RoBERTa-base fine-tuned via HuggingFace Trainer |
| **Explainability** | LIME perturbs the input and measures prediction changes per token |

### LIME colour guide
- **Red highlight** — word pushes the model toward *Fake*
- **Green highlight** — word pushes the model toward *Real*
- Opacity reflects the magnitude of each word's contribution

### Training environment
Fine-tuned on Nvidia T4 GPU (Google Colab). CPU inference supported but slower.
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                theme=gr.themes.Soft(), css=CSS)
