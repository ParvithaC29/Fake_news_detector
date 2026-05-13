# 📰 Fake News Detector

A machine learning-based web application that detects whether a news article is **Real** or **Fake** based on its textual content. This project uses Natural Language Processing (NLP) techniques and machine learning algorithms to analyze news text and classify its authenticity.

## 🚀 Features

- Detects fake and real news articles
- User-friendly web interface
- Text preprocessing using NLP techniques
- Machine learning-based prediction
- Fast and accurate classification
- Simple deployment-ready structure

## 🛠️ Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- NLTK
- HTML
- CSS
- JavaScript

## 📂 Project Structure

```bash
fake_news_detector/
│
├── static/              # CSS, JavaScript, images
├── templates/           # HTML templates
├── model/               # Saved ML model files
├── app.py               # Main Flask application
├── train_model.py       # Model training script
├── requirements.txt     # Dependencies
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fake_news_detector.git
cd fake_news_detector
```

### 2. Create virtual environment

```bash
python -m venv venv
```

Activate environment:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

```bash
python app.py
```

Open browser and visit:

```bash
http://127.0.0.1:5000
```

## 🧠 How It Works

1. User enters a news article or headline
2. Text is cleaned and preprocessed
3. NLP transforms the text into machine-readable features
4. Trained ML model predicts whether news is fake or real
5. Result is displayed instantly

## 📊 Machine Learning Workflow

- Data Collection
- Data Cleaning
- Text Preprocessing
- Feature Extraction (TF-IDF / Count Vectorizer)
- Model Training
- Model Evaluation
- Deployment using Flask

## 🎯 Future Improvements

- Deep learning-based fake news detection
- News URL verification
- Live news API integration
- Multi-language support
- Confidence score visualization

## 👩‍💻 Author

**Parvitha C**

---

⭐ If you found this project useful, consider giving it a star!
