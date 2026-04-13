# Fake News Detection: ML + DL Approaches

An academic project for detecting fake news using two complementary strategies:

- **Machine Learning approach** with a **Streamlit** web interface
- **Deep Learning approach** with **LSTM (PyTorch)** and **BERT (Transformers)** notebooks

The project uses the popular Fake/True news dataset and compares traditional NLP + ML with neural models.

## Project Overview

Fake news detection is a text classification problem where each news article is predicted as:

- `FAKE`
- `REAL`

This repository explores three pipelines:

1. **ML pipeline**: text preprocessing + TF-IDF + trained classifier + Streamlit app
2. **LSTM pipeline**: tokenization + vocabulary building + sequence modeling in PyTorch
3. **BERT pipeline**: transformer fine-tuning using Hugging Face Transformers

## Repository Structure

```text
Fake News Detection/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── ML approach/
│   ├── app.py
│   ├── fake-news-ml-approach.ipynb
│   ├── requirements.txt
│   └── models/
│       ├── fake_news_model.pkl
│       └── tfidf_vectorizer.pkl
└── DL approach/
    ├── RNNs (LSTM)/
    │   ├── lstm_notebook.ipynb
    │   └── model/
    │       └── lstm_fake_news_model.pth
    └── Transformers (BERT)/
        ├── Bert_notebook.ipynb
```

## Dataset

The project uses two CSV files:

- `Fake.csv`
- `True.csv`

Public dataset source:

- https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Typical columns include title/text metadata used for binary classification.

## 1) Machine Learning Approach (Streamlit)

### What it does

- Cleans text (regex, tokenization, stopword removal, lemmatization)
- Converts text to TF-IDF vectors
- Loads a trained ML classifier (`.pkl`)
- Predicts whether input news text is fake or real
- Provides an interactive Streamlit UI

### Setup

From the project root:

```bash
cd "ML approach"
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet
```

### Run the app

```bash
cd "ML approach"
streamlit run app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

ML interface link :

- fake-news-detection-adjghrijzvsp6xusayeams.streamlit.app

### Important note

In `ML approach/app.py`, model paths are currently absolute and local to one machine. For portability on GitHub/other machines, use relative paths such as:

- `models/fake_news_model.pkl`
- `models/tfidf_vectorizer.pkl`

## 2) Deep Learning Approach - LSTM (PyTorch)

Notebook: `DL approach/RNNs (LSTM)/lstm_notebook.ipynb`

### Pipeline summary

- Load and merge fake/real datasets
- Preprocess and tokenize text with NLTK
- Build vocabulary and convert text to padded sequences
- Train/validate an LSTM-based classifier in PyTorch
- Evaluate with common metrics (accuracy, precision, recall, F1)

Saved model:

- `DL approach/RNNs (LSTM)/model/lstm_fake_news_model.pth`

## 3) Deep Learning Approach - BERT (Transformers)

Notebook: `DL approach/Transformers (BERT)/Bert_notebook.ipynb`

### Pipeline summary

- Merge and label dataset
- Combine title + body text as model input
- Tokenize with `BertTokenizerFast`
- Fine-tune `bert-base-uncased` for sequence classification
- Evaluate and save trained artifacts

Saved artifacts:

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`

## Suggested Environment

- Python 3.9+
- Recommended: GPU for BERT/LSTM training
- Jupyter Notebook or VS Code Notebook support for DL notebooks

## Reproducibility Tips

- Keep a fixed random seed for splitting/training
- Use the same preprocessing in training and inference
- Document versions of Python and key libraries

## Future Improvements

- Add a unified evaluation table comparing ML vs LSTM vs BERT
- Add confusion matrix plots and ROC-AUC for all models
- Build one unified app with model selection (ML/LSTM/BERT)
- Add Docker support for one-command setup

## Author

Developed as an academic project on fake news detection using both classical ML and modern DL techniques.

---

If you find this project useful, consider starring the repository.
