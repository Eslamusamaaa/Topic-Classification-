# Topic-Classification-
End-to-end NLP project for classifying AG News articles into four topics. Includes data cleaning, manual N-gram implementation, scikit-learn modeling with TF-IDF + Naive Bayes, and a full Flask web app for live text
# AG News Topic Classifier

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Project Overview

This project implements a news topic classification system using the AG News dataset. It covers the full NLP pipeline: data preprocessing, manual N-gram implementation, scikit-learn-based modeling with TF-IDF and Naive Bayes, evaluation, and a Flask web app for real-time predictions.

Key components:

* Data cleaning and sentence splitting.
* Manual N-gram (Unigram, Bigram, Trigram) with Laplace smoothing.
* scikit-learn pipeline for efficient feature extraction and classification.
* Model saving/loading for reuse.
* Single-file Flask app with embedded HTML/CSS/JS frontend.

The project demonstrates both educational (manual) and practical (scikit-learn) approaches to NLP text classification.

---
## Dataset

The **AG News Classification Dataset** is a benchmark collection of news articles from the AG's corpus, categorized into 4 topics. It is commonly used for evaluating text classification models.

* **Source**: Derived from the AG News corpus (over 1 million news articles from more than 2000 sources).
* **Classes**: 4 balanced topics (30,000 articles per class in training).
* **Format**: Each sample includes `Class Index`, `Title`, and `Description`.
* **Training Data**: 120,000 articles (`train.csv`).
* **Test Data**: 7,600 articles (`test.csv`).
* **Preprocessed Data**: `train_cleaned_with_sentences.csv` (cleaned titles and sentence-split descriptions).
* **Average Description Length**: ~100–200 words.
* **Balanced Distribution**: Equal number of samples per class.
* **Preprocessing Applied**:

  * Lowercase
  * Remove URLs, emails, numbers, punctuation
  * Tokenization
  * Stop words removal
  * Lemmatization
  * Sentence segmentation

### Topics

| Label | Topic    | Example                                        |
| ----- | -------- | ---------------------------------------------- |
| 1     | World    | International politics, conflicts, UN meetings |
| 2     | Sports   | Match results, players, championships          |
| 3     | Business | Stocks, companies, oil prices, earnings        |
| 4     | Sci/Tech | AI, space, gadgets, scientific discoveries     |

> Download original: Kaggle AG News Dataset

---

## Project Structure

```
AG-News-Classifier/
│
├── app.py                          # Full Flask web app (backend + embedded frontend)
│
├── nb_classifier.pkl               # Trained Naive Bayes classifier (scikit-learn)
├── tfidf_vectorizer.pkl            # TF-IDF vectorizer with N-gram (1-3)
│
├── data/
│   ├── train.csv                   # Original training data (120,000 articles)
│   ├── test.csv                    # Test data (7,600 articles)
│   └── train_cleaned_with_sentences.csv  # Preprocessed training data
│
├── notebooks/
│   ├── NLP.ipynb                   # Data preprocessing and cleaning
│   ├── N-GRAM.ipynb                # Manual N-gram implementation
│   └── sklearn_ngram_classification.ipynb  # scikit-learn pipeline and evaluation
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Notebook Descriptions

| Notebook                             | Description                                                                                                                                                                                                                                             |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `NLP.ipynb`                          | Loads `train.csv`, performs preprocessing: lowercase → sentence segmentation → remove numbers/URLs/punctuation → tokenization → stop words removal → lemmatization. Saves result as `train_cleaned_with_sentences.csv`. Includes before/after examples. |
| `N-GRAM.ipynb`                       | Manual implementation of Unigram, Bigram, Trigram counters with Laplace smoothing. Computes probabilities for sample sentences and visualizes comparisons (bar charts).                                                                                 |
| `sklearn_ngram_classification.ipynb` | Builds scikit-learn pipeline: `TfidfVectorizer(ngram_range=(1,3))` + `MultinomialNB`. Trains on preprocessed data, evaluates on `test.csv`, saves models as `.pkl` files.                                                                               |

---

## Web Application (`app.py`)

A **single-file Flask app** with embedded HTML/CSS/JavaScript:

* User inputs raw news text.
* Backend preprocesses input, loads `.pkl` models, predicts topic.
* Output: Predicted topic, confidence %, cleaned text preview.
* Features: Progress bar, loading spinner, error handling.

### How to Run

```bash
pip install -r requirements.txt
python app.py
# Open: http://localhost:5000
```

---

## Requirements (`requirements.txt`)

```
flask==2.3.3
scikit-learn==1.3.0
joblib==1.3.2
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
jupyter==1.0.0
```

---

## How to Use

### Clone the repository

```bash
git clone https://github.com/yourusername/AG-News-Classifier.git
cd AG-News-Classifier
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run notebooks

Open Jupyter and explore preprocessing and model training.

### Run the web app

```bash
python app.py
```

---

## License

MIT License - Free to use, modify, and distribute.
