# Fake News Detection System

An end-to-end NLP project that classifies news articles as **real** or **fake**. It starts with text preprocessing, trains a classifier on the Kaggle fake news dataset, and serves predictions through a **FastAPI** backend and a **Streamlit** web app, with every prediction stored in a **PostgreSQL** database.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Text Cleaning Steps](#text-cleaning-steps)
- [Dataset](#dataset)
- [Model](#model)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Database Schema](#database-schema)
- [What I Learned](#what-i-learned)
- [Future Improvements](#future-improvements)

---

## Project Overview

The project has two parts:

1. **Text preprocessing:** building a seven-step text cleaning pipeline, practised on sample data from the Kaggle notebook [Getting Started with Text Preprocessing](https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing).
2. **Fake news detection system:** applying that preprocessing to a fake news data, training a classifier, and deploying it as a working application.

## Text Cleaning Steps

I implemented seven cleaning steps, chained into a single preprocessing pipeline that is used for both training and live predictions:

| # | Step | Why it matters |
|---|------|----------------|
| 1 | **HTML tag removal** | Strips leftover markup such as `<b>` or `<p>` that carries no meaning |
| 2 | **URL removal** | Removes links that would otherwise become meaningless tokens |
| 3 | **Source tag removal** | Removes agency tags like `WASHINGTON (Reuters) -`, which can leak the label (real articles often carry them) and cause the model to learn the tag instead of the content |
| 4 | **Lower casing** | Treats "Fake", "FAKE" and "fake" as the same word |
| 5 | **Punctuation removal** | Strips symbols that add noise but little meaning for classification |
| 6 | **Stop word removal** | Drops very common words ("the", "is", "and") that carry little signal |
| 7 | **Lemmatization** | Reduces words to their base form ("bills" becomes "bill") so related forms share one feature |

```python
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def remove_html(text):
    return re.sub(r"<.*?>", " ", text)

def remove_urls(text):
    return re.sub(r"(https?://\S+|www\.\S+)", " ", text)

def remove_source_tag(text):
    return re.sub(r"^.{0,60}?\((reuters|ap|afp)\)\s*[-–—]?\s*", "", text, flags=re.I)

def to_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def remove_stopwords(text):
    return " ".join(w for w in text.split() if w not in STOP_WORDS)

def lemmatize(text):
    return " ".join(lemmatizer.lemmatize(w) for w in text.split())

def clean_text(text):
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_source_tag(text)
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return re.sub(r"\s+", " ", text).strip()
```


## Dataset

- **Source:** Fake News dataset (in data folder)
- **Classes:** `fake` and `real`
- **Columns used:** `<e.g. title, text, label>`

## Model

The classifier is a **Multi-Layer Perceptron (neural network)** from scikit-learn, `sklearn.neural_network.MLPClassifier`. The cleaned text is converted into numerical features, and the MLP learns to separate real from fake articles.


## System Architecture

```
User  ->  Streamlit app  ->  FastAPI backend  ->  Text cleaning + Model  ->  Prediction
                                   |
                                   v
                             PostgreSQL (stores article text, prediction, timestamp)
```

- **Streamlit** provides the user interface for pasting in an article.
- **FastAPI** exposes the prediction endpoint, cleans the text and runs the model.
- **PostgreSQL** keeps a record of each submitted article and its prediction.

## Project Structure

```
.
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app and routes
│   ├── predictor.py         # Loads the model/vectorizer and makes predictions
│   └── schemas.py           # Request and response schemas
├── database/                # PostgreSQL connection and models
├── frontend/
│   └── app.py               # Streamlit app
├── src/
│   ├── data_prep.py         # Loads and prepares the dataset
│   ├── preprocessor.py      # Text cleaning pipeline
│   ├── train_test_split.py  # Splits the data into train and test sets
│   ├── lg_train.py          # Trains the Logistic Regression model
│   ├── mlp_train.py         # Trains the MLPClassifier model
│   ├── utils.py             # Helper functions
│   └── test.py              # Tests
├── nlp.ipynb                # Exploration and experiments
└── README.md

```


## Database Schema

| Column | Type | Description |
|--------|------|-------------|
| `News_code` | SERIAL, primary key | Unique record ID |
| `Station     |  Str | name of TV |      
| `article_text` | TEXT | The submitted article |
| `prediction` | VARCHAR | `fake` or `real` |

## What I Learned

- How text cleaning choices (HTML, URL and source tag removal, lower casing, punctuation, stop words, lemmatization) affect the text a model sees
- How source tags like `(Reuters)` can leak the label and inflate accuracy if left in
- Why the order of preprocessing steps matters
- How to turn text into numerical features and train a neural network classifier with scikit-learn's `MLPClassifier`
- Why the fitted vectorizer must be saved and reused so training and live predictions match
- How to expose a model through a REST API with FastAPI
- How to build an interactive front end with Streamlit
- How to connect an application to PostgreSQL and store predictions

## Future Improvements

- [ ] Add more cleaning steps: spelling correction, emoji and emoticon handling, rare/frequent word removal
- [ ] Use part-of-speech tagging so lemmatization also handles verbs
- [ ] Compare the MLP against other models (Naive Bayes, SVM. etc)
- [ ] Try stronger models such as transformers (BERT)
- [ ] Containerize the app with Docker
- [ ] Add user authentication
- [ ] Deploy the app to the cloud
- [ ] Show which words influenced each prediction

*Built by `<Ayodele Ogunmakinde>` · [GitHub](https://github.com/<hexa-source>)*
