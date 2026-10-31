
"""In this module, we define the Preprocessor class, which is responsible
for preprocessing the input data for machine learning tasks. The class includes
methods for text cleaning step (lowercasing, removing HTML tags, etc.), tokenization, and lemmatization.
It also provides functionality to handle missing values and encode categorical variables."""

import re
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from spellchecker import SpellChecker

# Local module
from Deliverable3.src.data_prep import load_and_prepare_data


# One-time downloads (safe to re-run)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


# Individual cleaning functions (each does exactly one job)

def to_lowercase(text: str) -> str:
    return text.lower()


def remove_html(text: str) -> str:
    """Remove HTML tags like <p>, <br/>, <a href=...>."""
    return re.sub(r"<.*?>", " ", text)


def remove_urls(text: str) -> str:
    """Remove http(s):// links and www. links."""
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_source_tags(text: str) -> str:
    """Remove the 'WASHINGTON (Reuters) -' prefix to avoid data leakage.
    Expects lowercase text, so run it after to_lowercase."""
    return re.sub(r"^.{0,60}?\(reuters\)\s*-\s*", "", text)


def remove_punctuation(text: str) -> str:
    PUNC_TO_REMOVE = string.punctuation
    """Keep only letters and spaces (drops punctuation, digits, symbols)."""
    return text.translate(str.maketrans("", "", PUNC_TO_REMOVE))


def remove_stopwords(text: str) -> str:
    return " ".join(word for word in text.split() if word not in STOP_WORDS)


def lemmatize_text(text: str) -> str:
    """Reduce words to dictionary form: 'studies' -> 'study', 'countries' -> 'country'."""
    return " ".join(lemmatizer.lemmatize(word) for word in text.split())


def stem_text(text: str) -> str:
    """Chop words to a crude root: 'running' -> 'run', 'political' -> 'polit'."""
    return " ".join(stemmer.stem(word) for word in text.split())

def correct_spelling(text: str) -> str:
    """Correct spelling mistakes in the text."""
    spell = SpellChecker()
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)


# Pipeline:

def pipeline(text: str, method: str = "lemmatize", spelling_correction: bool = False) -> str:
    """method: 'lemmatize', 'stem', or None (skip normalization)."""
    text = to_lowercase(text)
    print(f"After lowercase: {text[:60]}...")
    text = remove_html(text)
    print(f"After html: {text[:60]}...")
    text = remove_urls(text)
    print(f"After Url: {text[:60]}...")
    text = remove_source_tags(text)
    print(f"After source_tag: {text[:60]}...")
    text = remove_punctuation(text)
    print(f"After punctuation: {text[:60]}...")
    text = remove_stopwords(text)
    print(f"After stopwords: {text[:60]}...")
    if spelling_correction:
        text = correct_spelling(text)
        print(f"After spelling correction: {text[:60]}...")
    if method == "lemmatize":
        text = lemmatize_text(text)
        print(f"After lematized: {text[:60]}...")
    elif method == "stem":
        text = stem_text(text)
        print(f"After stemmer: {text[:60]}...")
    return text

def preprocessor(file_path: str):
    df = load_and_prepare_data(file_path)
    df["clean_text"] = df["content"].apply(lambda t: pipeline(t, method="lemmatize", spelling_correction=False))
    df = df[df["clean_text"] != ""].reset_index(drop=True)
    
    return df
    