"""Text cleaning and sentence splitting used by training, the n-gram model and the web app."""
import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

import config

_URL = re.compile(r"http[s]?://\S+")
_EMAIL = re.compile(r"\S+@\S+")
_DIGITS = re.compile(r"\d+")
_PUNCT = str.maketrans("", "", string.punctuation)

_resources_ready = False
_lemmatizer = None
_stop_words = None


def ensure_nltk():
    """Download the NLTK data this project needs (only the first time)."""
    global _resources_ready, _lemmatizer, _stop_words
    if _resources_ready:
        return
    for pkg in ("punkt", "punkt_tab", "wordnet", "stopwords"):
        nltk.download(pkg, quiet=True)
    _lemmatizer = WordNetLemmatizer()
    _stop_words = set(stopwords.words("english"))
    _resources_ready = True


def clean_text(text):
    """Lowercase, strip urls/emails/numbers/punctuation, drop stop words, lemmatize.

    Training and the web app both call this function, so inference sees exactly the text the model was trained on.
    """
    ensure_nltk()
    text = str(text).lower()
    text = _URL.sub("", text)
    text = _EMAIL.sub("", text)
    text = _DIGITS.sub("", text)
    text = text.translate(_PUNCT)
    tokens = [_lemmatizer.lemmatize(w) for w in word_tokenize(text) if w not in _stop_words]
    return " ".join(tokens)


def split_and_clean(text, min_words=2):
    """Split a description into sentences and clean each one; sentences shorter than min_words are dropped."""
    ensure_nltk()
    cleaned = (clean_text(s) for s in sent_tokenize(str(text).lower()))
    return [s for s in cleaned if len(s.split()) >= min_words]


def build_sentence_dataset(force=False):
    """One row per sentence with its article id and label. Cached because cleaning 120k articles takes minutes."""
    if config.CLEAN_TRAIN_PATH.exists() and not force:
        return pd.read_csv(config.CLEAN_TRAIN_PATH)
    raw = pd.read_csv(config.TRAIN_PATH)
    rows = [
        {"article_id": i, "Sentence": sent, "Label": label}
        for i, (desc, label) in enumerate(zip(raw["Description"], raw["Class Index"]))
        for sent in split_and_clean(desc)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(config.CLEAN_TRAIN_PATH, index=False)
    return df
