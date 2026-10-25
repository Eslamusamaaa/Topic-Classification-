"""Central configuration: paths and constants shared by every module."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
IMG_DIR = ROOT / "img" / "Topic-Classification-"

TRAIN_PATH = DATA_DIR / "train.csv.gz"
TEST_PATH = DATA_DIR / "test.csv"
CLEAN_TRAIN_PATH = DATA_DIR / "train_cleaned_sentences.csv.gz"  # generated, not committed

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "topic_classifier.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

SEED = 42
LABEL_NAMES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
