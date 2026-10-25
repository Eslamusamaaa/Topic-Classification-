"""Loads the trained vectorizer + classifier and predicts the topic of a text."""
import joblib

import config
from src.preprocessing import clean_text


class TopicPredictor:
    def __init__(self):
        self.vectorizer = joblib.load(config.VECTORIZER_PATH)
        self.model = joblib.load(config.MODEL_PATH)

    def predict(self, text):
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("The text has no usable words after cleaning (only numbers, symbols or stop words).")
        vec = self.vectorizer.transform([cleaned])
        proba = self.model.predict_proba(vec)[0]
        best = int(proba.argmax())
        label = int(self.model.classes_[best])
        return {
            "topic": config.LABEL_NAMES[label],
            "confidence": round(float(proba[best]) * 100, 1),
            "probabilities": {config.LABEL_NAMES[int(c)]: round(float(p) * 100, 1) for c, p in zip(self.model.classes_, proba)},
            "cleaned": cleaned,
        }
