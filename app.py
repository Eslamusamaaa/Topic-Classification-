"""Flask app for live news topic classification.  Run:  python app.py  ->  http://127.0.0.1:5000"""
import os

from flask import Flask, jsonify, render_template, request

from src.predictor import TopicPredictor

app = Flask(__name__)

try:
    predictor = TopicPredictor()
except FileNotFoundError as exc:
    raise SystemExit(f"{exc}\nTrain the model first:  python run_pipeline.py") from exc


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Please enter text!"}), 400
    try:
        return jsonify(predictor.predict(text))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
