# app.py - Full News Topic Classifier (Frontend + Backend in ONE FILE)
from flask import Flask, request, jsonify
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np

app = Flask(__name__)

# ================================
# 1. Load Models (Pick best combination from summary or manually)
# ================================
BEST_VECTOR_PATH = r"C:\Users\Laptop World\Desktop\NLPPROJECT\tfidf_vectorizer.pkl"
BEST_MODEL_PATH  = r"C:\Users\Laptop World\Desktop\NLPPROJECT\svc_tfidf_model.pkl"

print("Loading vectorizer and model...")
vectorizer = joblib.load(BEST_VECTOR_PATH)
model = joblib.load(BEST_MODEL_PATH)
print("Models loaded successfully!\n")

# ================================
# 2. NLTK Setup
# ================================
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ================================
# 3. Preprocessing Function
# ================================
def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords + Lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    # Re-join
    return ' '.join(tokens)

# ================================
# 4. Prediction Endpoint
# ================================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Please enter text!'})

    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])

    # Prediction
    pred_label = model.predict(vec)[0]

    # Confidence: check if model supports predict_proba
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0]
        confidence = max(prob) * 100
    else:
        # For models like LinearSVC, approximate confidence via decision_function
        if hasattr(model, "decision_function"):
            df = model.decision_function(vec)
            # convert to probability-like using softmax
            df = np.exp(df) / np.sum(np.exp(df))
            confidence = max(df) * 100
        else:
            confidence = 100.0  # fallback

    topics = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
    return jsonify({
        'topic': topics[pred_label],
        'confidence': round(confidence, 1),
        'cleaned': cleaned
    })

# ================================
# 5. Frontend HTML (kept intact)
# ================================
@app.route('/')
def home():
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Topic Classifier</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
            max-width: 600px;
            width: 100%;
        }
        h1 { text-align: center; color: #333; margin-bottom: 10px; font-size: 2.2rem; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; font-size: 1.1rem; }
        textarea {
            width: 100%; height: 120px; padding: 15px; border: 2px solid #e1e1e1;
            border-radius: 12px; font-size: 1rem; resize: none; margin-bottom: 20px;
            transition: border 0.3s;
        }
        textarea:focus { outline: none; border-color: #667eea; }
        button {
            width: 100%; padding: 14px; background: #667eea; color: white;
            border: none; border-radius: 12px; font-size: 1.1rem; font-weight: bold;
            cursor: pointer; transition: 0.3s;
        }
        button:hover { background: #5a6fd8; transform: translateY(-2px); }
        .result { margin-top: 30px; padding: 25px; background: #f8f9ff;
            border-radius: 12px; border-left: 5px solid #667eea; }
        .result h2 { font-size: 1.8rem; margin-bottom: 10px; }
        .confidence { font-size: 1.1rem; color: #555; margin-bottom: 15px; }
        .progress-bar { height: 10px; background: #e0e0e0; border-radius: 5px;
            overflow: hidden; margin-bottom: 15px; }
        #progressFill { height: 100%; background: #667eea; width: 0%; transition: width 0.5s ease; }
        details { margin-top: 15px; font-family: monospace; font-size: 0.9rem; }
        summary { cursor: pointer; color: #667eea; font-weight: bold; }
        pre { background: #f1f1f1; padding: 10px; border-radius: 8px; margin-top: 8px; white-space: pre-wrap; }
        .loading { text-align: center; margin-top: 30px; }
        .spinner {
            width: 40px; height: 40px; border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea; border-radius: 50%;
            animation: spin 1s linear infinite; margin: 0 auto 15px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { background: #ffe6e6; color: #d8000c; padding: 15px;
            border-radius: 8px; margin-top: 20px; text-align: center; border: 1px solid #ffbaba; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>News Topic Classifier</h1>
        <p class="subtitle">Enter any news headline or description</p>

        <textarea id="newsInput" placeholder="e.g. Apple launches iPhone 16 with AI..."></textarea>
        <button id="predictBtn">Predict Topic</button>

        <div id="result" class="result hidden">
            <h2 id="topic"></h2>
            <div class="confidence">Confidence: <span id="confidence">0%</span></div>
            <div class="progress-bar"><div id="progressFill"></div></div>
            <details><summary>View Cleaned Text</summary><pre id="cleaned"></pre></details>
        </div>

        <div id="loading" class="loading hidden">
            <div class="spinner"></div><p>Analyzing...</p>
        </div>

        <div id="error" class="error hidden"></div>
    </div>

    <script>
        document.getElementById('predictBtn').addEventListener('click', async () => {
            const input = document.getElementById('newsInput').value.trim();
            const result = document.getElementById('result');
            const loading = document.getElementById('loading');
            const error = document.getElementById('error');

            result.classList.add('hidden');
            loading.classList.add('hidden');
            error.classList.add('hidden');

            if (!input) {
                error.textContent = 'Please enter some text!';
                error.classList.remove('hidden');
                return;
            }

            loading.classList.remove('hidden');

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: input })
                });
                const data = await res.json();

                if (data.error) {
                    error.textContent = data.error;
                    error.classList.remove('hidden');
                    return;
                }

                document.getElementById('topic').textContent = `${data.topic}`;
                document.getElementById('confidence').textContent = `${data.confidence}%`;
                document.getElementById('cleaned').textContent = data.cleaned;
                document.getElementById('progressFill').style.width = `${data.confidence}%`;

                result.classList.remove('hidden');
            } catch (err) {
                error.textContent = 'Prediction failed. Please try again.';
                error.classList.remove('hidden');
            } finally {
                loading.classList.add('hidden');
            }
        });
    </script>
</body>
</html>
    '''
    return html_content

# ================================
# 6. Run App
# ================================
if __name__ == '__main__':
    print("Server running at http://localhost:5000")
    app.run(debug=True, use_reloader=False)
