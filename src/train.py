"""Compare vectorizers x classifiers, evaluate on held-out data and save the best model.

Fixes compared with a plain sentence-level random split: sentences of the same article are never split between
train and validation (no leakage), the best combination is chosen on validation data only, and the external
test.csv is used once, at the end.
"""
import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import config
from src.preprocessing import build_sentence_dataset, clean_text

warnings.filterwarnings("ignore")

VEC_ARGS = dict(ngram_range=(1, 3), min_df=2, max_df=0.95, stop_words="english")
VECTORIZERS = {
    "binary": lambda: CountVectorizer(binary=True, **VEC_ARGS),
    "count": lambda: CountVectorizer(**VEC_ARGS),
    "tfidf": lambda: TfidfVectorizer(**VEC_ARGS),
}
CLASSIFIERS = {
    "Naive Bayes": lambda: MultinomialNB(alpha=1.0),
    "Logistic Regression": lambda: LogisticRegression(max_iter=200),
    "Linear SVC": lambda: LinearSVC(),
}
NAMES = [config.LABEL_NAMES[i] for i in sorted(config.LABEL_NAMES)]


def compare(X_train, y_train, X_val, y_val):
    """Validation accuracy of every vectorizer x classifier pair. Returns (results, fitted vectorizers)."""
    results, fitted = [], {}
    for vname, make_vec in VECTORIZERS.items():
        vec = make_vec().fit(X_train)
        fitted[vname] = vec
        Xtr, Xva = vec.transform(X_train), vec.transform(X_val)
        for cname, make_clf in CLASSIFIERS.items():
            clf = make_clf().fit(Xtr, y_train)
            acc = accuracy_score(y_val, clf.predict(Xva))
            results.append({"vectorizer": vname, "classifier": cname, "val_accuracy": round(acc, 4)})
            print(f"{vname:7s} + {cname:19s} validation accuracy: {acc:.4f}")
    return pd.DataFrame(results), fitted


def save_plots(results, y_true, y_pred):
    config.IMG_DIR.mkdir(parents=True, exist_ok=True)
    pivot = results.pivot(index="vectorizer", columns="classifier", values="val_accuracy")
    plt.figure(figsize=(7, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar=False)
    plt.title("Validation accuracy: vectorizer x classifier")
    plt.tight_layout(), plt.savefig(config.IMG_DIR / "model_comparison.png", dpi=130), plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=NAMES, yticklabels=NAMES)
    plt.xlabel("Predicted"), plt.ylabel("True")
    plt.title("Confusion matrix on the external test set (7,600 articles)")
    plt.tight_layout(), plt.savefig(config.IMG_DIR / "confusion_matrix.png", dpi=130), plt.close()


def main():
    config.MODEL_DIR.mkdir(exist_ok=True)
    df = build_sentence_dataset()
    print(f"{len(df)} sentences from {df['article_id'].nunique()} articles")

    # article-level split: all sentences of one article go to the same side
    train_idx, val_idx = next(GroupShuffleSplit(test_size=0.2, random_state=config.SEED).split(df, groups=df["article_id"]))
    train, val = df.iloc[train_idx], df.iloc[val_idx]
    results, fitted = compare(train["Sentence"], train["Label"], val["Sentence"], val["Label"])

    best = results.sort_values("val_accuracy", ascending=False).iloc[0]
    print("\nBest on validation:", best.to_dict())

    # external test set, evaluated once with the model trained on the train split
    test = pd.read_csv(config.TEST_PATH)
    X_ext = test["Description"].apply(clean_text)
    vec = fitted[best["vectorizer"]]
    clf = CLASSIFIERS[best["classifier"]]().fit(vec.transform(train["Sentence"]), train["Label"])
    pred = clf.predict(vec.transform(X_ext))
    ext_acc = accuracy_score(test["Class Index"], pred)
    print(f"External test accuracy: {ext_acc:.4f}")
    print(classification_report(test["Class Index"], pred, target_names=NAMES))
    save_plots(results, test["Class Index"], pred)

    # final model: refit on all sentences; LinearSVC is calibrated so the app can show real probabilities
    final_vec = VECTORIZERS[best["vectorizer"]]().fit(df["Sentence"])
    X_all = final_vec.transform(df["Sentence"])
    base = CLASSIFIERS[best["classifier"]]()
    final = CalibratedClassifierCV(base, cv=3).fit(X_all, df["Label"]) if best["classifier"] == "Linear SVC" else base.fit(X_all, df["Label"])
    joblib.dump(final_vec, config.VECTORIZER_PATH, compress=3)
    joblib.dump(final, config.MODEL_PATH, compress=3)

    metrics = {"best": {"vectorizer": best["vectorizer"], "classifier": best["classifier"]},
               "validation_accuracy": float(best["val_accuracy"]), "external_test_accuracy": round(ext_acc, 4),
               "comparison": results.to_dict(orient="records")}
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("Saved model to", config.MODEL_DIR)


if __name__ == "__main__":
    main()
