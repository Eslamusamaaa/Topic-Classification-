"""Manual unigram / bigram / trigram language models with Laplace (add-one) smoothing.

Educational counterpart to the scikit-learn pipeline: builds the counts by hand, scores sentences in log space
(raw products underflow to 0 for long sentences) and compares the three models on held-out sentences.
"""
import math
import random
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import config
from src.preprocessing import build_sentence_dataset


class NGramModel:
    def __init__(self, sentences):
        self.uni, self.bi, self.tri = Counter(), Counter(), Counter()
        for sent in sentences:
            words = sent.split()
            self.uni.update(words)
            self.bi.update(zip(words, words[1:]))
            self.tri.update(zip(words, words[1:], words[2:]))
        self.V = len(self.uni)
        self.total = sum(self.uni.values())

    def p_uni(self, w):
        return (self.uni[w] + 1) / (self.total + self.V)

    def p_bi(self, w1, w2):
        return (self.bi[(w1, w2)] + 1) / (self.uni[w1] + self.V)

    def p_tri(self, w1, w2, w3):
        return (self.tri[(w1, w2, w3)] + 1) / (self.bi[(w1, w2)] + self.V)

    def log_prob(self, sentence, n):
        """Natural-log probability of a sentence under the n-gram model (n = 1, 2 or 3)."""
        w = sentence.split()
        if len(w) < n:
            return float("-inf")
        if n == 1:
            return sum(math.log(self.p_uni(x)) for x in w)
        if n == 2:
            return math.log(self.p_uni(w[0])) + sum(math.log(self.p_bi(a, b)) for a, b in zip(w, w[1:]))
        return (math.log(self.p_uni(w[0])) + math.log(self.p_bi(w[0], w[1]))
                + sum(math.log(self.p_tri(a, b, c)) for a, b, c in zip(w, w[1:], w[2:])))

    def perplexity(self, sentences, n):
        total_lp = total_words = 0
        for s in sentences:
            if len(s.split()) >= n:
                total_lp += self.log_prob(s, n)
                total_words += len(s.split())
        return math.exp(-total_lp / total_words)


def main():
    df = build_sentence_dataset()
    df = df[df["Sentence"].str.split().str.len() >= 3]  # trigram needs 3+ words
    tr, te = next(GroupShuffleSplit(test_size=0.1, random_state=config.SEED).split(df, groups=df["article_id"]))
    model = NGramModel(df.iloc[tr]["Sentence"])
    held_out = df.iloc[te]["Sentence"].tolist()
    print(f"Vocabulary: {model.V:,} | bigrams: {len(model.bi):,} | trigrams: {len(model.tri):,}")

    sample = random.Random(config.SEED).sample(held_out, 2000)
    ppl = {n: model.perplexity(sample, n) for n in (1, 2, 3)}
    for n, name in zip((1, 2, 3), ("Unigram", "Bigram", "Trigram")):
        print(f"{name} perplexity on 2,000 unseen sentences: {ppl[n]:,.0f}")

    config.IMG_DIR.mkdir(parents=True, exist_ok=True)
    examples = sample[:10]
    per_word = {n: [model.log_prob(s, n) / len(s.split()) / math.log(10) for s in examples] for n in (1, 2, 3)}
    x, width = np.arange(len(examples)), 0.27
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for i, (n, name, color) in enumerate(zip((1, 2, 3), ("Unigram", "Bigram", "Trigram"), ("#FFB000", "#3498db", "#e74c3c"))):
        ax[0].bar(x + (i - 1) * width, per_word[n], width, label=name, color=color)
    ax[0].set_xticks(x), ax[0].set_xticklabels([f"S{i + 1}" for i in x])
    ax[0].set_ylabel("average log10 probability per word"), ax[0].set_title("Held-out sentences (higher is better)"), ax[0].legend()
    ax[1].bar(["Unigram", "Bigram", "Trigram"], [ppl[1], ppl[2], ppl[3]], color=["#FFB000", "#3498db", "#e74c3c"])
    ax[1].set_yscale("log"), ax[1].set_title("Perplexity on 2,000 unseen sentences (lower is better)")
    plt.tight_layout(), plt.savefig(config.IMG_DIR / "ngram_comparison.png", dpi=130), plt.close()


if __name__ == "__main__":
    main()
