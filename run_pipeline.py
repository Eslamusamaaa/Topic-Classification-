"""Run the whole pipeline: EDA -> n-gram language models -> classifier comparison and training.

    python run_pipeline.py
"""
from src import eda, ngram_lm, train

if __name__ == "__main__":
    print("== 1/3 Exploratory analysis ==")
    eda.main()
    print("\n== 2/3 Manual n-gram language models ==")
    ngram_lm.main()
    print("\n== 3/3 Classifier comparison and training ==")
    train.main()
