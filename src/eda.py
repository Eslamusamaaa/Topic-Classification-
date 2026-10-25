"""Exploratory analysis of the AG News training set: saves figures to img/Topic-Classification-/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

sns.set_theme(style="whitegrid")


def main():
    config.IMG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.TRAIN_PATH)
    df["Topic"] = df["Class Index"].map(config.LABEL_NAMES)
    df["Description words"] = df["Description"].astype(str).str.split().str.len()
    df["Title words"] = df["Title"].astype(str).str.split().str.len()
    print(df["Topic"].value_counts().to_string())
    print(df[["Description words", "Title words"]].describe().round(1).to_string())

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    order = [config.LABEL_NAMES[i] for i in sorted(config.LABEL_NAMES)]
    sns.countplot(data=df, x="Topic", order=order, hue="Topic", legend=False, palette="Set2", ax=ax[0])
    ax[0].set_title("Articles per topic (balanced)")
    sns.histplot(df["Description words"], bins=50, kde=True, ax=ax[1], color="#6c63ff")
    ax[1].set_title("Description length (words)")
    sns.histplot(df["Title words"], bins=30, kde=True, ax=ax[2], color="#ff7a59")
    ax[2].set_title("Title length (words)")
    plt.tight_layout(), plt.savefig(config.IMG_DIR / "eda_overview.png", dpi=130), plt.close()
    print("Saved", config.IMG_DIR / "eda_overview.png")


if __name__ == "__main__":
    main()
