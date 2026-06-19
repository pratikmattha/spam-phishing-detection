"""
Q3 (Option 1): Inspect what the combined model relies on.

Reads the trained Logistic Regression coefficients and shows the
top-weighted features per class, across all three feature groups:
  - word TF-IDF
  - char TF-IDF
  - the 8 custom phishing features

This is evidence that the model relies on meaningful signals
(e.g. shortened URLs, sensible words) rather than dataset shortcuts.
No perturbation - we read the weights the model already learned.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_PROCESSED, RANDOM_SEED
sys.path.append(str(Path(__file__).resolve().parent.parent / "features"))
from build_custom_features import extract_features

CUSTOM_COLS = [
    "url_count", "has_url", "has_shortened_url", "has_phone",
    "has_shortcode", "exclamation_count", "currency_count",
]


def build_custom_matrix(df):
    extracted = df["text"].apply(extract_features).apply(pd.Series)
    extracted = extracted.drop(columns=["capital_ratio"])
    extracted["capital_ratio"] = df["capital_ratio"].values
    return extracted[CUSTOM_COLS + ["capital_ratio"]]


def main():
    train = pd.read_csv(DATA_PROCESSED / "train_v2.csv")
    y_train = train["label"].values
    tr_text = train["text"].fillna("")

    print("Rebuilding combined features...")
    word_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                               min_df=2, max_df=0.95, stop_words="english")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                               max_features=5000, min_df=2)

    Xw = word_vec.fit_transform(tr_text)
    Xc = char_vec.fit_transform(tr_text)

    custom = build_custom_matrix(train)
    scaler = MinMaxScaler()
    Xcustom = sparse.csr_matrix(scaler.fit_transform(custom))

    X_train = sparse.hstack([Xw, Xc, Xcustom]).tocsr()

    # Build the full list of feature names, in the SAME order they were stacked
    word_names = [f"word:{w}" for w in word_vec.get_feature_names_out()]
    char_names = [f"char:{c}" for c in char_vec.get_feature_names_out()]
    custom_names = [f"custom:{c}" for c in (CUSTOM_COLS + ["capital_ratio"])]
    all_names = np.array(word_names + char_names + custom_names)

    print("Training combined model...")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                               class_weight="balanced")
    model.fit(X_train, y_train)

    # For each class, show the 15 features with the largest positive weight
    print("\n" + "=" * 60)
    print("TOP FEATURES PER CLASS (what the model relies on)")
    print("=" * 60)

    for class_idx, class_name in enumerate(model.classes_):
        weights = model.coef_[class_idx]
        # indices of the 15 highest weights (most 'toward this class')
        top_idx = np.argsort(weights)[-15:][::-1]
        print(f"\n--- {class_name.upper()} ---")
        for i in top_idx:
            print(f"  {all_names[i]:<30} {weights[i]:+.3f}")


if __name__ == "__main__":
    main()