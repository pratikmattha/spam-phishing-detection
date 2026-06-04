"""
Stage A: TF-IDF baseline features.

Fits a TF-IDF vectoriser on the TRAINING text only (to avoid data leakage),
then transforms both train and test. No custom features, no placeholder
substitution - this is the clean baseline that later experiments compare against.

Outputs (to data/processed/features/):
- X_train_tfidf.npz   sparse TF-IDF matrix for training
- X_test_tfidf.npz    sparse TF-IDF matrix for test
- y_train.npy         training labels
- y_test.npy          test labels

Saves the fitted vectoriser to:
- models/tfidf_vectorizer.pkl
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Make config.py importable
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_PROCESSED, MODELS_DIR

# TF-IDF settings (the baseline configuration)
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)      # unigrams and bigrams
MIN_DF = 2                # ignore terms in fewer than 2 documents
MAX_DF = 0.95             # ignore terms in more than 95% of documents
STOP_WORDS = "english"    # remove common English stopwords


def build_tfidf(train_texts, test_texts):
    """
    Fit TF-IDF on training texts, transform both train and test.
    Returns (X_train, X_test, vectoriser).
    """
    vectoriser = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        stop_words=STOP_WORDS,
    )

    # Fit on training data ONLY, then transform it
    X_train = vectoriser.fit_transform(train_texts)
    # Transform test data using the vocabulary learned from training
    X_test = vectoriser.transform(test_texts)

    return X_train, X_test, vectoriser


def main():
    train_path = DATA_PROCESSED / "train.csv"
    test_path = DATA_PROCESSED / "test.csv"
    features_dir = DATA_PROCESSED / "features"

    print("Loading train/test split...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"  Train: {len(train_df)} rows")
    print(f"  Test:  {len(test_df)} rows")
    print()

    # Guard against any stray missing text
    train_df["text"] = train_df["text"].fillna("")
    test_df["text"] = test_df["text"].fillna("")

    print("Building TF-IDF features (fit on train only)...")
    X_train, X_test, vectoriser = build_tfidf(
        train_df["text"], test_df["text"]
    )
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  Vocabulary size: {len(vectoriser.vocabulary_)}")
    print()

    # Labels
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # Save sparse matrices
    sparse.save_npz(features_dir / "X_train_tfidf.npz", X_train)
    sparse.save_npz(features_dir / "X_test_tfidf.npz", X_test)

    # Save labels
    np.save(features_dir / "y_train.npy", y_train)
    np.save(features_dir / "y_test.npy", y_test)

    # Save the fitted vectoriser
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectoriser, f)

    print("=== Saved ===")
    print(f"  {features_dir / 'X_train_tfidf.npz'}")
    print(f"  {features_dir / 'X_test_tfidf.npz'}")
    print(f"  {features_dir / 'y_train.npy'}")
    print(f"  {features_dir / 'y_test.npy'}")
    print(f"  {MODELS_DIR / 'tfidf_vectorizer.pkl'}")
    print()

    # Quick peek at some vocabulary terms (sanity check)
    vocab_sample = sorted(vectoriser.vocabulary_.keys())[:20]
    print("Sample vocabulary terms (first 20 alphabetically):")
    print(vocab_sample)


if __name__ == "__main__":
    main()