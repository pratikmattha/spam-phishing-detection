"""
RUN 6: Hyperparameter tuning + k-fold cross-validation (FINAL evaluation).

This is the FAST TEST version - word TF-IDF only - to confirm the
machinery (Pipeline + GridSearchCV + cross-validation) works before
scaling up to the full 8-feature model.

Why a Pipeline: in k-fold, the training data changes each fold, and
TF-IDF must be re-fit on each fold's training portion only (no leakage).
A Pipeline re-fits the whole chain inside each fold automatically.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_PROCESSED, RANDOM_SEED


def main():
    # For tuning/k-fold we use the TRAIN set (the held-out test set stays
    # untouched for a final check later). Cross-validation happens within train.
    train = pd.read_csv(DATA_PROCESSED / "train_v2.csv")
    X_text = train["text"].fillna("")
    y = train["label"].values

    # The pipeline: TF-IDF -> Logistic Regression.
    # In each CV fold, TfidfVectorizer is re-fit on that fold's train portion only.
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                  max_df=0.95, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                                   class_weight="balanced")),
    ])

    # The settings to try (tuning). C controls regularisation strength.
    # max_features lets us also check vocabulary size.
    param_grid = {
        "tfidf__max_features": [5000],
        "clf__C": [0.1, 1.0, 10.0],
    }

    # 5-fold, stratified so each fold keeps the class balance.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # GridSearchCV: tries every setting, cross-validates each, picks the best.
    # scoring = macro F1 (our honest headline metric).
    search = GridSearchCV(
        pipe, param_grid, scoring="f1_macro", cv=cv,
        n_jobs=-1, verbose=2, return_train_score=False,
    )

    print("Running grid search + 5-fold CV (fast test version)...")
    search.fit(X_text, y)

    print("\n" + "=" * 60)
    print("RESULTS (fast test - word TF-IDF only)")
    print("=" * 60)
    print(f"Best macro F1 (CV mean): {search.best_score_:.4f}")
    print(f"Best settings: {search.best_params_}")

    # Show the spread across folds for the best setting
    best_idx = search.best_index_
    fold_scores = [search.cv_results_[f"split{i}_test_score"][best_idx]
                   for i in range(5)]
    print(f"\nPer-fold macro F1 for best setting:")
    for i, s in enumerate(fold_scores):
        print(f"  fold {i+1}: {s:.4f}")
    print(f"  mean: {np.mean(fold_scores):.4f}  std: {np.std(fold_scores):.4f}")

    print("\nAll settings tried:")
    for params, mean, std in zip(
        search.cv_results_["params"],
        search.cv_results_["mean_test_score"],
        search.cv_results_["std_test_score"],
    ):
        print(f"  {params}  ->  {mean:.4f} (+/- {std:.4f})")


if __name__ == "__main__":
    main()