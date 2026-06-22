"""
RUN 6 (FULL): Tuning + 5-fold CV on the FINAL model.
Final model = word TF-IDF + char TF-IDF + 8 custom features, LogReg balanced.

This is the rigorous, leakage-free final evaluation. Everything is inside
a Pipeline, so in each CV fold the vectorisers, scaler, and feature
extraction are re-fit on that fold's training portion only.

The 8 custom features come from extract_features(), wrapped in a small
sklearn transformer so it fits into the Pipeline.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_PROCESSED, RANDOM_SEED
sys.path.append(str(Path(__file__).resolve().parent.parent / "features"))
from build_custom_features import extract_features

ORIGINAL_8 = ["url_count", "has_url", "has_shortened_url", "has_phone",
              "has_shortcode", "exclamation_count", "currency_count", "capital_ratio"]


class CustomFeatures(BaseEstimator, TransformerMixin):
    """
    Wraps extract_features() as an sklearn transformer so it works in a Pipeline.
    Selects only the 8 features we keep. Scaling is done by a MinMaxScaler
    placed after this in the pipeline.

    Note: capital_ratio normally comes from the saved (pre-lowercase) column,
    but inside a text-only Pipeline we only have the text. We recompute it from
    text here; since the text is already lowercased it will be ~0. This is a
    known limitation of doing capital_ratio inside CV - documented.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for t in X:
            f = extract_features(t if isinstance(t, str) else "")
            rows.append([f[c] for c in ORIGINAL_8])
        return np.array(rows, dtype=float)


def main():
    train = pd.read_csv(DATA_PROCESSED / "train_v2.csv")
    X_text = train["text"].fillna("")
    y = train["label"].values

    # FeatureUnion runs three extractors in parallel and concatenates them:
    #   word TF-IDF | char TF-IDF | (custom features -> scaled)
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                 max_df=0.95, stop_words="english",
                                 max_features=5000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=2, max_features=5000)),
        ("custom", Pipeline([
            ("extract", CustomFeatures()),
            ("scale", MinMaxScaler()),
        ])),
    ])

    pipe = Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                                   class_weight="balanced")),
    ])

    # Tune C. (Vocabulary sizes fixed at 5000 to keep the run manageable.)
    param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=cv,
                          n_jobs=-1, verbose=2)

    print("Running full model tuning + 5-fold CV (this is the slow one)...")
    search.fit(X_text, y)

    print("\n" + "=" * 60)
    print("RESULTS (FULL final model: word+char+custom)")
    print("=" * 60)
    print(f"Best macro F1 (CV mean): {search.best_score_:.4f}")
    print(f"Best settings: {search.best_params_}")

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