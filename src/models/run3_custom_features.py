"""
RUN 3: Add custom phishing features to word+char TF-IDF, retrain, measure.

Investigates the four research questions:
  Q1 which signals best separate phishing from spam/ham
  Q2 do they improve detection, especially SMS phishing
  Q3 (LIME, later) does the model rely on real signals vs shortcuts
  Q4 do signals generalise across SMS and email

This script handles Q1 and Q2 (measurement). Q3 is the LIME step after.

Compares:
  A: word+char TF-IDF only (baseline, = your 0.957 model)
  B: word+char TF-IDF + 8 custom features
Both use LogisticRegression with balanced class weights (your final model).
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_PROCESSED, RESULTS_DIR, RANDOM_SEED
# import our feature extractor
sys.path.append(str(Path(__file__).resolve().parent.parent / "features"))
from build_custom_features import extract_features

CUSTOM_COLS = [
    "url_count", "has_url", "has_shortened_url", "has_phone",
    "has_shortcode", "exclamation_count", "currency_count",
]
# note: capital_ratio handled separately (from saved column)


def build_custom_matrix(df):
    """
    Run the extractor on every message, plus pull in the saved capital_ratio.
    Returns a DataFrame of custom features in fixed column order.
    """
    extracted = df["text"].apply(extract_features).apply(pd.Series)
    # use the saved capital_ratio (computed before lowercasing), not the recomputed one
    extracted = extracted.drop(columns=["capital_ratio"])
    extracted["capital_ratio"] = df["capital_ratio"].values
    # fixed column order
    cols = CUSTOM_COLS + ["capital_ratio"]
    return extracted[cols]

def build_word_char_tfidf(train_text, test_text):
    """Build the word+char TF-IDF (your best baseline from Run 2)."""
    word_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                               min_df=2, max_df=0.95, stop_words="english")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                               max_features=5000, min_df=2)

    Xw_train = word_vec.fit_transform(train_text)
    Xw_test = word_vec.transform(test_text)
    Xc_train = char_vec.fit_transform(train_text)
    Xc_test = char_vec.transform(test_text)

    X_train = sparse.hstack([Xw_train, Xc_train]).tocsr()
    X_test = sparse.hstack([Xw_test, Xc_test]).tocsr()
    return X_train, X_test


def combine_with_custom(X_tfidf_train, X_tfidf_test, custom_train, custom_test):
    """
    Scale the custom features to 0-1, then glue onto the TF-IDF matrix.
    Scaler is fit on TRAIN only (same no-leakage rule as TF-IDF).
    """
    scaler = MinMaxScaler()
    custom_train_scaled = scaler.fit_transform(custom_train)   # fit + apply on train
    custom_test_scaled = scaler.transform(custom_test)          # apply only on test

    # convert the scaled custom features to sparse so they can join the TF-IDF
    custom_train_sparse = sparse.csr_matrix(custom_train_scaled)
    custom_test_sparse = sparse.csr_matrix(custom_test_scaled)

    X_train = sparse.hstack([X_tfidf_train, custom_train_sparse]).tocsr()
    X_test = sparse.hstack([X_tfidf_test, custom_test_sparse]).tocsr()
    return X_train, X_test

def evaluate(model, X_test, y_test, test_channels):
    y_pred = model.predict(X_test)
    out = {
        "macro_f1": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "phishing_precision": round(precision_score(
            y_test, y_pred, labels=["phishing"], average="macro", zero_division=0), 4),
        "phishing_recall": round(recall_score(
            y_test, y_pred, labels=["phishing"], average="macro", zero_division=0), 4),
    }
    for ch in ["sms", "email"]:
        m = test_channels == ch
        out[f"{ch}_macro_f1"] = round(
            f1_score(y_test[m], y_pred[m], average="macro", zero_division=0), 4)
    sms_phish = (test_channels == "sms") & (y_test == "phishing")
    if sms_phish.sum() > 0:
        out["sms_phishing_recall"] = round(
            (y_pred[sms_phish] == "phishing").sum() / sms_phish.sum(), 4)
        out["sms_phishing_n"] = int(sms_phish.sum())
    return out


def main():
    train = pd.read_csv(DATA_PROCESSED / "train_v2.csv")
    test = pd.read_csv(DATA_PROCESSED / "test_v2.csv")
    y_train, y_test = train["label"].values, test["label"].values
    channels = test["channel"].values
    tr_text = train["text"].fillna("")
    te_text = test["text"].fillna("")

    print("Building word+char TF-IDF...")
    X_tfidf_train, X_tfidf_test = build_word_char_tfidf(tr_text, te_text)

    print("Extracting custom features...")
    custom_train = build_custom_matrix(train)
    custom_test = build_custom_matrix(test)

    print("Combining...")
    X_combined_train, X_combined_test = combine_with_custom(
        X_tfidf_train, X_tfidf_test, custom_train, custom_test)

    results = {}

    # Model A: TF-IDF only (baseline)
    print("\nTraining A: word+char TF-IDF only...")
    model_a = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                                 class_weight="balanced")
    model_a.fit(X_tfidf_train, y_train)
    results["A_tfidf_only"] = evaluate(model_a, X_tfidf_test, y_test, channels)

    # Model B: TF-IDF + custom features
    print("Training B: word+char TF-IDF + custom features...")
    model_b = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                                 class_weight="balanced")
    model_b.fit(X_combined_train, y_train)
    results["B_tfidf_plus_custom"] = evaluate(model_b, X_combined_test, y_test, channels)

    # Q1: which custom features carry weight for the phishing class?
    # The custom features are the LAST 8 columns of the combined matrix.
    custom_names = CUSTOM_COLS + ["capital_ratio"]
    n_custom = len(custom_names)
    phishing_idx = list(model_b.classes_).index("phishing")
    custom_weights = model_b.coef_[phishing_idx][-n_custom:]
    feature_importance = sorted(
        zip(custom_names, custom_weights), key=lambda x: abs(x[1]), reverse=True)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<22}{'A (tfidf)':<12}{'B (+custom)':<12}")
    for k in ["macro_f1", "phishing_precision", "phishing_recall",
              "sms_macro_f1", "email_macro_f1", "sms_phishing_recall"]:
        a = results["A_tfidf_only"].get(k, "n/a")
        b = results["B_tfidf_plus_custom"].get(k, "n/a")
        print(f"{k:<22}{str(a):<12}{str(b):<12}")

    print("\nCustom feature weights for PHISHING class (Q1 - which signals matter):")
    for name, w in feature_importance:
        direction = "toward phishing" if w > 0 else "against phishing"
        print(f"  {name:<20} {w:+.3f}  ({direction})")

    out_path = RESULTS_DIR / "metrics" / "run3_custom_features.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results["custom_feature_weights_phishing"] = {
        n: round(float(w), 4) for n, w in feature_importance}
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()