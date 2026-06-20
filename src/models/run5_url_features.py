"""
RUN 5: Add deeper URL-structure features (5 new) on top of the original 8.
Tests whether IP-URLs, suspicious TLDs, lookalike domains, URL length,
and subdomain-count improve phishing detection - especially email phishing,
where raw URLs are more common than in shortened-link SMS.

Compares:
  A: word+char TF-IDF + original 8 custom features (= Run 3 model B)
  B: word+char TF-IDF + all 13 custom features
Reads which NEW features carry weight for phishing (Q1).
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
sys.path.append(str(Path(__file__).resolve().parent.parent / "features"))
from build_custom_features import extract_features

ORIGINAL_8 = ["url_count", "has_url", "has_shortened_url", "has_phone",
              "has_shortcode", "exclamation_count", "currency_count", "capital_ratio"]
NEW_5 = ["has_ip_url", "has_suspicious_tld", "has_lookalike",
         "longest_url_len", "max_url_dots"]
ALL_13 = ORIGINAL_8 + NEW_5


def build_custom_matrix(df, cols):
    extracted = df["text"].apply(extract_features).apply(pd.Series)
    extracted = extracted.drop(columns=["capital_ratio"])
    extracted["capital_ratio"] = df["capital_ratio"].values
    return extracted[cols]


def build_tfidf(train_text, test_text):
    word_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                               min_df=2, max_df=0.95, stop_words="english")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                               max_features=5000, min_df=2)
    Xw_tr = word_vec.fit_transform(train_text)
    Xw_te = word_vec.transform(test_text)
    Xc_tr = char_vec.fit_transform(train_text)
    Xc_te = char_vec.transform(test_text)
    return (sparse.hstack([Xw_tr, Xc_tr]).tocsr(),
            sparse.hstack([Xw_te, Xc_te]).tocsr())


def add_custom(X_tfidf_tr, X_tfidf_te, custom_tr, custom_te):
    scaler = MinMaxScaler()
    tr = sparse.csr_matrix(scaler.fit_transform(custom_tr))
    te = sparse.csr_matrix(scaler.transform(custom_te))
    return (sparse.hstack([X_tfidf_tr, tr]).tocsr(),
            sparse.hstack([X_tfidf_te, te]).tocsr())


def evaluate(model, X_te, y_te, channels):
    y_pred = model.predict(X_te)
    out = {
        "macro_f1": round(f1_score(y_te, y_pred, average="macro", zero_division=0), 4),
        "phishing_precision": round(precision_score(y_te, y_pred, labels=["phishing"], average="macro", zero_division=0), 4),
        "phishing_recall": round(recall_score(y_te, y_pred, labels=["phishing"], average="macro", zero_division=0), 4),
    }
    for ch in ["sms", "email"]:
        m = channels == ch
        out[f"{ch}_macro_f1"] = round(f1_score(y_te[m], y_pred[m], average="macro", zero_division=0), 4)
    sp = (channels == "sms") & (y_te == "phishing")
    if sp.sum() > 0:
        out["sms_phishing_recall"] = round((y_pred[sp] == "phishing").sum() / sp.sum(), 4)
    return out


def main():
    train = pd.read_csv(DATA_PROCESSED / "train_v2.csv")
    test = pd.read_csv(DATA_PROCESSED / "test_v2.csv")
    y_train, y_test = train["label"].values, test["label"].values
    channels = test["channel"].values

    print("Building TF-IDF...")
    X_tr, X_te = build_tfidf(train["text"].fillna(""), test["text"].fillna(""))

    results = {}

    # Model A: original 8 features
    print("Training A: + original 8 features...")
    cA_tr = build_custom_matrix(train, ORIGINAL_8)
    cA_te = build_custom_matrix(test, ORIGINAL_8)
    XA_tr, XA_te = add_custom(X_tr, X_te, cA_tr, cA_te)
    mA = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")
    mA.fit(XA_tr, y_train)
    results["A_original_8"] = evaluate(mA, XA_te, y_test, channels)

    # Model B: all 13 features
    print("Training B: + all 13 features...")
    cB_tr = build_custom_matrix(train, ALL_13)
    cB_te = build_custom_matrix(test, ALL_13)
    XB_tr, XB_te = add_custom(X_tr, X_te, cB_tr, cB_te)
    mB = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")
    mB.fit(XB_tr, y_train)
    results["B_all_13"] = evaluate(mB, XB_te, y_test, channels)

    # Q1: weights of the NEW features for phishing
    phishing_idx = list(mB.classes_).index("phishing")
    new_weights = mB.coef_[phishing_idx][-len(NEW_5):]  # last 5 columns are the new ones
    new_importance = sorted(zip(NEW_5, new_weights), key=lambda x: abs(x[1]), reverse=True)

    print("\n" + "=" * 60)
    print("RUN 5: original 8 vs all 13 features")
    print("=" * 60)
    print(f"{'Metric':<22}{'A (8)':<12}{'B (13)':<12}")
    for k in ["macro_f1", "phishing_precision", "phishing_recall",
              "sms_macro_f1", "email_macro_f1", "sms_phishing_recall"]:
        print(f"{k:<22}{str(results['A_original_8'].get(k)):<12}{str(results['B_all_13'].get(k)):<12}")

    print("\nNEW feature weights for PHISHING (Q1 - do the new URL features matter):")
    for name, w in new_importance:
        direction = "toward phishing" if w > 0 else "against phishing"
        print(f"  {name:<22} {w:+.3f}  ({direction})")

    out_path = RESULTS_DIR / "metrics" / "run5_url_features.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results["new_feature_weights_phishing"] = {n: round(float(w), 4) for n, w in new_importance}
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()