"""
RUN 4: Remove source/era confounds, retrain, re-measure.

Decision (documented):
  STRIP - clear source/era artefacts that are not content:
    sender/source: jose, monkey, monkey.org
    year tokens:   any 1990-2029
    header junk:   utf
  LEAVE - real corpus words (spamassassin, razor, cnet, sightings)
    handled as a documented limitation, not stripped, because they are
    genuine message content and editing them would be gaming the data.

Compares:
  A: model B as-is (word+char+custom, the 0.955 model)
  B: same, but with confound tokens stripped from text first
Shows the F1 change (quantifies the confound's effect) and re-runs
coefficient inspection to confirm the artefacts are gone from the top features.
"""

import sys
import re
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

CUSTOM_ALL = ["url_count", "has_url", "has_shortened_url", "has_phone",
              "has_shortcode", "exclamation_count", "currency_count", "capital_ratio"]

# Patterns for the confound tokens we strip
YEAR_PATTERN = re.compile(r"\b(19[9][0-9]|20[0-2][0-9])\b")
# whole-word source artefacts (word boundaries so we don't hit substrings)
SOURCE_PATTERN = re.compile(r"\b(jose|monkey|monkey\.org|org|utf)\b")


def strip_confounds(text):
    """Remove source/era artefact tokens from a message."""
    if not isinstance(text, str):
        return ""
    text = YEAR_PATTERN.sub(" ", text)
    text = SOURCE_PATTERN.sub(" ", text)
    return text


def build_custom_matrix(df):
    extracted = df["text"].apply(extract_features).apply(pd.Series)
    extracted = extracted.drop(columns=["capital_ratio"])
    extracted["capital_ratio"] = df["capital_ratio"].values
    return extracted[CUSTOM_ALL]


def build_combined(train_text, test_text, custom_train, custom_test):
    """Word+char TF-IDF + scaled custom, stacked. Returns matrices + vectorisers."""
    word_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                               min_df=2, max_df=0.95, stop_words="english")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                               max_features=5000, min_df=2)
    Xw_tr = word_vec.fit_transform(train_text)
    Xw_te = word_vec.transform(test_text)
    Xc_tr = char_vec.fit_transform(train_text)
    Xc_te = char_vec.transform(test_text)

    scaler = MinMaxScaler()
    Xcu_tr = sparse.csr_matrix(scaler.fit_transform(custom_train))
    Xcu_te = sparse.csr_matrix(scaler.transform(custom_test))

    X_tr = sparse.hstack([Xw_tr, Xc_tr, Xcu_tr]).tocsr()
    X_te = sparse.hstack([Xw_te, Xc_te, Xcu_te]).tocsr()
    return X_tr, X_te, word_vec, char_vec

def evaluate(model, X_test, y_test, channels):
    y_pred = model.predict(X_test)
    out = {
        "macro_f1": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "phishing_precision": round(precision_score(
            y_test, y_pred, labels=["phishing"], average="macro", zero_division=0), 4),
        "phishing_recall": round(recall_score(
            y_test, y_pred, labels=["phishing"], average="macro", zero_division=0), 4),
    }
    for ch in ["sms", "email"]:
        m = channels == ch
        out[f"{ch}_macro_f1"] = round(
            f1_score(y_test[m], y_pred[m], average="macro", zero_division=0), 4)
    sms_phish = (channels == "sms") & (y_test == "phishing")
    if sms_phish.sum() > 0:
        out["sms_phishing_recall"] = round(
            (y_pred[sms_phish] == "phishing").sum() / sms_phish.sum(), 4)
    return out


def top_phishing_features(model, word_vec, char_vec):
    """Return the top-15 features for the phishing class, to check artefacts are gone."""
    word_names = [f"word:{w}" for w in word_vec.get_feature_names_out()]
    char_names = [f"char:{c}" for c in char_vec.get_feature_names_out()]
    custom_names = [f"custom:{c}" for c in CUSTOM_ALL]
    all_names = np.array(word_names + char_names + custom_names)

    phishing_idx = list(model.classes_).index("phishing")
    weights = model.coef_[phishing_idx]
    top_idx = np.argsort(weights)[-15:][::-1]
    return [(all_names[i], round(float(weights[i]), 3)) for i in top_idx]


def main():
    train = pd.read_csv(DATA_PROCESSED / "train_v2.csv")
    test = pd.read_csv(DATA_PROCESSED / "test_v2.csv")
    y_train, y_test = train["label"].values, test["label"].values
    channels = test["channel"].values

    # custom features computed from ORIGINAL text (confounds don't affect them)
    custom_train = build_custom_matrix(train)
    custom_test = build_custom_matrix(test)

    results = {}

    # --- Model A: as-is (original text) ---
    print("Training A: model B as-is (with confounds)...")
    Xa_tr, Xa_te, wv_a, cv_a = build_combined(
        train["text"].fillna(""), test["text"].fillna(""), custom_train, custom_test)
    model_a = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                                 class_weight="balanced")
    model_a.fit(Xa_tr, y_train)
    results["A_with_confounds"] = evaluate(model_a, Xa_te, y_test, channels)

    # --- Model B: confounds stripped from text ---
    print("Training B: confounds stripped from text...")
    tr_text_stripped = train["text"].fillna("").apply(strip_confounds)
    te_text_stripped = test["text"].fillna("").apply(strip_confounds)
    Xb_tr, Xb_te, wv_b, cv_b = build_combined(
        tr_text_stripped, te_text_stripped, custom_train, custom_test)
    model_b = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                                 class_weight="balanced")
    model_b.fit(Xb_tr, y_train)
    results["B_confounds_stripped"] = evaluate(model_b, Xb_te, y_test, channels)

    # --- Report ---
    print("\n" + "=" * 60)
    print("CONFOUND REMOVAL: before vs after")
    print("=" * 60)
    print(f"{'Metric':<22}{'A (with)':<12}{'B (stripped)':<12}")
    for k in ["macro_f1", "phishing_precision", "phishing_recall",
              "sms_macro_f1", "email_macro_f1", "sms_phishing_recall"]:
        a = results["A_with_confounds"].get(k, "n/a")
        b = results["B_confounds_stripped"].get(k, "n/a")
        print(f"{k:<22}{str(a):<12}{str(b):<12}")

    print("\nTop phishing features WITH confounds (model A):")
    for name, w in top_phishing_features(model_a, wv_a, cv_a):
        print(f"  {name:<30} {w:+.3f}")

    print("\nTop phishing features AFTER stripping (model B):")
    for name, w in top_phishing_features(model_b, wv_b, cv_b):
        print(f"  {name:<30} {w:+.3f}")

    out_path = RESULTS_DIR / "metrics" / "run4_confound_removal.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results["top_phishing_A"] = top_phishing_features(model_a, wv_a, cv_a)
    results["top_phishing_B"] = top_phishing_features(model_b, wv_b, cv_b)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
