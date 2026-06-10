"""
RUN 2: Batched experiments against the v2 baseline.

Experiments (all on v2 data, class_weight='balanced' where supported):
  E0: baseline repeat (word TF-IDF)            - reference row
  E1: year tokens stripped (temporal confound test)
  E2: char n-grams (3-5, word-boundary)
  E3: word + char combined
  E4: ComplementNB on word TF-IDF

Each experiment is wrapped in try/except; partial results always save.
"""

import sys
import json
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_PROCESSED, RESULTS_DIR, RANDOM_SEED

YEAR_PATTERN = re.compile(r"\b(19[9][0-9]|20[0-2][0-9])\b")


def strip_years(text):
    return YEAR_PATTERN.sub(" ", text) if isinstance(text, str) else ""


def word_vec():
    return TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                           min_df=2, max_df=0.95, stop_words="english")


def char_vec():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           max_features=5000, min_df=2)


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
    return out


def fit_models(X_train, X_test, y_train, y_test, channels, include_cnb=False):
    """Train LogReg + LinearSVC (balanced); optionally ComplementNB. Return metrics dict."""
    models = {
        "LogReg_bal": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                                         class_weight="balanced"),
        "LinearSVC_bal": LinearSVC(random_state=RANDOM_SEED, class_weight="balanced"),
    }
    if include_cnb:
        models["ComplementNB"] = ComplementNB()
    out = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        out[name] = evaluate(m, X_test, y_test, channels)
    return out


def main():
    train = pd.read_csv(DATA_PROCESSED / "train_v2.csv")
    test = pd.read_csv(DATA_PROCESSED / "test_v2.csv")
    y_train, y_test = train["label"].values, test["label"].values
    channels = test["channel"].values
    tr_text = train["text"].fillna("")
    te_text = test["text"].fillna("")

    results = {}

    experiments = {
        "E0_baseline_word": lambda: fit_models(
            *_transform(word_vec(), tr_text, te_text),
            y_train=y_train, y_test=y_test, channels=channels, include_cnb=False),
        "E1_years_stripped": lambda: fit_models(
            *_transform(word_vec(), tr_text.apply(strip_years), te_text.apply(strip_years)),
            y_train=y_train, y_test=y_test, channels=channels, include_cnb=False),
        "E2_char_ngrams": lambda: fit_models(
            *_transform(char_vec(), tr_text, te_text),
            y_train=y_train, y_test=y_test, channels=channels, include_cnb=False),
        "E3_word_plus_char": lambda: _combined(tr_text, te_text, y_train, y_test, channels),
        "E4_complement_nb": lambda: fit_models(
            *_transform(word_vec(), tr_text, te_text),
            y_train=y_train, y_test=y_test, channels=channels, include_cnb=True),
    }

    for name, fn in experiments.items():
        print(f"\nRunning {name}...")
        try:
            results[name] = fn()
            print(f"  done")
        except Exception as e:
            print(f"  !! FAILED: {e}")
            results[name] = {"error": str(e)}
        # save after every experiment so nothing is lost
        out_path = RESULTS_DIR / "metrics" / "run2_experiments.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # summary table
    print("\n" + "=" * 78)
    print("RUN 2 SUMMARY  (v2 data, balanced class weights)")
    print("=" * 78)
    print(f"{'Experiment/Model':<36}{'MacroF1':<9}{'SMS-F1':<9}{'Email-F1':<10}{'SMSPhRec':<9}")
    for exp, models in results.items():
        if "error" in models:
            print(f"{exp:<36}FAILED")
            continue
        for mname, m in models.items():
            print(f"{exp + '/' + mname:<36}{m['macro_f1']:<9}{m['sms_macro_f1']:<9}"
                  f"{m['email_macro_f1']:<10}{m.get('sms_phishing_recall', 'n/a'):<9}")
    print(f"\nSaved to {out_path}")


def _transform(vec, tr_text, te_text):
    X_train = vec.fit_transform(tr_text)
    X_test = vec.transform(te_text)
    return X_train, X_test


def _combined(tr_text, te_text, y_train, y_test, channels):
    wv, cv = word_vec(), char_vec()
    Xw_tr, Xw_te = _transform(wv, tr_text, te_text)
    Xc_tr, Xc_te = _transform(cv, tr_text, te_text)
    X_train = sparse.hstack([Xw_tr, Xc_tr]).tocsr()
    X_test = sparse.hstack([Xw_te, Xc_te]).tocsr()
    return fit_models(X_train, X_test, y_train=y_train, y_test=y_test,
                      channels=channels, include_cnb=False)


if __name__ == "__main__":
    main()