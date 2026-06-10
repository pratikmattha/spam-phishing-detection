"""
RUN 1: One-shot data fixes + v2 baseline + class-weight comparison.

Steps:
1. Fix text encoding (ftfy) on the cleaned data
2. Remove near-duplicate templates (digit-normalised dedup)
3. Re-split train/test (same settings, same seed)
4. Rebuild TF-IDF (same settings, fit on train only)
5. Train models with and without class weights; save all metrics

Outputs use _v2 names so v1 results stay for comparison.
"""

import sys
import json
import pickle
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

import ftfy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, recall_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, RANDOM_SEED

TEST_SIZE = 0.2
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95


# ---------- Step 1: encoding fix ----------

def fix_encoding(df):
    print("Step 1: Fixing text encoding with ftfy...")
    before_sample = df["text"].iloc[0][:50]
    df = df.copy()
    df["text"] = df["text"].apply(lambda t: ftfy.fix_text(t) if isinstance(t, str) else "")
    print(f"  done. ({len(df)} rows)")
    return df


# ---------- Step 2: near-duplicate removal ----------

def remove_near_duplicates(df):
    """
    Make a 'shadow' of each text where all digit runs become '0',
    then drop rows whose shadow already appeared.
    Catches same-template messages that differ only in numbers/dates/codes.
    """
    print("Step 2: Removing near-duplicate templates...")
    df = df.copy()
    shadow = df["text"].str.replace(r"\d+", "0", regex=True)
    before = len(df)
    df = df[~shadow.duplicated(keep="first")].reset_index(drop=True)
    removed = before - len(df)
    print(f"  removed {removed} near-duplicates ({before} -> {len(df)})")
    return df


# ---------- Step 3: split ----------

def split(df):
    print("Step 3: Train/test split (stratified by label+channel)...")
    df = df.copy()
    df["stratify_key"] = df["label"] + "_" + df["channel"]
    train, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["stratify_key"],
        random_state=RANDOM_SEED,
    )
    train = train.drop(columns=["stratify_key"]).reset_index(drop=True)
    test = test.drop(columns=["stratify_key"]).reset_index(drop=True)
    print(f"  train {len(train)} / test {len(test)}")
    print("  test channel x label:")
    print(pd.crosstab(test["channel"], test["label"]))
    return train, test


# ---------- Step 4: TF-IDF ----------

def build_features(train, test):
    print("Step 4: Building TF-IDF (fit on train only)...")
    vec = TfidfVectorizer(
        max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE,
        min_df=MIN_DF, max_df=MAX_DF, stop_words="english",
    )
    X_train = vec.fit_transform(train["text"].fillna(""))
    X_test = vec.transform(test["text"].fillna(""))
    print(f"  X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, X_test, vec


# ---------- Step 5: train + evaluate ----------

def evaluate(model, X_test, y_test, test_channels):
    y_pred = model.predict(X_test)
    out = {
        "macro_f1": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "phishing_recall": round(
            recall_score(y_test, y_pred, labels=["phishing"], average="macro", zero_division=0), 4),
    }
    for ch in ["sms", "email"]:
        m = test_channels == ch
        out[f"{ch}_macro_f1"] = round(
            f1_score(y_test[m], y_pred[m], average="macro", zero_division=0), 4)
    # SMS phishing recall specifically (the number we care most about)
    sms_phish = (test_channels == "sms") & (y_test == "phishing")
    if sms_phish.sum() > 0:
        caught = (y_pred[sms_phish] == "phishing").sum()
        out["sms_phishing_recall"] = round(caught / sms_phish.sum(), 4)
        out["sms_phishing_n"] = int(sms_phish.sum())
    return out


def train_and_evaluate(X_train, y_train, X_test, y_test, test_channels):
    print("Step 5: Training models (with and without class weights)...")
    configs = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "LogisticRegression_balanced": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"),
        "LinearSVC": LinearSVC(random_state=RANDOM_SEED),
        "LinearSVC_balanced": LinearSVC(random_state=RANDOM_SEED, class_weight="balanced"),
    }
    all_metrics = {}
    for name, model in configs.items():
        try:
            print(f"  training {name}...")
            model.fit(X_train, y_train)
            all_metrics[name] = evaluate(model, X_test, y_test, test_channels)
            with open(MODELS_DIR / f"v2_{name}.pkl", "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"  !! {name} failed: {e}")
            all_metrics[name] = {"error": str(e)}
    return all_metrics


def main():
    print("=" * 60)
    print("RUN 1: data fixes + v2 baseline + class weights")
    print("=" * 60)

    df = pd.read_csv(DATA_PROCESSED / "cleaned.csv")
    print(f"Loaded cleaned.csv: {len(df)} rows\n")

    df = fix_encoding(df)
    df = remove_near_duplicates(df)

    df.to_csv(DATA_PROCESSED / "cleaned_v2.csv", index=False)

    train, test = split(df)
    train.to_csv(DATA_PROCESSED / "train_v2.csv", index=False)
    test.to_csv(DATA_PROCESSED / "test_v2.csv", index=False)

    X_train, X_test, vec = build_features(train, test)
    fdir = DATA_PROCESSED / "features"
    sparse.save_npz(fdir / "X_train_tfidf_v2.npz", X_train)
    sparse.save_npz(fdir / "X_test_tfidf_v2.npz", X_test)
    np.save(fdir / "y_train_v2.npy", train["label"].values)
    np.save(fdir / "y_test_v2.npy", test["label"].values)
    with open(MODELS_DIR / "tfidf_vectorizer_v2.pkl", "wb") as f:
        pickle.dump(vec, f)

    metrics = train_and_evaluate(
        X_train, train["label"].values, X_test, test["label"].values,
        test["channel"].values,
    )

    out_path = RESULTS_DIR / "metrics" / "run1_v2_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS (v2 data)")
    print("=" * 60)
    print(f"{'Model':<30}{'MacroF1':<9}{'SMS-F1':<9}{'Email-F1':<10}{'SMSPhishRec':<12}")
    for name, m in metrics.items():
        if "error" in m:
            print(f"{name:<30}FAILED: {m['error']}")
            continue
        print(f"{name:<30}{m['macro_f1']:<9}{m['sms_macro_f1']:<9}"
              f"{m['email_macro_f1']:<10}{m.get('sms_phishing_recall', 'n/a'):<12}")
    print(f"\nSaved metrics to {out_path}")
    print("Reference v1: LinearSVC macro F1 0.9394, SMS 0.7815, email 0.9732, SMS phishing recall 0.42")


if __name__ == "__main__":
    main()