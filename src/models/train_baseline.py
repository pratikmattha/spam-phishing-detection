"""
Stage B: Train baseline models on the TF-IDF features.

Trains three classifiers with default settings (no tuning yet):
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

Evaluates each on the test set with per-class precision/recall/F1,
confusion matrix, and a per-channel breakdown.

This is the BASELINE. Later experiments (tuning, custom features,
placeholder substitution) compare against these numbers.
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

# Make config.py importable
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, RANDOM_SEED


def evaluate_model(name, model, X_test, y_test, test_channels):
    """
    Evaluate a trained model on the test set.
    Prints per-class metrics, confusion matrix, and per-channel F1.
    Returns a dict of metrics for saving.
    """
    y_pred = model.predict(X_test)

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    # Per-class precision/recall/F1
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Confusion matrix
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"Labels: {labels}")
    print(cm)

    # Per-channel breakdown - this is the project-specific check
    print("\nPer-channel F1 (macro):")
    results_by_channel = {}
    for channel in ["sms", "email"]:
        mask = test_channels == channel
        if mask.sum() == 0:
            continue
        channel_f1 = f1_score(
            y_test[mask], y_pred[mask], average="macro", zero_division=0
        )
        results_by_channel[channel] = round(channel_f1, 4)
        print(f"  {channel}: {channel_f1:.4f}  (n={mask.sum()})")

    # Collect metrics for saving
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    return {
        "model": name,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_channel_f1": results_by_channel,
    }
    
    
def train_all_models(X_train, y_train):
    """
    Train the three baseline models with default settings.
    Returns a dict of {name: trained_model}.
    """
    models = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
        "LinearSVC": LinearSVC(
            random_state=RANDOM_SEED,
        ),
    }

    trained = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  done")

    return trained


def main():
    features_dir = DATA_PROCESSED / "features"

    print("Loading features...")
    X_train = sparse.load_npz(features_dir / "X_train_tfidf.npz")
    X_test = sparse.load_npz(features_dir / "X_test_tfidf.npz")
    y_train = np.load(features_dir / "y_train.npy", allow_pickle=True)
    y_test = np.load(features_dir / "y_test.npy", allow_pickle=True)

    # Load the channel column from the test set for the per-channel breakdown
    test_df = pd.read_csv(DATA_PROCESSED / "test.csv")
    test_channels = test_df["channel"].values

    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print()

    # Train all three
    trained = train_all_models(X_train, y_train)

    # Evaluate each
    all_metrics = []
    for name, model in trained.items():
        metrics = evaluate_model(name, model, X_test, y_test, test_channels)
        all_metrics.append(metrics)

    # Comparison summary
    print(f"\n{'=' * 60}")
    print("  COMPARISON (baseline, no tuning)")
    print(f"{'=' * 60}")
    print(f"{'Model':<22}{'Macro F1':<12}{'Weighted F1':<12}")
    for m in all_metrics:
        print(f"{m['model']:<22}{m['macro_f1']:<12}{m['weighted_f1']:<12}")

    # Identify best by macro F1
    best = max(all_metrics, key=lambda m: m["macro_f1"])
    print(f"\nBest by macro F1: {best['model']} ({best['macro_f1']})")

    # Save trained models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in trained.items():
        with open(MODELS_DIR / f"baseline_{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    # Save metrics as JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "metrics" / "baseline_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved models to: {MODELS_DIR}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()