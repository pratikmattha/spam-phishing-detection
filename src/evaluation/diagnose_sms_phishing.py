"""
Diagnose SMS phishing errors from the best baseline model (LinearSVC).

Loads the trained model and test set, finds all SMS phishing messages,
and shows what the model predicted for each - focusing on the ones it
got wrong. The goal is to understand WHY SMS phishing is hard:
- Is the model calling them ham (dangerous - phishing slips through)?
- Is it calling them spam (less dangerous - still gets filtered)?
- What do the misclassified messages look like?
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

# Make config.py importable
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_PROCESSED, MODELS_DIR

def show_errors(sms_phishing):
    """Print the misclassified SMS phishing messages, grouped by wrong prediction."""

    # The dangerous ones first: phishing called ham
    called_ham = sms_phishing[sms_phishing["predicted"] == "ham"]
    print(f"{'=' * 60}")
    print(f"  DANGEROUS: phishing classified as HAM ({len(called_ham)})")
    print(f"  (these would reach the user's inbox)")
    print(f"{'=' * 60}")
    for i, row in enumerate(called_ham.itertuples(), 1):
        print(f"\n[{i}] {row.text[:250]}")

    # The less-bad ones: phishing called spam
    called_spam = sms_phishing[sms_phishing["predicted"] == "spam"]
    print(f"\n{'=' * 60}")
    print(f"  phishing classified as SPAM ({len(called_spam)})")
    print(f"  (still filtered, less dangerous)")
    print(f"{'=' * 60}")
    for i, row in enumerate(called_spam.itertuples(), 1):
        print(f"\n[{i}] {row.text[:250]}")
        
def main():
    features_dir = DATA_PROCESSED / "features"

    # Load the best baseline model
    with open(MODELS_DIR / "baseline_LinearSVC.pkl", "rb") as f:
        model = pickle.load(f)

    # Load test features and the test dataframe (for text + channel)
    X_test = sparse.load_npz(features_dir / "X_test_tfidf.npz")
    test_df = pd.read_csv(DATA_PROCESSED / "test.csv")

    # Predict on the whole test set
    test_df["predicted"] = model.predict(X_test)

    # Focus on SMS phishing only
    sms_phishing = test_df[
        (test_df["channel"] == "sms") & (test_df["label"] == "phishing")
    ].copy()

    print(f"Total SMS phishing in test set: {len(sms_phishing)}")
    print()

    # How were they classified?
    print("What the model predicted for SMS phishing messages:")
    print(sms_phishing["predicted"].value_counts())
    print()

    # Correct vs wrong
    correct = (sms_phishing["predicted"] == "phishing").sum()
    wrong = len(sms_phishing) - correct
    print(f"Correctly identified: {correct}/{len(sms_phishing)}")
    print(f"Misclassified: {wrong}/{len(sms_phishing)}")
    print()
    
    show_errors(sms_phishing)


if __name__ == "__main__":
    main()