"""
Explain model predictions using LIME.

LIME works on raw text, but our model works on TF-IDF vectors.
So we build a pipeline function that LIME can call:
    raw text -> TF-IDF transform -> model -> class probabilities

We use LogisticRegression because it outputs probabilities natively
(LinearSVC does not, without extra calibration).

For a given message, LIME shows which words pushed the prediction
toward each class.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from lime.lime_text import LimeTextExplainer

# Make config.py importable
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR

# The class names, in the order the model uses them
# (we'll confirm this matches model.classes_ at runtime)
CLASS_NAMES = ["ham", "phishing", "spam"]


def load_model_and_vectoriser():
    """Load the trained LogisticRegression model and the fitted TF-IDF vectoriser."""
    with open(MODELS_DIR / "baseline_LogisticRegression.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectoriser = pickle.load(f)
    return model, vectoriser


def make_predict_function(model, vectoriser):
    """
    Build the function LIME needs: takes a list of raw text strings,
    returns a 2D array of class probabilities (one row per text).
    """
    def predict_proba(texts):
        X = vectoriser.transform(texts)
        return model.predict_proba(X)
    return predict_proba


def explain_message(text, true_label, explainer, predict_fn, model):
    """
    Run LIME on a single message and print the word contributions.
    """
    # Get the model's actual prediction and confidence
    proba = predict_fn([text])[0]
    pred_idx = proba.argmax()
    pred_label = model.classes_[pred_idx]

    print(f"\n{'=' * 60}")
    print(f"True label: {true_label}  |  Predicted: {pred_label}")
    print(f"Message: {text[:200]}")
    print(f"{'=' * 60}")
    print("Class probabilities:")
    for cls, p in zip(model.classes_, proba):
        print(f"  {cls}: {p:.3f}")

    # Run LIME - explain the predicted class
    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features=10,
        labels=[pred_idx],
    )

    print(f"\nTop words influencing the '{pred_label}' prediction:")
    for word, weight in explanation.as_list(label=pred_idx):
        direction = "toward" if weight > 0 else "against"
        print(f"  {word:<20} {weight:+.3f}  ({direction} {pred_label})")


def main():
    model, vectoriser = load_model_and_vectoriser()

    # Confirm the class order matches our assumption
    print(f"Model classes (in order): {list(model.classes_)}")
    print()

    predict_fn = make_predict_function(model, vectoriser)
    explainer = LimeTextExplainer(class_names=list(model.classes_))

    # Load the test set and pick a few illustrative messages
    test_df = pd.read_csv(DATA_PROCESSED / "test.csv")

    # Pick one clear example of each class to explain
    examples = []
    for label in ["phishing", "spam", "ham"]:
        sample = test_df[test_df["label"] == label].iloc[0]
        examples.append((sample["text"], label))

    for text, true_label in examples:
        explain_message(text, true_label, explainer, predict_fn, model)


if __name__ == "__main__":
    main()