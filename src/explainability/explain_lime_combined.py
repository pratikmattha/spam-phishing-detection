"""
Combined LIME: explain model B (word+char TF-IDF + custom features).

The challenge: LIME perturbs raw text, but model B needs word TF-IDF,
char TF-IDF, AND the 8 custom features - all recomputed from each
perturbed text variant, scaled, and stacked in the right order.

So we build one predict function that does the whole pipeline:
  text -> [word tfidf | char tfidf | scaled custom] -> model -> probabilities
and hand that to LIME.

Treated as a stretch goal. The coefficient inspection already answers Q3;
this adds per-message explanations.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_PROCESSED, RANDOM_SEED
sys.path.append(str(Path(__file__).resolve().parent.parent / "features"))
from build_custom_features import extract_features

CUSTOM_COLS = [
    "url_count", "has_url", "has_shortened_url", "has_phone",
    "has_shortcode", "exclamation_count", "currency_count",
]
CUSTOM_ALL = CUSTOM_COLS + ["capital_ratio"]


def build_pipeline():
    """
    Train model B and return everything needed to predict from raw text:
    the two vectorisers, the scaler, and the model.
    """
    train = pd.read_csv(DATA_PROCESSED / "train_v2.csv")
    y_train = train["label"].values
    tr_text = train["text"].fillna("")

    word_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                               min_df=2, max_df=0.95, stop_words="english")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                               max_features=5000, min_df=2)

    Xw = word_vec.fit_transform(tr_text)
    Xc = char_vec.fit_transform(tr_text)

    # custom features for training (with saved capital_ratio)
    extracted = tr_text.apply(extract_features).apply(pd.Series)
    extracted = extracted.drop(columns=["capital_ratio"])
    extracted["capital_ratio"] = train["capital_ratio"].values
    custom_train = extracted[CUSTOM_ALL]

    scaler = MinMaxScaler()
    Xcustom = sparse.csr_matrix(scaler.fit_transform(custom_train))

    X_train = sparse.hstack([Xw, Xc, Xcustom]).tocsr()

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                               class_weight="balanced")
    model.fit(X_train, y_train)

    return word_vec, char_vec, scaler, model


def make_predict_function(word_vec, char_vec, scaler, model):
    """
    Returns a function LIME can call: list of texts -> probability array.
    For each text it rebuilds word+char TF-IDF AND recomputes the custom
    features (so deleting a URL really does change has_url), scales them,
    stacks everything, and runs the model.

    Note on capital_ratio: LIME works on the already-lowercased text, so a
    recomputed capital_ratio would be ~0. We set it to 0 here for all
    perturbed variants - it simply doesn't participate in the local
    explanation, which is acceptable (the other features carry the signal).
    """
    def predict_proba(texts):
        Xw = word_vec.transform(texts)
        Xc = char_vec.transform(texts)

        # recompute custom features from each perturbed text
        rows = []
        for t in texts:
            feats = extract_features(t)
            feats["capital_ratio"] = 0.0  # see note above
            rows.append([feats[c] for c in CUSTOM_ALL])
        custom = np.array(rows, dtype=float)
        Xcustom = sparse.csr_matrix(scaler.transform(custom))

        X = sparse.hstack([Xw, Xc, Xcustom]).tocsr()
        return model.predict_proba(X)

    return predict_proba

def main():
    print("Building pipeline (training model B)...")
    word_vec, char_vec, scaler, model = build_pipeline()
    predict_fn = make_predict_function(word_vec, char_vec, scaler, model)
    explainer = LimeTextExplainer(class_names=list(model.classes_))

    # pick a few test messages to explain - focus on SMS phishing,
    # since that's where the custom features were designed to help
    test = pd.read_csv(DATA_PROCESSED / "test_v2.csv")
    sms_phish = test[(test["channel"] == "sms") & (test["label"] == "phishing")]

    # take the first 3 SMS phishing messages
    examples = sms_phish["text"].head(3).tolist()

    for text in examples:
        proba = predict_fn([text])[0]
        pred_idx = proba.argmax()
        pred_label = model.classes_[pred_idx]

        print("\n" + "=" * 60)
        print(f"Predicted: {pred_label}")
        print(f"Message: {text[:150]}")
        print("Class probabilities:")
        for cls, p in zip(model.classes_, proba):
            print(f"  {cls}: {p:.3f}")

        explanation = explainer.explain_instance(
            text, predict_fn, num_features=10, labels=[pred_idx])

        print(f"\nTop features influencing '{pred_label}':")
        for word, weight in explanation.as_list(label=pred_idx):
            direction = "toward" if weight > 0 else "against"
            print(f"  {word:<20} {weight:+.3f}  ({direction} {pred_label})")


if __name__ == "__main__":
    main()