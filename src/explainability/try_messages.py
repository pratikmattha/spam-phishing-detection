"""
Try-it-out: predict on your own messages and see why.

Uses the final model (word+char TF-IDF + 8 custom features) and LIME
to show the prediction, confidence, and the words that drove it.

Edit the MESSAGES list below to test your own examples.
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

ORIGINAL_8 = ["url_count", "has_url", "has_shortened_url", "has_phone",
              "has_shortcode", "exclamation_count", "currency_count", "capital_ratio"]

# ============================================================
# EDIT THIS LIST - put the messages you want to test here
# ============================================================
MESSAGES = [
    # 1. Phishing with NO url and NO obvious scam words - tests if it needs links/keywords
    "Hi, this is your bank's security team. We noticed unusual activity. Please call us back on the number on your card to confirm it was you.",

    # 2. Real ham that LOOKS suspicious - has urgency + a link (tests false positives)
    "Reminder: your dentist appointment is tomorrow at 3pm. Reschedule here if needed: https://smiledental.co.uk/booking",

    # 3. Phishing disguised as friendly/casual - no formal scam language
    "hey its mike from accounts, can you quickly send me the login for the shared drive? locked out and boss needs the file asap",

    # 4. Spam vs phishing ambiguity - a prize that asks for bank details (could be either)
    "You have been selected for a 1000 cash reward. To receive payment, reply with your name, sort code and account number.",

    # 5. Modern phishing with a lookalike domain (tests the lookalike idea)
    "Your Netflix membership is on hold. Update your payment details at http://netf1ix-billing.com to continue watching.",

    # 6. Legitimate transactional email with money + link (tests false positives on real business)
    "Thank you for your order #48213. Your total was 34.99. Track your delivery at https://orders.amazon.co.uk/track",

    # 7. Very short, almost no context - tests behaviour with minimal signal
    "claim your refund now",

    # 8. Phishing in a non-financial frame (account takeover, no money mentioned)
    "Someone tried to sign in to your email from a new device in Russia. If this wasn't you, secure your account here: bit.ly/secure-mail",
]
# ============================================================


def build_pipeline():
    train = pd.read_csv(DATA_PROCESSED / "train_v2.csv")
    y_train = train["label"].values
    tr_text = train["text"].fillna("")

    word_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                               min_df=2, max_df=0.95, stop_words="english")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                               max_features=5000, min_df=2)
    Xw = word_vec.fit_transform(tr_text)
    Xc = char_vec.fit_transform(tr_text)

    extracted = tr_text.apply(extract_features).apply(pd.Series)
    extracted = extracted.drop(columns=["capital_ratio"])
    extracted["capital_ratio"] = train["capital_ratio"].values
    custom_train = extracted[ORIGINAL_8]

    scaler = MinMaxScaler()
    Xcustom = sparse.csr_matrix(scaler.fit_transform(custom_train))

    X_train = sparse.hstack([Xw, Xc, Xcustom]).tocsr()
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                               class_weight="balanced")
    model.fit(X_train, y_train)
    return word_vec, char_vec, scaler, model


def make_predict_fn(word_vec, char_vec, scaler, model):
    def predict_proba(texts):
        Xw = word_vec.transform(texts)
        Xc = char_vec.transform(texts)
        rows = []
        for t in texts:
            f = extract_features(t)
            f["capital_ratio"] = 0.0  # recomputed-from-lowercased limitation
            rows.append([f[c] for c in ORIGINAL_8])
        Xcustom = sparse.csr_matrix(scaler.transform(np.array(rows, dtype=float)))
        X = sparse.hstack([Xw, Xc, Xcustom]).tocsr()
        return model.predict_proba(X)
    return predict_proba


def main():
    print("Building model...")
    word_vec, char_vec, scaler, model = build_pipeline()
    predict_fn = make_predict_fn(word_vec, char_vec, scaler, model)
    explainer = LimeTextExplainer(class_names=list(model.classes_))

    for text in MESSAGES:
        proba = predict_fn([text])[0]
        pred_idx = int(proba.argmax())
        pred = model.classes_[pred_idx]

        print("\n" + "=" * 60)
        print(f"MESSAGE: {text}")
        print(f"PREDICTION: {pred.upper()}")
        print("Confidence:")
        for cls, p in zip(model.classes_, proba):
            print(f"  {cls}: {p:.3f}")

        exp = explainer.explain_instance(text, predict_fn,
                                         num_features=8, labels=[pred_idx])
        print(f"Top words pushing toward '{pred}':")
        for word, weight in exp.as_list(label=pred_idx):
            arrow = "toward" if weight > 0 else "against"
            print(f"  {word:<18} {weight:+.3f}  ({arrow} {pred})")


if __name__ == "__main__":
    main()