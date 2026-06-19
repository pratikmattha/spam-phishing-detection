"""
Extract phishing-specific custom features from message text.

These features are the 'mechanism' for investigating:
- Which phishing signals best separate phishing from spam/ham
- Whether they help the SMS phishing class
- (later, via LIME) whether the model relies on them vs shortcuts

Features (8):
  url_count, has_url, has_shortened_url,
  has_phone, has_shortcode,
  capital_ratio, exclamation_count, currency_count
"""

import re
import numpy as np
import pandas as pd

# --- patterns, compiled once ---

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|\b\w+\.(com|net|org|uk|io|co|info|biz)\b")

# Common URL shorteners and smishing link domains
SHORTENER_PATTERN = re.compile(
    r"\b(bit\.ly|bit\.do|tinyurl|goo\.gl|t\.co|smsg\.io|ow\.ly|is\.gd|buff\.ly|tiny\.cc)\b"
)

# A phone number: a run of 7+ digits, possibly with +, spaces, dashes, brackets
PHONE_PATTERN = re.compile(r"\+?\d[\d\s().-]{6,}\d")

# A shortcode: a standalone 5- or 6-digit number (common in smishing: 'txt to 88039')
SHORTCODE_PATTERN = re.compile(r"\b\d{5,6}\b")

# Currency symbols
CURRENCY_PATTERN = re.compile("[$\u00a3\u20ac]")


def extract_features(text):
    """
    Compute the 8 custom features for one message.
    Returns a dict.
    """
    if not isinstance(text, str):
        text = ""

    # URL features
    urls = URL_PATTERN.findall(text)
    url_count = len(urls)
    has_url = 1 if url_count > 0 else 0
    has_shortened_url = 1 if SHORTENER_PATTERN.search(text) else 0

    # Contact features
    has_phone = 1 if PHONE_PATTERN.search(text) else 0
    has_shortcode = 1 if SHORTCODE_PATTERN.search(text) else 0

    # Structural features
    letters = [c for c in text if c.isalpha()]
    if letters:
        capital_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    else:
        capital_ratio = 0.0
    exclamation_count = text.count("!")
    currency_count = len(CURRENCY_PATTERN.findall(text))

    return {
        "url_count": url_count,
        "has_url": has_url,
        "has_shortened_url": has_shortened_url,
        "has_phone": has_phone,
        "has_shortcode": has_shortcode,
        "capital_ratio": round(capital_ratio, 4),
        "exclamation_count": exclamation_count,
        "currency_count": currency_count,
    }
    
    
def main():
    """Test the extractor on a few example messages so we can see each feature fire."""
    examples = [
        ("phishing-sms", "bankofamerica alert 137943. please follow http://bit.do/cgjk-and re-activate"),
        ("phishing-sms", "ur awarded a city break! txt store to 88039 now. smsg.io/fcvbd"),
        ("phishing-email", "dear customer, confirm your apple id at http://verifyapple.uk immediately!"),
        ("spam-sms", "WIN a £1000 prize!!! call 09061701461 now to claim your reward"),
        ("ham-sms", "hey are we still meeting for lunch tomorrow?"),
        ("ham-email", "thanks for sending the report, i'll review it this afternoon"),
    ]

    for label, text in examples:
        feats = extract_features(text)
        print(f"\n[{label}]")
        print(f"  text: {text[:70]}")
        for name, value in feats.items():
            print(f"    {name}: {value}")


if __name__ == "__main__":
    main()