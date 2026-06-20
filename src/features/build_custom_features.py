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

# --- deeper URL-structure patterns (phishing-specific) ---

# Grab the URLs themselves so we can analyse their structure
URL_GRAB = re.compile(r"(https?://[^\s]+|www\.[^\s]+|\b[a-z0-9.-]+\.(?:com|net|org|uk|io|co|info|biz|tk|xyz|top|ml|ga)\b)")

# IP-based URL: a link whose host is raw numbers (1.2.3.4) instead of a name
IP_URL_PATTERN = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

# Suspicious / commonly-abused top-level domains
SUSPICIOUS_TLD_PATTERN = re.compile(r"\.(tk|xyz|top|ml|ga|gq|cf|buzz|club|work)\b")

# Lookalike domain: a word that mixes letters with digits used as letters
# (paypa1, g00gle, amaz0n) - a letter-run containing 0 or 1 standing in for o/l
LOOKALIKE_PATTERN = re.compile(r"\b[a-z]*[a-z][01][a-z]*[a-z]\b")


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

    # --- deeper URL-structure features ---
    grabbed_urls = [u[0] if isinstance(u, tuple) else u for u in URL_GRAB.findall(text)]
    longest_url_len = max((len(u) for u in grabbed_urls), default=0)
    has_ip_url = 1 if IP_URL_PATTERN.search(text) else 0
    has_suspicious_tld = 1 if SUSPICIOUS_TLD_PATTERN.search(text) else 0
    has_lookalike = 1 if LOOKALIKE_PATTERN.search(text) else 0
    # max dots in any single URL (proxy for excessive subdomains)
    max_url_dots = max((u.count(".") for u in grabbed_urls), default=0)
    
    return {
        "url_count": url_count,
        "has_url": has_url,
        "has_shortened_url": has_shortened_url,
        "has_phone": has_phone,
        "has_shortcode": has_shortcode,
        "capital_ratio": round(capital_ratio, 4),
        "exclamation_count": exclamation_count,
        "currency_count": currency_count,
        "has_ip_url": has_ip_url,
        "has_suspicious_tld": has_suspicious_tld,
        "has_lookalike": has_lookalike,
        "longest_url_len": longest_url_len,
        "max_url_dots": max_url_dots,
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