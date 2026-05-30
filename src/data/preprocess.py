"""
Preprocess the combined dataset.

Order of operations:
1. Drop missing text
2. Strip and collapse whitespace
3. Drop by length (10 to 10,000 characters)
4. Calculate capital_ratio (before lowercasing)
5. Lowercase the text
6. Drop exact duplicates

Input: data/interim/combined.csv
Output: data/processed/cleaned.csv
"""

import re
import pandas as pd
import sys
from pathlib import Path

# Make config.py importable
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_INTERIM, DATA_PROCESSED

# Length bounds for filtering
MIN_LENGTH = 10
MAX_LENGTH = 10_000

def clean_whitespace(text):
    """
    Strip leading/trailing whitespace and collapse multiple spaces/newlines/tabs
    into a single space.
    """
    if not isinstance(text, str):
        return ""
    # \s+ matches any run of whitespace (spaces, tabs, newlines)
    return re.sub(r"\s+", " ", text).strip()

def calculate_capital_ratio(text):
    """
    Fraction of letters in the text that are uppercase.
    Returns 0 for text with no letters.
    """
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    uppercase_count = sum(1 for c in letters if c.isupper())
    return uppercase_count / len(letters)

def preprocess(df):
    """
    Run the full preprocessing pipeline on a DataFrame.
    Returns the cleaned DataFrame.
    Logs row counts at each step.
    """
    print(f"Starting rows: {len(df)}")

    # Step 1: Drop rows with missing text
    before = len(df)
    df = df.dropna(subset=["text"])
    print(f"  After dropping missing text: {len(df)} (lost {before - len(df)})")

    # Step 2: Clean whitespace
    df = df.copy()
    df["text"] = df["text"].apply(clean_whitespace)
    # Re-check for empty strings created by whitespace cleaning
    before = len(df)
    df = df[df["text"].str.len() > 0]
    print(f"  After whitespace cleaning: {len(df)} (lost {before - len(df)})")

    # Step 3: Drop by length bounds
    before = len(df)
    lengths = df["text"].str.len()
    df = df[(lengths >= MIN_LENGTH) & (lengths <= MAX_LENGTH)]
    print(f"  After length filter ({MIN_LENGTH} to {MAX_LENGTH} chars): {len(df)} (lost {before - len(df)})")

    # Step 4: Capital ratio (computed on original-cased text)
    df["capital_ratio"] = df["text"].apply(calculate_capital_ratio)
    print(f"  Capital ratio calculated. Mean: {df['capital_ratio'].mean():.3f}")

    # Step 5: Lowercase the text
    df["text"] = df["text"].str.lower()
    print(f"  Text lowercased.")

    # Step 6: Drop exact duplicates (text only)
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    print(f"  After dedup: {len(df)} (lost {before - len(df)})")

    return df.reset_index(drop=True)

def main():
    input_path = DATA_INTERIM / "combined.csv"
    output_path = DATA_PROCESSED / "cleaned.csv"

    print(f"Loading {input_path}")
    df = pd.read_csv(input_path)
    print()

    df_clean = preprocess(df)
    print()

    # Final summary
    print("=== Final summary ===")
    print(f"Total rows: {len(df_clean)}")
    print()
    print("Label distribution:")
    print(df_clean["label"].value_counts())
    print()
    print("Channel x Label crosstab:")
    print(pd.crosstab(df_clean["channel"], df_clean["label"]))
    print()
    print("Source distribution:")
    print(df_clean["source"].value_counts())
    print()
    print("Capital ratio percentiles:")
    print(df_clean["capital_ratio"].describe().round(3))

    df_clean.to_csv(output_path, index=False)
    print()
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
