"""
Combine all four sources (UCI SMS, Mishra, Nazario, SpamAssassin)
into a single standardised CSV.

Output columns: text, label, source, channel
Labels: ham, spam, phishing
Channels: sms, email
"""

import pandas as pd
import sys
from pathlib import Path

# Make config.py importable
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_RAW, DATA_INTERIM


def load_uci_sms():
    """
    UCI SMS Spam Collection: tab-separated, two columns (label, text), no header.
    """
    path = DATA_RAW / "uci_sms" / "SMSSpamCollection"
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    df["source"] = "uci_sms"
    df["channel"] = "sms"
    print(f"  UCI SMS: {len(df)} rows, labels={df['label'].unique().tolist()}")
    return df[["text", "label", "source", "channel"]]


def load_mishra_sms():
    """
    Mishra SMS Phishing: CSV with columns LABEL, TEXT, URL, EMAIL, PHONE.
    We drop URL/EMAIL/PHONE - we'll re-extract those ourselves later.
    Normalise 'Smishing' (capital S) -> 'phishing'.
    """
    path = DATA_RAW / "mishra_sms_phishing" / "Dataset_5971.csv"
    df = pd.read_csv(path)
    # Standardise column names to lowercase
    df = df.rename(columns={"LABEL": "label", "TEXT": "text"})
    # Normalise the smishing label
    df["label"] = df["label"].str.lower().replace({"smishing": "phishing"})
    df["source"] = "mishra_sms"
    df["channel"] = "sms"
    print(f"  Mishra: {len(df)} rows, labels={df['label'].unique().tolist()}")
    return df[["text", "label", "source", "channel"]]


def load_nazario():
    """
    Nazario phishing emails: already loaded to interim CSV.
    Concatenate subject into text (subject is content too).
    """
    path = DATA_INTERIM / "nazario_phishing.csv"
    df = pd.read_csv(path)
    # Concatenate subject into text. Handle missing subjects gracefully.
    df["subject"] = df["subject"].fillna("")
    df["text"] = df["subject"] + " " + df["text"]
    df["source"] = "nazario"
    df["channel"] = "email"
    print(f"  Nazario: {len(df)} rows, labels={df['label'].unique().tolist()}")
    return df[["text", "label", "source", "channel"]]


def load_spamassassin():
    """
    SpamAssassin: already loaded to interim CSV.
    Same subject + text treatment as Nazario.
    """
    path = DATA_INTERIM / "spamassassin.csv"
    df = pd.read_csv(path)
    df["subject"] = df["subject"].fillna("")
    df["text"] = df["subject"] + " " + df["text"]
    df["source"] = "spamassassin"
    df["channel"] = "email"
    print(f"  SpamAssassin: {len(df)} rows, labels={df['label'].unique().tolist()}")
    return df[["text", "label", "source", "channel"]]

def validate(df):
    """
    Sanity checks on the combined DataFrame.
    Prints anything that looks suspicious.
    """
    print()
    print("=== Validation ===")

    # Check for missing values in critical columns
    for col in ["text", "label", "source", "channel"]:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"  WARNING: {missing} missing values in '{col}'")

    # Check labels are exactly what we expect
    expected_labels = {"ham", "spam", "phishing"}
    actual_labels = set(df["label"].unique())
    unexpected = actual_labels - expected_labels
    if unexpected:
        print(f"  WARNING: unexpected label values: {unexpected}")

    # Check channels are exactly what we expect
    expected_channels = {"sms", "email"}
    actual_channels = set(df["channel"].unique())
    unexpected = actual_channels - expected_channels
    if unexpected:
        print(f"  WARNING: unexpected channel values: {unexpected}")

    # Check for very short text (likely junk)
    short_text = (df["text"].str.len() < 5).sum()
    if short_text > 0:
        print(f"  Note: {short_text} rows have text shorter than 5 characters")

    # Check the channel-label matrix
    print()
    print("  Channel x Label crosstab:")
    print(pd.crosstab(df["channel"], df["label"]))


def main():
    print("Loading all four sources...")

    parts = [
        load_uci_sms(),
        load_mishra_sms(),
        load_nazario(),
        load_spamassassin(),
    ]

    df = pd.concat(parts, ignore_index=True)
    print()
    print(f"Combined total: {len(df)} rows")

    validate(df)

    # Save
    output_path = DATA_INTERIM / "combined.csv"
    df.to_csv(output_path, index=False)
    print()
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
