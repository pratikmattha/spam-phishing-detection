"""
Train/test split for the cleaned dataset.

Strategy: stratified 80/20 split by (label, channel).
This ensures every combination of channel and label is represented
in both the training and test sets in the same proportions as the
original data.

Input: data/processed/cleaned.csv
Output: data/processed/train.csv and data/processed/test.csv
"""

import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Make config.py importable
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_PROCESSED, RANDOM_SEED


TEST_SIZE = 0.2


def stratified_split(df):
    """
    Split df into train and test sets, stratified by both label and channel.
    Returns (train_df, test_df).
    """
    # Create a combined stratification key
    df = df.copy()
    df["stratify_key"] = df["label"] + "_" + df["channel"]

    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["stratify_key"],
        random_state=RANDOM_SEED,
    )

    # Drop the helper column - it was only needed for stratification
    train = train.drop(columns=["stratify_key"]).reset_index(drop=True)
    test = test.drop(columns=["stratify_key"]).reset_index(drop=True)

    return train, test

def main():
    input_path = DATA_PROCESSED / "cleaned.csv"
    train_path = DATA_PROCESSED / "train.csv"
    test_path = DATA_PROCESSED / "test.csv"

    print(f"Loading {input_path}")
    df = pd.read_csv(input_path)
    print(f"Total rows: {len(df)}")
    print()

    print(f"Splitting 80/20, stratified by (label, channel)...")
    train, test = stratified_split(df)
    print(f"  Train: {len(train)} rows")
    print(f"  Test:  {len(test)} rows")
    print()

    # Verify the stratification worked - both splits should have the same
    # channel x label proportions
    print("Train channel x label:")
    print(pd.crosstab(train["channel"], train["label"]))
    print()
    print("Test channel x label:")
    print(pd.crosstab(test["channel"], test["label"]))
    print()

    # Save
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"Saved train to: {train_path}")
    print(f"Saved test to:  {test_path}")


if __name__ == "__main__":
    main()