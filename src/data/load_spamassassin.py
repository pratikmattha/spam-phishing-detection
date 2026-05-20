"""
Load all SpamAssassin emails from the extracted folders,
extract the text body, and save to a single CSV.
"""

import email
import pandas as pd
import sys
from pathlib import Path
from bs4 import BeautifulSoup

# Make config.py importable
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_INTERIM


# Map folder names to labels
FOLDER_LABELS = {
    "easy_ham": "ham",
    "easy_ham_2": "ham",
    "hard_ham": "ham",
    "spam": "spam",
    "spam_2": "spam",
}


def get_text_body(msg):
    """
    Extract plain text body from an email message.
    Tries text/plain first. Falls back to text/html (stripped of tags).
    Returns empty string if no usable text found.
    """
    plain_text = None
    html_text = None

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            try:
                decoded = payload.decode("utf-8", errors="replace")
            except (UnicodeDecodeError, AttributeError):
                continue
            if content_type == "text/plain" and plain_text is None:
                plain_text = decoded
            elif content_type == "text/html" and html_text is None:
                html_text = decoded
    else:
        content_type = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        if payload:
            try:
                decoded = payload.decode("utf-8", errors="replace")
                if content_type == "text/plain":
                    plain_text = decoded
                elif content_type == "text/html":
                    html_text = decoded
            except (UnicodeDecodeError, AttributeError):
                pass

    if plain_text:
        return plain_text
    if html_text:
        soup = BeautifulSoup(html_text, "lxml")
        return soup.get_text(separator=" ", strip=True)
    return ""


def load_email_file(file_path):
    """Parse a single .eml-style file and return its body and subject."""
    with open(file_path, "rb") as f:
        msg = email.message_from_bytes(f.read())
    body = get_text_body(msg)
    subject = msg["Subject"] or ""
    return body, subject


def load_folder(folder_path, label):
    """Load all email files from one folder. Skips non-email files like 'cmds'."""
    records = []
    skipped = 0

    for file_path in sorted(folder_path.iterdir()):
        # Skip the 'cmds' file and any other non-email files
        if not file_path.name[0].isdigit():
            continue

        try:
            body, subject = load_email_file(file_path)
            if not body or len(body.strip()) < 10:
                skipped += 1
                continue
            records.append({
                "text": body.strip(),
                "subject": subject,
                "source_file": f"{folder_path.name}/{file_path.name}",
                "label": label,
            })
        except Exception as e:
            skipped += 1
            continue

    return records, skipped


def main():
    extracted_dir = DATA_INTERIM / "spamassassin_extracted"

    all_records = []
    for folder_name, label in FOLDER_LABELS.items():
        folder_path = extracted_dir / folder_name
        if not folder_path.exists():
            print(f"Warning: {folder_path} not found, skipping")
            continue

        print(f"Processing {folder_name} (label: {label})...")
        records, skipped = load_folder(folder_path, label)
        print(f"  -> {len(records)} usable, {skipped} skipped")
        all_records.extend(records)

    df = pd.DataFrame(all_records)

    output_path = DATA_INTERIM / "spamassassin.csv"
    df.to_csv(output_path, index=False)

    print()
    print("=== Summary ===")
    print(f"Total usable emails: {len(df)}")
    print()
    print("Label counts:")
    print(df["label"].value_counts())
    print()
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()