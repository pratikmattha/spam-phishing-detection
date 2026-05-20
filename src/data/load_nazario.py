"""
Load all Nazario phishing emails from the four mbox files,
extract the text body, and save to a single CSV.
"""

import mailbox
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

# Make config.py importable - add project root to path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_RAW, DATA_INTERIM


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


def load_mbox_file(mbox_path):
    """
    Load all emails from one mbox file.
    Returns a list of dicts with text, source_file, subject.
    """
    records = []
    mbox = mailbox.mbox(str(mbox_path))
    skipped = 0

    for msg in mbox:
        try:
            body = get_text_body(msg)
            if not body or len(body.strip()) < 10:
                skipped += 1
                continue
            records.append({
                "text": body.strip(),
                "subject": msg["Subject"] or "",
                "source_file": mbox_path.name,
            })
        except Exception as e:
            skipped += 1
            continue

    return records, skipped


def main():
    nazario_dir = DATA_RAW / "nazario_phishing"
    mbox_files = sorted(nazario_dir.glob("*.mbox"))

    print(f"Found {len(mbox_files)} mbox files:")
    for f in mbox_files:
        print(f"  - {f.name}")
    print()

    all_records = []
    for mbox_path in mbox_files:
        print(f"Processing {mbox_path.name}...")
        records, skipped = load_mbox_file(mbox_path)
        print(f"  -> {len(records)} usable emails, {skipped} skipped")
        all_records.extend(records)

    # Add the phishing label - every email in Nazario is phishing
    df = pd.DataFrame(all_records)
    df["label"] = "phishing"

    # Save to interim
    output_path = DATA_INTERIM / "nazario_phishing.csv"
    df.to_csv(output_path, index=False)

    print()
    print(f"=== Summary ===")
    print(f"Total usable emails: {len(df)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()