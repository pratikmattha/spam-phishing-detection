"""
Read the first email from a Nazario mbox file and print what's inside.
This is exploration code - not the final loader.
"""

import mailbox
from pathlib import Path

#this import is being used for removing the html tag and provide the simple text from the mail
from bs4 import BeautifulSoup

# Path to the Nazario 2022 file (the smallest one)
mbox_path = Path("data/raw/nazario_phishing/phishing-2022.mbox")

# Open the mailbox
mbox = mailbox.mbox(str(mbox_path))

# Get the very first message
first_message = mbox[0]

def get_text_body(msg):
    """
    Extract plain text body from an email message.
    Tries text/plain first. Falls back to text/html (stripped of tags).
    Returns empty string if no usable text found.
    """
    plain_text = None
    html_text = None

    if msg.is_multipart():
        # Walk through all parts and collect what we find
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
        # Simple message - check its content type
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

    # Prefer plain text. Fall back to HTML stripped of tags.
    if plain_text:
        return plain_text
    if html_text:
        soup = BeautifulSoup(html_text, "lxml")
        return soup.get_text(separator=" ", strip=True)
    return ""

# Print what we found
print("Number of emails in this file:", len(mbox))

print()
print("---- Stats across the whole file ----")
multipart_count = 0
simple_count = 0
for msg in mbox:
    if msg.is_multipart():
        multipart_count += 1
    else:
        simple_count += 1
print(f"Multipart: {multipart_count}")
print(f"Simple: {simple_count}")

print()
print("---- 50th email ----")
fiftieth = mbox[49]
print("From:", fiftieth["From"])
print("Subject:", fiftieth["Subject"])
print()
print("---- Body preview ----")
print(get_text_body(fiftieth)[:500])