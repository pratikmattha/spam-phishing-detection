"""
Extract the five SpamAssassin .tar.bz2 archives into data/interim/spamassassin_extracted/.
Each archive becomes its own subfolder named after the label (easy_ham, spam, etc.).
"""

import tarfile
import sys
from pathlib import Path

# Make config.py importable
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import DATA_RAW, DATA_INTERIM


def extract_archive(archive_path, output_root):
    """Extract one .tar.bz2 archive into output_root."""
    print(f"Extracting {archive_path.name}...")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(path=output_root)
    print(f"  -> done")


def main():
    archive_dir = DATA_RAW / "spamassassin"
    output_dir = DATA_INTERIM / "spamassassin_extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    archives = sorted(archive_dir.glob("*.tar.bz2"))
    print(f"Found {len(archives)} archives to extract:")
    for a in archives:
        print(f"  - {a.name}")
    print()

    for archive in archives:
        extract_archive(archive, output_dir)

    print()
    print(f"=== Extraction complete ===")
    print(f"Output directory: {output_dir}")
    print()
    print("Contents:")
    for item in sorted(output_dir.iterdir()):
        if item.is_dir():
            file_count = sum(1 for _ in item.iterdir())
            print(f"  {item.name}/  ({file_count} files)")


if __name__ == "__main__":
    main()