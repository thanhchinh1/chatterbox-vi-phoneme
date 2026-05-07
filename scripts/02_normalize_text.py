"""
Normalize text trong metadata: số → chữ, viết tắt, lowercase, bỏ ký tự lạ.

Output: metadata_norm.csv với cột thứ 3 là normalized text.

Usage:
    python scripts/02_normalize_text.py --input data/raw --output data/normalized
"""
import argparse
import csv
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Import từ src
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.vi_text_processor import vi_punc_norm, normalize_numbers, expand_abbreviations


def normalize_full(text: str) -> str:
    """Apply tất cả normalize steps."""
    text = text.lower().strip()
    text = expand_abbreviations(text)
    text = normalize_numbers(text)
    text = vi_punc_norm(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/normalized")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_meta = in_dir / "metadata_raw.csv"
    out_meta = out_dir / "metadata_norm.csv"

    if not in_meta.exists():
        print(f"[ERROR] {in_meta} not found. Run 01_download_dataset.py first.")
        sys.exit(1)

    n_total = 0
    n_kept = 0
    n_skip_empty = 0

    with open(in_meta, "r", encoding="utf-8") as fin, \
         open(out_meta, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin, delimiter="|")
        writer = csv.writer(fout, delimiter="|", quoting=csv.QUOTE_MINIMAL)

        for row in tqdm(reader, desc="Normalizing"):
            n_total += 1
            if len(row) < 2:
                continue

            fname = row[0]
            raw_text = row[1]
            extra = row[2:] if len(row) > 2 else []

            try:
                norm = normalize_full(raw_text)
            except Exception as e:
                print(f"[WARN] {fname}: {e}")
                continue

            if not norm or len(norm) < 5:
                n_skip_empty += 1
                continue

            # Output: filename | raw | normalized | (sr | duration)
            writer.writerow([fname, raw_text, norm] + extra)
            n_kept += 1

    print(f"\nDone!")
    print(f"  Total:   {n_total}")
    print(f"  Kept:    {n_kept}")
    print(f"  Skipped: {n_skip_empty}")
    print(f"  Output:  {out_meta}")


if __name__ == "__main__":
    main()
