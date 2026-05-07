"""
Build phoneme vocabulary từ toàn bộ dataset đã normalize.

Output: src/vi_phoneme_tokenizer.json

Usage:
    python scripts/03_build_tokenizer.py
        --metadata data/normalized/metadata_norm.csv
        --output src/vi_phoneme_tokenizer.json
        --dialect s --tone_format pham
"""
import argparse
import csv
import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.vi_text_processor import vi_text_to_phonemes
from src.phoneme_tokenizer import (
    PhonemeTokenizer, build_vocab_from_corpus, SPECIAL_TOKENS
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str,
                        default="data/normalized/metadata_norm.csv")
    parser.add_argument("--output", type=str,
                        default="src/vi_phoneme_tokenizer.json")
    parser.add_argument("--dialect", type=str, default="s",
                        choices=["s", "n", "c", "o"])
    parser.add_argument("--tone_format", type=str, default="pham",
                        choices=["pham", "chao", "cao", "super"])
    parser.add_argument("--min_freq", type=int, default=2,
                        help="Phoneme phải xuất hiện >= n lần để vào vocab")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="-1 = dùng hết, hoặc giới hạn để test nhanh")
    args = parser.parse_args()

    if not os.path.exists(args.metadata):
        print(f"[ERROR] Metadata not found: {args.metadata}")
        sys.exit(1)

    print(f"Building phoneme vocab")
    print(f"  Dialect: {args.dialect}")
    print(f"  Tone format: {args.tone_format}")
    print(f"  Min frequency: {args.min_freq}")
    print()

    # Read all normalized texts + run G2P
    phoneme_lines = []
    n_errors = 0

    with open(args.metadata, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        rows = list(reader)

    if args.max_samples > 0:
        rows = rows[:args.max_samples]

    print(f"Running G2P on {len(rows)} samples...")
    for row in tqdm(rows):
        if len(row) < 3:
            continue
        norm_text = row[2]

        try:
            phon = vi_text_to_phonemes(
                norm_text, dialect=args.dialect, tone_format=args.tone_format
            )
            if phon:
                phoneme_lines.append(phon)
        except Exception as e:
            n_errors += 1
            continue

    print(f"\nG2P done. {len(phoneme_lines)} valid lines, {n_errors} errors.")

    # Build vocab
    vocab = build_vocab_from_corpus(phoneme_lines, min_freq=args.min_freq)

    # Save
    tokenizer = PhonemeTokenizer(vocab)
    tokenizer.save(args.output)

    print(f"\n{'='*60}")
    print(f"Tokenizer saved to: {args.output}")
    print(f"Vocab size: {len(vocab)}")
    print()
    print(f"⚠️  CHỈNH src/config.py:")
    print(f"     new_vocab_size = {len(vocab)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
