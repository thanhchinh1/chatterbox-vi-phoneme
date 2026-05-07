import argparse
import csv
import os
import re
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.vi_text_processor import (
    vi_punc_norm, normalize_numbers, expand_abbreviations,
    expand_units_after_number, vi_text_to_phonemes
)


def normalize_full(text: str) -> str:
    """Apply tất cả normalize steps.

    Order:
      1. Lowercase
      2. Expand đơn vị đo sau số (5km → 5 ki lô mét) — TRƯỚC normalize_numbers
      3. Số → chữ
      4. Expand viết tắt còn lại
      5. Bỏ ký tự lạ
    """
    text = text.lower().strip()
    text = expand_units_after_number(text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    text = vi_punc_norm(text)
    return text


# Pattern detect OOV trong vPhon output: từ trong dấu ngoặc vuông [foo]
OOV_PATTERN = re.compile(r"\[[^\]]+\]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/normalized")
    parser.add_argument("--dialect", type=str, default="s",
                        choices=["s", "n", "c"])
    parser.add_argument("--tone_format", type=str, default="letter")
    parser.add_argument("--max_oov_ratio", type=float, default=0.0,
                        help="Tỉ lệ OOV/total tokens cho phép (0 = strict)")
    parser.add_argument("--no_filter_oov", action="store_true",
                        help="Tắt OOV filter, giữ mọi sample")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_meta = in_dir / "metadata_raw.csv"
    out_meta = out_dir / "metadata_norm.csv"
    oov_log = out_dir / "oov_samples.txt"

    if not in_meta.exists():
        print(f"[ERROR] {in_meta} not found.")
        sys.exit(1)

    print(f"--- Cấu hình ---")
    print(f"Input:        {in_meta}")
    print(f"Output:       {out_meta}")
    print(f"Dialect:      {args.dialect}")
    print(f"Tone format:  {args.tone_format}")
    print(f"Filter OOV:   {not args.no_filter_oov} "
          f"(max ratio: {args.max_oov_ratio})")
    print(f"----------------\n")

    n_total = 0
    n_kept = 0
    n_skip_empty = 0
    n_skip_oov = 0
    n_g2p_error = 0

    with open(in_meta, "r", encoding="utf-8") as fin, \
         open(out_meta, "w", encoding="utf-8", newline="") as fout, \
         open(oov_log, "w", encoding="utf-8") as flog:
        reader = csv.reader(fin, delimiter="|")
        writer = csv.writer(fout, delimiter="|", quoting=csv.QUOTE_MINIMAL)

        flog.write("# Samples bị skip do OOV\n")
        flog.write("# filename | raw_text | oov_words\n\n")

        for row in tqdm(reader, desc="Normalizing"):
            n_total += 1
            if len(row) < 2:
                continue

            fname = row[0]
            raw_text = row[1]
            extra = row[2:] if len(row) > 2 else []

            try:
                norm = normalize_full(raw_text)
            except Exception:
                continue

            if not norm or len(norm) < 5:
                n_skip_empty += 1
                continue

            if not args.no_filter_oov:
                try:
                    phon = vi_text_to_phonemes(
                        norm,
                        dialect=args.dialect,
                        tone_format=args.tone_format,
                    )
                except Exception:
                    n_g2p_error += 1
                    continue

                if not phon:
                    n_skip_empty += 1
                    continue

                oov_words = OOV_PATTERN.findall(phon)
                total_tokens = len(phon.split())

                if total_tokens == 0:
                    n_skip_empty += 1
                    continue

                oov_ratio = len(oov_words) / max(total_tokens, 1)
                if oov_ratio > args.max_oov_ratio:
                    n_skip_oov += 1
                    flog.write(f"{fname}|{raw_text}|{','.join(oov_words)}\n")
                    continue

            writer.writerow([fname, raw_text, norm] + extra)
            n_kept += 1

    print(f"\n{'='*60}")
    print(f"Hoàn tất!")
    print(f"  Tổng samples:     {n_total}")
    print(f"  Đã giữ:           {n_kept}  ({100*n_kept/max(n_total,1):.1f}%)")
    print(f"  Skip (text ngắn): {n_skip_empty}")
    print(f"  Skip (OOV):       {n_skip_oov}")
    print(f"  Lỗi G2P:          {n_g2p_error}")
    print(f"")
    print(f"  File output:      {out_meta}")
    print(f"  Log OOV samples:  {oov_log}")
    print(f"{'='*60}")

    if n_skip_oov > n_kept * 0.3:
        print(f"\n⚠️  CẢNH BÁO: skip OOV nhiều ({n_skip_oov} samples).")
        print(f"   Cân nhắc: --max_oov_ratio 0.05 (cho phép 5% OOV)")
        print(f"   Hoặc:     --no_filter_oov (giữ mọi sample)")


if __name__ == "__main__":
    main()
