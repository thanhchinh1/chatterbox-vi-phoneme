"""
Sanity check vPhon output cho dialect Nam + Pham tones.

Chạy script này TRƯỚC khi build tokenizer để verify G2P hoạt động đúng.

Usage:
    python scripts/04_test_g2p.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.vi_text_processor import vi_text_to_phonemes, vi_g2p, vi_punc_norm


TEST_CASES = [
    # (input, expected_pattern_or_description)
    ("xin chào", "Greeting cơ bản"),
    ("hà nội", "Tone huyền + nặng"),
    ("hồ chí minh", "Multiple syllables"),
    ("ăn uống", "ă diphthong"),
    ("nguyễn", "Complex onset (ngu)"),
    ("phở bò", "ph onset + diphthong"),
    ("tiếng việt", "Triphthong"),
    ("trường", "tr cluster (Southern: ʈ)"),
    ("cám ơn", "ơ vowel"),
    ("không sao", "Negation"),
    ("một hai ba", "Numbers as words"),
    ("đẹp", "đ + p coda"),
    ("anh yêu em", "Common phrase"),
    ("chúc mừng năm mới", "Holiday greeting"),
    ("sài gòn", "Southern dialect specific"),
]


def main():
    print("=" * 70)
    print("vPhon SANITY CHECK")
    print(f"Dialect: Southern (Saigon)")
    print(f"Tone format: Pham (1-6)")
    print("=" * 70)

    n_ok = 0
    n_fail = 0

    for text, desc in TEST_CASES:
        print(f"\n[{desc}]")
        print(f"  Input:  '{text}'")

        try:
            normalized = vi_punc_norm(text)
            print(f"  Norm:   '{normalized}'")

            # Test với dialect Nam, Pham tone
            phon_s = vi_g2p(normalized, dialect="s", tone_format="letter")
            print(f"  Phon:   '{phon_s}'")

            # So sánh với dialect Bắc để xem khác biệt
            phon_n = vi_g2p(normalized, dialect="n", tone_format="letter")
            if phon_s != phon_n:
                print(f"  [Bắc:   '{phon_n}']")

            n_ok += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            n_fail += 1

    print("\n" + "=" * 70)
    print(f"Result: {n_ok} OK, {n_fail} FAILED")

    # Liệt kê tất cả phoneme unique trong test set
    print("\n[Inventory check]")
    all_tokens = set()
    for text, _ in TEST_CASES:
        try:
            phon = vi_text_to_phonemes(text, dialect="s", tone_format="letter")
            for tok in phon.split():
                all_tokens.add(tok)
        except:
            pass

    sorted_tokens = sorted(all_tokens)
    print(f"  Unique tokens in test set: {len(sorted_tokens)}")
    print(f"  Tokens: {sorted_tokens}")


if __name__ == "__main__":
    main()
