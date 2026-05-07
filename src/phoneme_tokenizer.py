"""
Build phoneme-based tokenizer cho tiếng Việt.

Khác với grapheme tokenizer của Chatterbox gốc (~2454 tokens cover 23 ngôn ngữ),
tokenizer này chỉ có ~80-150 tokens — bao gồm:
  - IPA phonemes (consonants + vowels, ~70-90)
  - Tone markers 1-6 (Pham format)
  - Special tokens: <pad>, <bos>, <eos>, <unk>, |, <sil>

Vocab này được build từ chính dataset của bạn (chạy scripts/03_build_tokenizer.py)
để đảm bảo cover hết mọi phoneme xuất hiện thực tế.
"""
import json
import os
from collections import Counter
from typing import List, Dict, Optional


# Special tokens - thứ tự QUAN TRỌNG (id 0-5 reserved)
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "|": 4,        # syllable boundary
    "<sil>": 5,    # silence (cho pause)
}


class PhonemeTokenizer:
    """Tokenizer đơn giản theo word-level (mỗi phoneme là 1 token).

    Tương thích interface với HuggingFace tokenizer cơ bản:
      - encode(text) → token_ids
      - decode(ids) → text
      - __len__ → vocab size
    """

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.pad_id = vocab.get("<pad>", 0)
        self.bos_id = vocab.get("<bos>", 1)
        self.eos_id = vocab.get("<eos>", 2)
        self.unk_id = vocab.get("<unk>", 3)

    def __len__(self):
        return len(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Phoneme string → token IDs.
        
        Input đã phải qua vi_text_to_phonemes(), tức là chuỗi phoneme phân
        cách bằng space, ví dụ: "s i n 1 | c a w 2"
        """
        tokens = text.split()
        ids = [self.vocab.get(t, self.unk_id) for t in tokens]

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Ngược lại: token IDs → phoneme string."""
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        out = []
        for i in ids:
            if skip_special_tokens and i in special_ids:
                continue
            out.append(self.id_to_token.get(i, "<unk>"))
        return " ".join(out)

    def save(self, path: str):
        """Save tokenizer ra JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab": self.vocab,
                "version": "1.0",
                "type": "phoneme_word_level",
            }, f, ensure_ascii=False, indent=2)
        print(f"[PhonemeTokenizer] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PhonemeTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["vocab"])


def build_vocab_from_corpus(phoneme_lines: List[str],
                            min_freq: int = 1) -> Dict[str, int]:
    """Build vocab từ list các chuỗi phoneme đã tokenize.

    Args:
        phoneme_lines: list các string đã qua vi_text_to_phonemes().
        min_freq: chỉ include phoneme xuất hiện ít nhất N lần.

    Returns:
        dict {token: id} với special tokens ở đầu.
    """
    counter = Counter()
    for line in phoneme_lines:
        for tok in line.split():
            counter[tok] += 1

    # Bỏ token đã là special
    for sp in SPECIAL_TOKENS:
        counter.pop(sp, None)

    # Sort theo frequency giảm dần để dễ debug
    sorted_tokens = sorted(counter.items(), key=lambda x: -x[1])

    vocab = dict(SPECIAL_TOKENS)
    next_id = len(SPECIAL_TOKENS)

    skipped = 0
    for tok, freq in sorted_tokens:
        if freq < min_freq:
            skipped += 1
            continue
        vocab[tok] = next_id
        next_id += 1

    print(f"[build_vocab] Total unique tokens: {len(counter)}")
    print(f"[build_vocab] Skipped (freq<{min_freq}): {skipped}")
    print(f"[build_vocab] Final vocab size: {len(vocab)}")
    print(f"[build_vocab] Top 20 most common:")
    for tok, freq in sorted_tokens[:20]:
        print(f"    {tok!r:8s} → {freq}")

    return vocab


if __name__ == "__main__":
    # Test cơ bản
    vocab = {
        "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "|": 4, "<sil>": 5,
        "s": 6, "i": 7, "n": 8, "1": 9, "c": 10, "a": 11, "w": 12, "2": 13,
    }
    tok = PhonemeTokenizer(vocab)
    text = "s i n 1 | c a w 2"
    ids = tok.encode(text)
    print(f"Text: {text}")
    print(f"IDs:  {ids}")
    print(f"Decoded: {tok.decode(ids)}")
