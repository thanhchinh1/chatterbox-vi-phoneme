"""
Embedding warmup: khởi tạo embedding của các phoneme token mới
bằng trung bình embedding của các grapheme token tương tự trong tokenizer cũ.

Vì sao quan trọng:
  - Random init → loss khởi điểm rất cao (~10-12), training phải mất hàng nghìn
    steps chỉ để model "biết" rằng các token là chữ.
  - Init thông minh → loss khởi điểm ~7-8, hội tụ nhanh hơn 2-3x.

Cách làm:
  Mỗi phoneme IPA → map sang một hoặc nhiều grapheme tương tự trong tokenizer cũ.
  Embedding mới = trung bình embedding các grapheme đó.

Ví dụ:
  /a/  → "a"           → emb[id_of_"a"]
  /ɲ/  → "n", "h"      → mean(emb["n"], emb["h"])  # vì /ɲ/ ≈ "nh" trong tiếng Việt
  /1/  → tone 1, gán random nhỏ vì không có tương đương
"""
import json
import os
import torch
from typing import Dict, List, Optional


# ============================================================
# DEFAULT IPA → GRAPHEME MAPPING (Southern Vietnamese)
# ============================================================

# Mapping này bạn có thể tinh chỉnh sau khi xem vocab thực tế.
# Format: phoneme → list các ký tự/chuỗi grapheme tương tự
# (sẽ encode bằng tokenizer cũ rồi lấy mean embedding)
DEFAULT_IPA_TO_GRAPHEME = {
    # ===== Vowels =====
    "a": ["a"],
    "aː": ["a"],
    "ă": ["ă", "a"],
    "ə": ["ơ", "â"],
    "əː": ["ơ", "â"],
    "ɤ": ["ơ"],
    "ɛ": ["e"],
    "ɛː": ["ê", "e"],
    "e": ["ê", "e"],
    "eː": ["ê", "e"],
    "i": ["i", "y"],
    "iː": ["i", "y"],
    "ɔ": ["o"],
    "ɔː": ["ô", "o"],
    "o": ["ô", "o"],
    "oː": ["ô", "o"],
    "u": ["u"],
    "uː": ["u"],
    "ɯ": ["ư"],
    "ɨ": ["ư"],
    "ɨː": ["ư"],

    # ===== Diphthongs =====
    "iə": ["iê", "ia"],
    "uə": ["ua", "uâ"],
    "ɨə": ["ưa", "ưa"],

    # ===== Consonants - Onsets =====
    "b": ["b"],
    "ɓ": ["b"],
    "m": ["m"],
    "f": ["ph", "f"],
    "v": ["v"],
    "w": ["o", "u"],
    "t": ["t"],
    "tʰ": ["th"],
    "d": ["d", "đ"],
    "ɗ": ["đ", "d"],
    "n": ["n"],
    "s": ["x", "s"],
    "z": ["d", "gi"],
    "l": ["l"],
    "r": ["r"],
    "ʈ": ["tr"],
    "ʂ": ["s"],
    "ʐ": ["r"],
    "c": ["ch"],
    "ɲ": ["nh"],
    "j": ["i", "y"],
    "k": ["k", "c", "q"],
    "x": ["kh"],
    "ɣ": ["g", "gh"],
    "ɡ": ["g", "gh"],
    "ŋ": ["ng", "ngh"],
    "ŋ͡m": ["ng", "m"],
    "h": ["h"],
    "ʔ": ["a"],

    # ===== Codas / glide-like tokens =====
    "p": ["p"],
    "p̚": ["p"],
    "t̚": ["t"],
    "k̚": ["c", "k"],
    "k͡p": ["c", "k", "p"],
    "j̆": ["i", "y"],
    "w̆": ["u", "o"],

    # ===== Tones (current vocab uses letter tones) =====
    # Không có tương đương grapheme → để rỗng, sẽ init nhỏ random
    "A1": [],
    "A2": [],
    "B1": [],
    "B2": [],
    "C1": [],
    "C2": [],
    "D1": [],
    "D2": [],

    # Legacy aliases kept for compatibility with older configs
    "1": [],
    "2": [],
    "3": [],
    "4": [],
    "5": [],
    "6": [],

    # ===== Special =====
    "|": [" "],
    "<sil>": [],
}


def warmup_embedding(
    new_emb: torch.Tensor,
    old_emb: torch.Tensor,
    new_vocab: Dict[str, int],
    old_tokenizer,
    ipa_to_grapheme: Optional[Dict[str, List[str]]] = None,
    no_match_init_std: float = 0.02,
) -> torch.Tensor:
    """Init `new_emb` (vocab mới) từ `old_emb` (vocab cũ).

    Args:
        new_emb: tensor [new_vocab_size, hidden_dim] - sẽ được modify in-place.
        old_emb: tensor [old_vocab_size, hidden_dim] từ pretrained model.
        new_vocab: dict {phoneme: id} của tokenizer mới.
        old_tokenizer: tokenizer cũ của Chatterbox (để encode grapheme).
        ipa_to_grapheme: mapping {phoneme: [grapheme_str,...]}. Default dùng VN.
        no_match_init_std: std cho random init khi không có grapheme tương ứng.

    Returns:
        new_emb đã được init.
    """
    if ipa_to_grapheme is None:
        ipa_to_grapheme = DEFAULT_IPA_TO_GRAPHEME

    n_warm = 0
    n_random = 0
    n_skip = 0

    # Tính mean+std của old embedding để random init "khớp scale"
    old_mean = old_emb.mean().item()
    old_std = old_emb.std().item()

    for phoneme, new_id in new_vocab.items():
        # Special tokens dùng init mặc định (có thể giữ nguyên random)
        if phoneme in ["<pad>", "<bos>", "<eos>", "<unk>"]:
            new_emb[new_id].normal_(mean=old_mean, std=old_std * 0.5)
            n_skip += 1
            continue

        candidates = ipa_to_grapheme.get(phoneme, None)

        if (candidates is None or len(candidates) == 0) and phoneme.startswith("[") and phoneme.endswith("]"):
            inner_text = phoneme[1:-1].strip()
            if inner_text:
                candidates = [inner_text]

        if candidates is None or len(candidates) == 0:
            # Không có mapping → random nhỏ matching scale
            new_emb[new_id].normal_(mean=old_mean, std=old_std * no_match_init_std / 0.02)
            n_random += 1
            continue

        # Encode từng candidate, lấy embedding, average
        all_embs = []
        for grapheme_str in candidates:
            try:
                # Tokenizer Chatterbox gốc dùng grapheme-based, encode trả về list of IDs
                # Tuỳ implementation, có thể là encode().ids hoặc tokenize()+convert
                if hasattr(old_tokenizer, "encode"):
                    encoded = old_tokenizer.encode(grapheme_str)
                    if hasattr(encoded, "ids"):
                        ids = encoded.ids
                    else:
                        ids = encoded
                else:
                    # Fallback: assume callable trả về IDs
                    ids = old_tokenizer(grapheme_str)

                for tok_id in ids:
                    if 0 <= tok_id < old_emb.shape[0]:
                        all_embs.append(old_emb[tok_id])
            except Exception as e:
                print(f"[warmup] Cannot encode '{grapheme_str}' for /{phoneme}/: {e}")
                continue

        if all_embs:
            new_emb[new_id] = torch.stack(all_embs).mean(dim=0)
            n_warm += 1
        else:
            new_emb[new_id].normal_(mean=old_mean, std=old_std * 0.5)
            n_random += 1

    print(f"[Embedding Warmup]")
    print(f"  Warmed (from grapheme): {n_warm}")
    print(f"  Random init:            {n_random}")
    print(f"  Special tokens:         {n_skip}")
    print(f"  Total:                  {len(new_vocab)}")

    return new_emb


def save_default_mapping(path: str):
    """Save default IPA→grapheme map ra JSON để bạn có thể tinh chỉnh."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_IPA_TO_GRAPHEME, f, ensure_ascii=False, indent=2)
    print(f"[save_default_mapping] Saved to {path}")


def load_mapping(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # Lưu default mapping ra để user tinh chỉnh
    save_default_mapping("src/phoneme_to_grapheme_init.json")
    print("\nĐã tạo file mapping mặc định.")
    print("Bạn có thể sửa src/phoneme_to_grapheme_init.json để tinh chỉnh init.")
