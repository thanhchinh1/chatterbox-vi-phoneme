import re
import sys
import os
from pathlib import Path

# vPhon được clone vào tools/vPhon
_VPHON_PATH = os.path.join(os.path.dirname(__file__), "..", "tools", "vPhon")
if _VPHON_PATH not in sys.path:
    sys.path.insert(0, os.path.abspath(_VPHON_PATH))

try:
    from vPhon import convert as _vphon_convert
except ImportError:
    _vphon_convert = None
    print("[WARNING] vPhon not found. Run: git clone https://github.com/kirbyj/vPhon.git tools/vPhon")


# ============================================================
# 1. PUNCTUATION NORMALIZATION
# ============================================================

_VI_CHARS = (
    "aàáảãạăằắẳẵặâầấẩẫậ"
    "eèéẻẽẹêềếểễệ"
    "iìíỉĩị"
    "oòóỏõọôồốổỗộơờớởỡợ"
    "uùúủũụưừứửữự"
    "yỳýỷỹỵ"
    "đ"
)

_PUNCT_KEEP = ".,!?;:"


def vi_punc_norm(text: str) -> str:
    """Chuẩn hoá dấu câu: xoá ký tự lạ, lowercase, giữ punctuation cần thiết."""
    text = text.lower().strip()

    # Bỏ ngoặc, dấu gạch ngang dài, em-dash...
    text = re.sub(r"[\(\)\[\]\{\}\<\>«»\"\'`]", " ", text)
    text = re.sub(r"[—–‒]", " ", text)

    # Giữ chữ Việt + số + punctuation cần thiết, còn lại thay bằng space
    pattern = rf"[^a-z0-9{_VI_CHARS}{re.escape(_PUNCT_KEEP)}\s]"
    text = re.sub(pattern, " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# 2. NUMBER NORMALIZATION
# ============================================================

_VI_DIGITS = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín",
}


def _read_number_vi(n: int) -> str:
    """Convert integer to Vietnamese words (basic implementation)."""
    if n == 0:
        return "không"
    if n < 0:
        return "âm " + _read_number_vi(-n)

    # TODO: implement đầy đủ với num2words hoặc tự viết logic
    # Dùng num2words tạm thời (có thể thay bằng logic VN tốt hơn)
    try:
        from num2words import num2words
        return num2words(n, lang="vi")
    except ImportError:
        # Fallback: đọc từng số
        return " ".join(_VI_DIGITS[d] for d in str(n))


def expand_units_after_number(text: str) -> str:
    """Expand đơn vị đo đứng cạnh số: "5km" → "5 ki lô mét", "1h" → "1 giờ".

    Phải chạy TRƯỚC normalize_numbers vì cần phát hiện pattern <digit><unit>.

    Xử lý 2 dạng:
      - Liền: "5km", "1h", "200gb"
      - Cách space: "5 km", "1 h", "200 gb"

    Multi-char unit (km, kg, gb...) lấy từ _ABBREV.
    Single-char unit (m, g, h, s, l) lấy từ _SINGLE_CHAR_UNITS (chỉ expand khi sau số).
    """
    # Build pattern alternation từ multi-char trong _ABBREV (chỉ phần đơn vị đo)
    multi_char_units = ["km", "kg", "cm", "mm", "ml", "mg", "gb", "mb", "kb", "tb"]
    single_char_units = list(_SINGLE_CHAR_UNITS.keys())

    # Sort theo độ dài giảm dần để regex match dài nhất trước (km trước k)
    all_units_sorted = sorted(multi_char_units + single_char_units, key=len, reverse=True)
    units_pattern = "|".join(re.escape(u) for u in all_units_sorted)

    # Pattern: số (có thể có dấu phẩy/chấm) + (optional space) + unit + word boundary
    #  ở cuối quan trọng để không match "10minute" → "10 phút inute"
    pattern = re.compile(
        r"(\d+(?:[.,]\d+)?)\s*(" + units_pattern + r")\b",
        flags=re.IGNORECASE,
    )

    def repl(m):
        num = m.group(1)
        unit = m.group(2).lower()
        if unit in _SINGLE_CHAR_UNITS:
            unit_word = _SINGLE_CHAR_UNITS[unit]
        elif unit in _ABBREV:
            unit_word = _ABBREV[unit]
        else:
            unit_word = unit
        return f"{num} {unit_word}"

    text = pattern.sub(repl, text)
    return text


def normalize_numbers(text: str) -> str:
    """Chuyển các con số trong text thành chữ.
    Lưu ý: đây là logic đơn giản, chưa xử lý ngày tháng, tiền, đơn vị.
    Xem scripts/02_normalize_text.py để có version nâng cao.
    """
    def repl(match):
        num_str = match.group(0)
        try:
            n = int(num_str)
            return " " + _read_number_vi(n) + " "
        except ValueError:
            return num_str

    text = re.sub(r"\d+", repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# 3. ABBREVIATIONS  (Định nghĩa TRƯỚC expand_units_after_number)
# ============================================================

_ABBREV = {
    # Hành chính
    "tp": "thành phố",
    "tp.": "thành phố",
    "q.": "quận",
    "p.": "phường",
    "h.": "huyện",
    "tx.": "thị xã",
    "đ.": "đường",
    "tt": "thị trấn",
    "vn": "việt nam",
    "tphcm": "thành phố hồ chí minh",
    "hcm": "hồ chí minh",
    "hn": "hà nội",

    # Học vị / chức danh
    "ts.": "tiến sĩ",
    "ths.": "thạc sĩ",
    "gs.": "giáo sư",
    "pgs.": "phó giáo sư",
    "bs.": "bác sĩ",
    "ks.": "kỹ sư",
    "ô.": "ông",
    "b.": "bà",
    "a.": "anh",
    "c.": "chị",

    # Đơn vị đo (multi-char SAFE — không gây ambiguity với tên riêng)
    "km": "ki lô mét",
    "kg": "ki lô gam",
    "cm": "xăng ti mét",
    "mm": "mi li mét",
    "ml": "mi li lít",
    "mg": "mi li gam",

    # Đơn vị tin học
    "gb": "gi ga bai",
    "mb": "mê ga bai",
    "kb": "ki lô bai",
    "tb": "tê ra bai",

    # Đời sống
    "tv": "ti vi",
    "vtv": "vê tê vê",
    "htv": "hát tê vê",

    # Tiền tệ
    "đ": "đồng",
    "vnđ": "việt nam đồng",
    "usd": "đô la mỹ",
    "eur": "ơ rô",
}

# Đơn vị/chữ cái 1-ký-tự — CHỈ expand khi đi sau số
_SINGLE_CHAR_UNITS = {
    "m": "mét",
    "g": "gam",
    "l": "lít",
    "h": "giờ",
    "s": "giây",
}


def expand_abbreviations(text: str) -> str:
    """Mở rộng viết tắt phổ biến."""
    words = text.split()
    out = []
    for w in words:
        key = w.lower()
        if key in _ABBREV:
            out.append(_ABBREV[key])
        else:
            out.append(w)
    return " ".join(out)


# ============================================================
# 4. G2P qua vPhon
# ============================================================

# Mapping tone format → vPhon flags
# vPhon convert(word, dialect, chao, eight, nosuper, glottal, phonemic, delimit)
def _vphon_args(tone_format: str):
    """Return tuple (chao, eight, nosuper) cho vPhon.convert().

    vPhon tone formats:
      - "letter" (default, KHUYẾN NGHỊ): output dạng chữ + số "A1, A2, B1, B2, C1, C2, D1, D2".
                  8 tokens phân biệt syllable mở (A/B/C) và khép (D).
                  Đây là cái thực tế vPhon trả về khi --nosuper bật mà không có flag khác.
      - "chao":   Chao tone numbers "33, 24, 32, 21..." (cao độ thực tế, 2-char).
      - "eight":  số 1-8 (sắc + coda /p t k/ → 7, nặng + coda → 8).
      - "super":  superscript Unicode mặc định "ᴬ¹, ᴮ²" (không khuyến nghị cho TTS).
    """
    if tone_format == "letter":
        # Output: A1/A2/B1/B2/C1/C2/D1/D2 — 8 tone tokens
        return (False, False, True)
    elif tone_format == "chao":
        return (True, False, True)
    elif tone_format == "eight":
        return (False, True, True)
    elif tone_format == "super":
        return (False, False, False)
    else:
        raise ValueError(f"Unknown tone_format: {tone_format!r}. "
                         f"Use 'letter' (recommended), 'chao', 'eight', or 'super'.")


def vi_g2p(text: str, dialect: str = "s", tone_format: str = "letter",
           syllable_sep: str = " | ", phoneme_sep: str = " ") -> str:
    """Convert Vietnamese text → phoneme sequence.

    Args:
        text: input đã được normalize.
        dialect: 's' (Nam), 'n' (Bắc), 'c' (Trung), 'o' (orthographic).
        tone_format: 'letter' (A1-D2) khuyến nghị.
        syllable_sep: ký hiệu phân cách giữa các âm tiết.
        phoneme_sep: ký hiệu phân cách giữa các phoneme trong cùng âm tiết.

    Returns:
        Chuỗi phoneme. Ví dụ: "s i n 1 | c a w 2"
    """
    if _vphon_convert is None:
        raise RuntimeError("vPhon not available. Clone it to tools/vPhon/")

    chao, eight, nosuper = _vphon_args(tone_format)

    syllables = text.split()
    out = []
    for syl in syllables:
        # Bỏ các âm tiết chỉ là punctuation
        if all(c in _PUNCT_KEEP for c in syl):
            continue

        # Tách punctuation đuôi (ví dụ "chào." → "chào", ".")
        trailing_punct = ""
        while syl and syl[-1] in _PUNCT_KEEP:
            trailing_punct = syl[-1] + trailing_punct
            syl = syl[:-1]

        if not syl:
            continue

        try:
            # convert(word, dialect, chao, eight, nosuper, glottal, phonemic, delimit)
            ipa = _vphon_convert(syl, dialect, chao, eight, nosuper, False, False, phoneme_sep)
            ipa = ipa.strip()
            if ipa:
                out.append(ipa)
        except Exception as e:
            # Âm tiết không phải tiếng Việt → bỏ qua hoặc giữ raw
            # Có thể log để debug
            print(f"[vi_g2p] Cannot phonetize '{syl}': {e}")
            continue

    return syllable_sep.join(out)


# ============================================================
# FULL PIPELINE
# ============================================================

def vi_text_to_phonemes(text: str, dialect: str = "s",
                        tone_format: str = "letter") -> str:
    r"""Pipeline đầy đủ: raw text → phoneme sequence.

    Order matters:
    1. Lowercase + bỏ ký tự lạ
    2. Expand đơn vị đo SAU SỐ (5km → "5 ki lô mét") — phải làm TRƯỚC (3)
    vì normalize_numbers dùng regex r'\d+' sẽ match số còn lại sau khi expand
    3. Số → chữ ("5 ki lô mét" → "năm ki lô mét")
    4. Expand viết tắt còn lại (tv, hcm, đ.)
    5. G2P
    """
    text = text.lower().strip()
    text = expand_units_after_number(text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    text = vi_punc_norm(text)
    phonemes = vi_g2p(text, dialect=dialect, tone_format=tone_format)
    return phonemes


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    test_sentences = [
        "Xin chào, tôi là trợ lý ảo.",
        "Hôm nay là ngày 15 tháng 3 năm 2024.",
        "TP.HCM có khoảng 9 triệu dân.",
        "Cảm ơn bạn đã sử dụng dịch vụ!",
    ]

    print(f"Dialect: Southern (Saigon)")
    print(f"Tone format: Pham (1-6)")
    print("=" * 60)

    for sent in test_sentences:
        print(f"\nInput:  {sent}")
        normalized = vi_punc_norm(expand_abbreviations(normalize_numbers(sent.lower())))
        print(f"Norm:   {normalized}")
        try:
            phonemes = vi_text_to_phonemes(sent, dialect="s", tone_format="letter")
            print(f"Phon:   {phonemes}")
        except RuntimeError as e:
            print(f"Phon:   [ERROR] {e}")