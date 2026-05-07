"""
Dataset cho Chatterbox Vietnamese fine-tuning.

Format LJSpeech mong đợi:
  data/
    metadata.csv         # filename|raw_text|normalized_text (3 cột, separator='|')
    wavs/
      000001.wav
      000002.wav
      ...

Dataset này preprocess on-the-fly: text → phoneme → token IDs.
Audio đã được resample sang 24kHz từ trước (xem scripts/05_prepare_dataset.py).
"""
import os
import csv
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional

from src.phoneme_tokenizer import PhonemeTokenizer
from src.vi_text_processor import vi_text_to_phonemes


class ChatterboxViDataset(Dataset):
    """Vietnamese phoneme-based dataset cho fine-tune T3."""

    def __init__(self, cfg, tts_engine=None):
        """
        Args:
            cfg: TrainConfig instance.
            tts_engine: optional ChatterboxTTS engine để compute speech tokens
                       on-the-fly. Nếu None, expect data đã được preprocessed
                       (có file .pt chứa speech tokens cạnh mỗi .wav).
        """
        self.cfg = cfg
        self.tts_engine = tts_engine
        self.tokenizer = PhonemeTokenizer.load(cfg.tokenizer_path)

        # Load metadata
        self.entries = self._load_metadata()
        print(f"[Dataset] Loaded {len(self.entries)} samples")

    def _load_metadata(self) -> List[Dict]:
        entries = []
        with open(self.cfg.metadata_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if len(row) < 2:
                    continue
                # filename | raw_text | (optional) normalized_text
                fn = row[0].strip()
                raw = row[1].strip()
                norm = row[2].strip() if len(row) > 2 else raw

                wav_path = os.path.join(self.cfg.wavs_dir, fn)
                if not wav_path.endswith(".wav"):
                    wav_path += ".wav"

                if os.path.exists(wav_path):
                    entries.append({
                        "wav_path": wav_path,
                        "raw_text": raw,
                        "norm_text": norm,
                    })
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx) -> Dict:
        entry = self.entries[idx]

        # === Text → phoneme tokens ===
        try:
            phoneme_str = vi_text_to_phonemes(
                entry["norm_text"],
                dialect=self.cfg.dialect,
                tone_format=self.cfg.tone_format,
            )
            text_token_ids = self.tokenizer.encode(phoneme_str, add_special_tokens=True)
        except Exception as e:
            print(f"[Dataset] G2P error on '{entry['norm_text']}': {e}")
            # Return next sample (đệ quy đơn giản)
            return self[(idx + 1) % len(self)]

        # === Audio ===
        wav, sr = torchaudio.load(entry["wav_path"])
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # mono

        wav = wav.squeeze(0)
        duration = wav.shape[0] / self.cfg.sample_rate

        # Filter out by length
        if duration < self.cfg.min_audio_length or duration > self.cfg.max_audio_length:
            return self[(idx + 1) % len(self)]

        # === Speech tokens (precomputed hoặc compute on-the-fly) ===
        speech_tokens_path = entry["wav_path"].replace(".wav", ".speech_tokens.pt")
        if os.path.exists(speech_tokens_path):
            speech_tokens = torch.load(speech_tokens_path, weights_only=True)
        elif self.tts_engine is not None:
            # Compute on-the-fly (chậm, nên preprocess trước)
            speech_tokens = self._compute_speech_tokens(wav)
        else:
            raise FileNotFoundError(
                f"Speech tokens not precomputed: {speech_tokens_path}\n"
                f"Run scripts/06_precompute_speech_tokens.py first."
            )

        # === Speaker conditioning embedding ===
        cond_path = entry["wav_path"].replace(".wav", ".cond.pt")
        if os.path.exists(cond_path):
            cond_emb = torch.load(cond_path, weights_only=True)
        elif self.tts_engine is not None:
            cond_emb = self._compute_cond(wav)
        else:
            cond_emb = None  # T3 sẽ dùng default

        return {
            "text_tokens": torch.tensor(text_token_ids, dtype=torch.long),
            "speech_tokens": speech_tokens,
            "cond_emb": cond_emb,
            "audio_length": duration,
        }

    def _compute_speech_tokens(self, wav):
        """Compute speech tokens dùng S3Gen tokenizer (on-the-fly)."""
        with torch.no_grad():
            # API có thể khác tuỳ version Chatterbox, sửa cho khớp
            tokens = self.tts_engine.s3gen.tokenize(wav.unsqueeze(0))
        return tokens.squeeze(0)

    def _compute_cond(self, wav):
        """Compute speaker conditioning embedding dùng VoiceEncoder."""
        with torch.no_grad():
            emb = self.tts_engine.ve(wav.unsqueeze(0))
        return emb.squeeze(0)


# ============================================================
# DATA COLLATOR
# ============================================================

def data_collator_phoneme(batch: List[Dict]) -> Dict:
    """Pad text + speech tokens ở batch level.

    Returns dict tương thích với T3 forward signature.
    """
    # Filter None (sample bị skip)
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    text_lens = [len(b["text_tokens"]) for b in batch]
    speech_lens = [len(b["speech_tokens"]) for b in batch]

    max_text = max(text_lens)
    max_speech = max(speech_lens)

    bs = len(batch)
    text_padded = torch.zeros(bs, max_text, dtype=torch.long)
    speech_padded = torch.zeros(bs, max_speech, dtype=torch.long)
    text_mask = torch.zeros(bs, max_text, dtype=torch.bool)
    speech_mask = torch.zeros(bs, max_speech, dtype=torch.bool)

    cond_embs = []

    for i, b in enumerate(batch):
        tl = text_lens[i]
        sl = speech_lens[i]
        text_padded[i, :tl] = b["text_tokens"]
        speech_padded[i, :sl] = b["speech_tokens"]
        text_mask[i, :tl] = True
        speech_mask[i, :sl] = True

        if b["cond_emb"] is not None:
            cond_embs.append(b["cond_emb"])

    output = {
        "text_tokens": text_padded,
        "text_mask": text_mask,
        "speech_tokens": speech_padded,
        "speech_mask": speech_mask,
        "text_lengths": torch.tensor(text_lens, dtype=torch.long),
        "speech_lengths": torch.tensor(speech_lens, dtype=torch.long),
    }

    if len(cond_embs) == bs:
        output["cond_emb"] = torch.stack(cond_embs)

    return output
