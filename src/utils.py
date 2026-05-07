"""Utility functions: logger, VAD trim, model file checks."""
import logging
import os
import sys
import numpy as np
from typing import Optional


def setup_logger(name: str = "Chatterbox-VI", level: int = logging.INFO) -> logging.Logger:
    """Setup logger với format gọn."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def check_pretrained_models(model_dir: str = "pretrained_models") -> bool:
    """Verify pretrained Chatterbox files đã được tải."""
    required = [
        "ve.safetensors",
        "t3_cfg.safetensors",
        "s3gen.safetensors",
        "conds.pt",
        "tokenizer.json",
    ]
    missing = []
    for f in required:
        path = os.path.join(model_dir, f)
        if not os.path.exists(path):
            missing.append(f)

    if missing:
        print(f"[ERROR] Missing pretrained files in {model_dir}/:")
        for f in missing:
            print(f"  - {f}")
        print(f"\nRun: python setup.py")
        return False
    return True


def trim_silence_with_vad(wav: np.ndarray, sr: int = 24000,
                          threshold: float = 0.5,
                          min_silence_ms: int = 200) -> np.ndarray:
    """Trim leading/trailing silence dùng Silero VAD.

    Args:
        wav: 1-D numpy array float32 [-1, 1].
        sr: sample rate.
        threshold: VAD threshold (0.0-1.0).
        min_silence_ms: minimum silence duration để consider trim.

    Returns:
        Trimmed wav.
    """
    try:
        import torch
        from silero_vad import load_silero_vad, get_speech_timestamps
    except ImportError:
        print("[trim_silence_with_vad] silero-vad not installed, skipping")
        return wav

    model = load_silero_vad()
    wav_t = torch.from_numpy(wav).float()

    timestamps = get_speech_timestamps(
        wav_t, model, sampling_rate=sr,
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
    )

    if not timestamps:
        return wav

    start = timestamps[0]["start"]
    end = timestamps[-1]["end"]

    # Pad nhẹ để không cắt quá sát
    pad_samples = int(0.05 * sr)  # 50ms
    start = max(0, start - pad_samples)
    end = min(len(wav), end + pad_samples)

    return wav[start:end]


def loudness_normalize(wav: np.ndarray, sr: int = 24000,
                       target_lufs: float = -23.0) -> np.ndarray:
    """Normalize loudness về target LUFS (chuẩn EBU R128).

    Yêu cầu: pip install pyloudnorm
    """
    try:
        import pyloudnorm as pyln
    except ImportError:
        return wav

    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(wav)
        if loudness > -70:  # tránh chia 0 trên file gần silence
            wav = pyln.normalize.loudness(wav, loudness, target_lufs)
    except Exception as e:
        print(f"[loudness_normalize] Error: {e}")
    return wav


def count_parameters(model) -> dict:
    """Đếm trainable + total params."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": 100 * trainable / total if total > 0 else 0,
    }
