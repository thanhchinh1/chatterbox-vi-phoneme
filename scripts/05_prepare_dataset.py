"""
Final preparation: resample audio sang 24kHz, loudness normalize, VAD trim.
Convert sang LJSpeech format cho training.

Output:
    data/wavs/*.wav (24kHz mono, normalized, trimmed)
    data/metadata.csv

Usage:
    python scripts/05_prepare_dataset.py
        --input data/normalized
        --output data/
"""
import argparse
import csv
import os
import sys
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import trim_silence_with_vad, loudness_normalize


def resample_audio(wav: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return wav
    try:
        import librosa
        return librosa.resample(wav, orig_sr=src_sr, target_sr=target_sr)
    except ImportError:
        # Fallback dùng scipy
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(src_sr, target_sr)
        return resample_poly(wav, target_sr // g, src_sr // g)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/normalized")
    parser.add_argument("--input_wavs", type=str, default="data/raw/wavs")
    parser.add_argument("--output", type=str, default="data/")
    parser.add_argument("--target_sr", type=int, default=24000)
    parser.add_argument("--target_lufs", type=float, default=-23.0)
    parser.add_argument("--apply_vad", action="store_true",
                        help="Trim silence dùng Silero VAD")
    parser.add_argument("--apply_loudness", action="store_true",
                        help="Loudness normalize sang LUFS target")
    args = parser.parse_args()

    in_meta = Path(args.input) / "metadata_norm.csv"
    in_wavs = Path(args.input_wavs)
    out_dir = Path(args.output)
    out_wavs = out_dir / "wavs"
    out_wavs.mkdir(parents=True, exist_ok=True)

    out_meta_path = out_dir / "metadata.csv"

    if not in_meta.exists():
        print(f"[ERROR] {in_meta} not found")
        sys.exit(1)

    n_total = 0
    n_kept = 0
    n_skip_missing = 0
    n_skip_short = 0
    total_duration = 0.0

    with open(in_meta, "r", encoding="utf-8") as fin, \
         open(out_meta_path, "w", encoding="utf-8", newline="") as fout:
        reader = list(csv.reader(fin, delimiter="|"))
        writer = csv.writer(fout, delimiter="|", quoting=csv.QUOTE_MINIMAL)

        for row in tqdm(reader, desc="Processing"):
            n_total += 1
            if len(row) < 3:
                continue

            fname = row[0]
            raw_text = row[1]
            norm_text = row[2]

            in_path = in_wavs / fname
            if not in_path.exists():
                n_skip_missing += 1
                continue

            try:
                wav, sr = sf.read(str(in_path))
                # Mono
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                wav = wav.astype(np.float32)

                # Resample
                wav = resample_audio(wav, sr, args.target_sr)

                # VAD trim (optional, mặc định off để giữ nguyên)
                if args.apply_vad:
                    wav = trim_silence_with_vad(wav, args.target_sr)

                # Loudness normalize
                if args.apply_loudness:
                    wav = loudness_normalize(wav, args.target_sr, args.target_lufs)

                # Clip để tránh distortion sau loudness norm
                wav = np.clip(wav, -1.0, 1.0)

                duration = len(wav) / args.target_sr
                if duration < 1.0:
                    n_skip_short += 1
                    continue

                out_path = out_wavs / fname
                sf.write(str(out_path), wav, args.target_sr)

                writer.writerow([fname, raw_text, norm_text])
                n_kept += 1
                total_duration += duration

            except Exception as e:
                print(f"[WARN] {fname}: {e}")
                continue

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Total:           {n_total}")
    print(f"  Kept:            {n_kept}")
    print(f"  Skip (missing):  {n_skip_missing}")
    print(f"  Skip (too short):{n_skip_short}")
    print(f"  Total duration:  {total_duration/3600:.2f}h")
    print(f"  Sample rate:     {args.target_sr}Hz")
    print(f"  Output meta:     {out_meta_path}")
    print(f"  Output wavs:     {out_wavs}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
