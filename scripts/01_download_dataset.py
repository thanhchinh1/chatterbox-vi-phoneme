"""
Download viVoice từ HuggingFace với streaming + filter on-the-fly.
Lưu audio đã filter ra disk để bước sau xử lý.

Usage:
    python scripts/01_download_dataset.py --target_hours 100 --output data/raw
"""
import argparse
import os
import csv
import sys
from pathlib import Path
from tqdm import tqdm

import soundfile as sf
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_hours", type=float, default=100.0,
                        help="Tổng số giờ audio muốn lấy (subset)")
    parser.add_argument("--output", type=str, default="data/raw",
                        help="Thư mục output")
    parser.add_argument("--min_duration", type=float, default=2.0)
    parser.add_argument("--max_duration", type=float, default=12.0)
    parser.add_argument("--min_text_length", type=int, default=10,
                        help="Số ký tự text tối thiểu")
    parser.add_argument("--max_text_length", type=int, default=300)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Cài đặt: pip install datasets")
        sys.exit(1)

    out_dir = Path(args.output)
    wavs_dir = out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / "metadata_raw.csv"
    target_seconds = args.target_hours * 3600

    print(f"Target: {args.target_hours}h ({target_seconds:.0f}s)")
    print(f"Filter: {args.min_duration}-{args.max_duration}s audio, "
          f"{args.min_text_length}-{args.max_text_length} chars text")
    print(f"Output: {out_dir}")
    print()

    # Streaming load - không tải hết về máy
    print("Loading viVoice dataset (streaming)...")
    ds = load_dataset("capleaf/viVoice", split="train", streaming=True)

    total_seconds = 0.0
    n_kept = 0
    n_skip_duration = 0
    n_skip_text = 0

    pbar = tqdm(unit="samples")

    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)

        for sample in ds:
            pbar.update(1)

            if total_seconds >= target_seconds:
                break

            # viVoice schema: {audio: {array, sampling_rate}, text, channel, ...}
            try:
                audio = sample["audio"]
                wav = audio["array"]
                sr = audio["sampling_rate"]
                text = sample.get("text", "").strip()

                duration = len(wav) / sr

                if duration < args.min_duration or duration > args.max_duration:
                    n_skip_duration += 1
                    continue
                if len(text) < args.min_text_length or len(text) > args.max_text_length:
                    n_skip_text += 1
                    continue

                # Lưu audio
                fname = f"{n_kept:08d}.wav"
                fpath = wavs_dir / fname

                # Convert sang float32 nếu cần
                if wav.dtype != np.float32:
                    wav = wav.astype(np.float32)

                # viVoice thường ở 16kHz hoặc 22.05kHz, chưa resample về 24kHz ở đây
                # Sẽ resample ở bước 05_prepare_dataset.py
                sf.write(str(fpath), wav, sr)

                writer.writerow([fname, text, str(sr), f"{duration:.3f}"])
                n_kept += 1
                total_seconds += duration

                pbar.set_postfix({
                    "kept": n_kept,
                    "hours": f"{total_seconds/3600:.2f}",
                    "skip_dur": n_skip_duration,
                    "skip_txt": n_skip_text,
                })

            except Exception as e:
                continue

    pbar.close()

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Kept:               {n_kept} samples")
    print(f"  Total duration:     {total_seconds/3600:.2f} hours")
    print(f"  Skipped (duration): {n_skip_duration}")
    print(f"  Skipped (text):     {n_skip_text}")
    print(f"  Metadata:           {metadata_path}")
    print(f"  Audio dir:          {wavs_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
