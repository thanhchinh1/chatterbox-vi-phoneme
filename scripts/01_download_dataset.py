import argparse
import os
import csv
import sys
import io
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
        from datasets import load_dataset, Audio
    except ImportError:
        print("Cài đặt: pip install datasets")
        sys.exit(1)

    out_dir = Path(args.output)
    wavs_dir = out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / "metadata_raw.csv"
    target_seconds = args.target_hours * 3600

    print(f"--- Cấu hình tải ---")
    print(f"Target: {args.target_hours}h ({target_seconds:.0f}s)")
    print(f"Filter: {args.min_duration}-{args.max_duration}s, {args.min_text_length}-{args.max_text_length} chars")
    print(f"Output: {out_dir}")
    print(f"--------------------\n")

    # Streaming load - không tải hết về máy
    print("Loading viVoice dataset (streaming)...")
    ds = load_dataset("capleaf/viVoice", split="train", streaming=True)
    
    # Quan trọng: Tắt tính năng tự động decode của datasets để tránh lỗi torchcodec/ffmpeg
    ds = ds.cast_column("audio", Audio(decode=False))

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
                print("\nĐã đạt đủ số giờ mục tiêu!")
                break

            try:
                # Đọc audio thủ công từ bytes
                audio_data = sample["audio"]
                audio_bytes = audio_data["bytes"]
                
                # Dùng soundfile đọc trực tiếp từ bộ nhớ (BytesIO)
                with io.BytesIO(audio_bytes) as b:
                    wav, sr = sf.read(b)
                
                text = sample.get("text", "").strip()
                duration = len(wav) / sr

                # Filter duration
                if duration < args.min_duration or duration > args.max_duration:
                    n_skip_duration += 1
                    continue
                
                # Filter text
                if len(text) < args.min_text_length or len(text) > args.max_text_length:
                    n_skip_text += 1
                    continue

                # Lưu audio ra disk
                fname = f"{n_kept:08d}.wav"
                fpath = wavs_dir / fname

                # Đảm bảo định dạng float32
                if wav.dtype != np.float32:
                    wav = wav.astype(np.float32)

                # Ghi file wav
                sf.write(str(fpath), wav, sr)

                # Ghi log vào metadata
                writer.writerow([fname, text, str(sr), f"{duration:.3f}"])
                
                n_kept += 1
                total_seconds += duration

                # Cập nhật thông tin trên thanh tiến trình
                pbar.set_postfix({
                    "kept": n_kept,
                    "hrs": f"{total_seconds/3600:.2f}",
                    "skip_d": n_skip_duration,
                    "skip_t": n_skip_text,
                })

            except Exception as e:
                # Bỏ qua các mẫu bị lỗi format
                continue

    pbar.close()

    print(f"\n{'='*60}")
    print(f"Hoàn tất!")
    print(f"  Số mẫu đã lưu:     {n_kept}")
    print(f"  Tổng thời lượng:   {total_seconds/3600:.2f} giờ")
    print(f"  Bỏ qua (độ dài):   {n_skip_duration}")
    print(f"  Bỏ qua (văn bản):  {n_skip_text}")
    print(f"  File metadata:     {metadata_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()