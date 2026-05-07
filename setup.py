"""
Tải pretrained Chatterbox model từ HuggingFace.
Chỉ tải bản gốc (không Turbo) vì repo này chuyên cho phoneme-based fine-tuning.
"""
import os
import sys
import requests
from tqdm import tqdm

DEST_DIR = "pretrained_models"

CHATTERBOX_FILES = {
    "ve.safetensors": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/ve.safetensors?download=true",
    "t3_cfg.safetensors": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/t3_cfg.safetensors?download=true",
    "s3gen.safetensors": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/s3gen.safetensors?download=true",
    "conds.pt": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/conds.pt?download=true",
    "tokenizer.json": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/grapheme_mtl_merged_expanded_v1.json?download=true",
}


def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"[SKIP] Already exists: {dest_path}")
        return

    print(f"[DOWNLOAD] {os.path.basename(dest_path)}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 8

        with open(dest_path, "wb") as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        print(f"[DONE] {dest_path}\n")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] {url}: {e}")
        sys.exit(1)


def main():
    print("=" * 60)
    print("Chatterbox Pretrained Model Setup (vi-phoneme variant)")
    print("=" * 60)

    os.makedirs(DEST_DIR, exist_ok=True)

    for filename, url in CHATTERBOX_FILES.items():
        dest = os.path.join(DEST_DIR, filename)
        download_file(url, dest)

    print("\n" + "=" * 60)
    print("INSTALLATION COMPLETE")
    print(f"Models saved to: {DEST_DIR}/")
    print("\nNext steps:")
    print("  1. git clone https://github.com/kirbyj/vPhon.git tools/vPhon")
    print("  2. python scripts/01_download_dataset.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
