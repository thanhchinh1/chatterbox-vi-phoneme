# Chatterbox-VI-Phoneme

Fine-tune Chatterbox TTS cho tiếng Việt (giọng Nam Sài Gòn) sử dụng phoneme-based tokenization với vPhon.

## Tổng quan

- **Base model:** Chatterbox TTS gốc (Resemble AI, 0.5B Llama backbone, grapheme tokenizer)
- **Target:** Tiếng Việt giọng Nam, biểu diễn phoneme bằng IPA + Pham tone (1-6)
- **G2P:** vPhon (kirbyj/vPhon)
- **Dataset:** viVoice (capleaf/viVoice) — subset 100-150h chất lượng cao
- **Compute target:** RTX A4000 (16GB VRAM), bf16, gradient checkpointing, batch 1 + grad_accum 16

## Pipeline

```
Vietnamese text
  ↓ normalize (số→chữ, viết tắt, lowercase)
  ↓ vPhon (dialect=s, pham tones, no superscript)
Phoneme sequence ("s i n¹ | c a w²")
  ↓ Phoneme tokenizer (~80-120 tokens)
T3 (fine-tuned)
  ↓ Speech tokens
S3Gen (frozen)
  ↓ Mel + waveform
Audio output
```

## Cấu trúc repo

```
chatterbox-vi-phoneme/
├── README.md                    # File này
├── requirements.txt             # Dependencies
├── setup.py                     # Tải pretrained Chatterbox
├── train.py                     # Script train chính
├── inference.py                 # Script inference
│
├── src/
│   ├── config.py                # TrainConfig
│   ├── dataset.py               # ChatterboxDataset + collator
│   ├── model.py                 # T3 wrapper + resize embedding
│   ├── vi_text_processor.py     # Pipeline: normalize → vPhon → tokens
│   ├── phoneme_tokenizer.py     # Build vocab + tokenizer
│   ├── embedding_warmup.py      # Khởi tạo embedding mới từ grapheme tương tự
│   └── utils.py                 # Logger, VAD trim, helpers
│
├── tools/
│   └── vPhon/                   # Clone từ kirbyj/vPhon
│
├── scripts/
│   ├── 01_download_dataset.py   # Tải + filter viVoice
│   ├── 02_normalize_text.py     # Normalize toàn bộ transcript
│   ├── 03_build_tokenizer.py    # Build phoneme vocab từ dataset
│   ├── 04_test_g2p.py           # Sanity check vPhon output
│   └── 05_prepare_dataset.py    # Convert sang LJSpeech format
│
├── data/                        # Dataset (gitignored)
│   ├── metadata.csv
│   └── wavs/
│
├── pretrained_models/           # (gitignored, tải bằng setup.py)
├── checkpoints/                 # (gitignored, output training)
└── speaker_reference/           # File .wav giọng tham chiếu
    └── reference.wav
```

## Bắt đầu

### Bước 1: Setup môi trường

```bash
# Clone repo này + Chatterbox gốc + vPhon
git clone <this-repo> chatterbox-vi-phoneme
cd chatterbox-vi-phoneme

# Cài Chatterbox gốc (hoặc copy vào src/chatterbox_/ như chatterbox-vn)
git clone https://github.com/resemble-ai/chatterbox.git tools/chatterbox-orig
pip install -e tools/chatterbox-orig

# Clone vPhon
git clone https://github.com/kirbyj/vPhon.git tools/vPhon

# Cài dependencies còn lại
pip install -r requirements.txt
```

### Bước 2: Tải pretrained model

```bash
python setup.py
```

Sẽ tải các file sau về `pretrained_models/`:
- `ve.safetensors` — Voice Encoder
- `t3_cfg.safetensors` — T3 (sẽ fine-tune)
- `s3gen.safetensors` — S3Gen decoder
- `conds.pt` — Default conditioning
- `tokenizer.json` — Grapheme tokenizer gốc (sẽ thay)

### Bước 3: Chuẩn bị dataset

```bash
# Tải viVoice (streaming, filter on-the-fly)
python scripts/01_download_dataset.py --target_hours 100 --output data/raw

# Normalize text
python scripts/02_normalize_text.py --input data/raw --output data/normalized

# Convert sang LJSpeech format
python scripts/05_prepare_dataset.py --input data/normalized --output data/
```

Output: `data/metadata.csv` + `data/wavs/*.wav`

### Bước 4: Test G2P + build tokenizer

```bash
# Verify vPhon hoạt động đúng
python scripts/04_test_g2p.py

# Build phoneme vocab từ toàn bộ dataset
python scripts/03_build_tokenizer.py --metadata data/metadata.csv --output src/vi_phoneme_tokenizer.json
```

Sau bước này, bạn cần update `new_vocab_size` trong `src/config.py` cho khớp size tokenizer mới.

### Bước 5: Train

```bash
# Đặt 1 file .wav giọng tham chiếu vào speaker_reference/reference.wav
# (3-10 giây, giọng Nam, audio sạch)

# Chạy training
python train.py
```

Monitor:
```bash
tensorboard --logdir checkpoints/runs
nvidia-smi -l 5
```

### Bước 6: Inference

```bash
# Sửa TEXT_TO_SAY trong inference.py
python inference.py
```

Output: `output.wav`

## Lưu ý quan trọng

1. **VRAM A4000 (16GB) sát giới hạn.** Nếu OOM, giảm `max_audio_length` xuống 8s và `batch_size=1` hoàn toàn.

2. **Embedding warmup là bắt buộc** với compute hạn chế. Đừng skip bước này nếu không muốn train cả tháng.

3. **Subset dataset.** Đừng cố train full 1000h viVoice trên A4000 — không kham nổi. 100-150h là vừa.

4. **License viVoice là CC-BY-NC-SA-4.0** — chỉ dùng cho nghiên cứu, không thương mại.

5. **Chatterbox có Perth watermark** — mọi audio output đều có watermark imperceptible. Đây là yêu cầu của Resemble AI.

## Tham khảo

- Chatterbox: https://github.com/resemble-ai/chatterbox
- vPhon: https://github.com/kirbyj/vPhon
- viVoice: https://huggingface.co/datasets/capleaf/viVoice

- Fine-tune toolkit gốc: https://github.com/gokhaneraslan/chatterbox-finetuning
