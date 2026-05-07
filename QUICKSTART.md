# Quickstart

Thứ tự chạy chính xác để fine-tune Chatterbox cho tiếng Việt phoneme.

## 0. Setup môi trường

```bash
# Tạo venv riêng
python -m venv venv
source venv/bin/activate  # hoặc venv\Scripts\activate trên Windows

# Install dependencies
pip install -r requirements.txt

# Clone Chatterbox gốc (cài như editable package)
git clone https://github.com/resemble-ai/chatterbox.git tools/chatterbox-orig
pip install -e tools/chatterbox-orig

# Clone vPhon
git clone https://github.com/kirbyj/vPhon.git tools/vPhon
```

## 1. Tải pretrained models

```bash
python setup.py
```

Output: `pretrained_models/` chứa 5 file (~1.5GB).

## 2. Test vPhon trước

```bash
python scripts/04_test_g2p.py
```

Verify output có dạng `s i n 1 | c a w 2` cho "xin chào". Nếu fail, kiểm tra `tools/vPhon` có tồn tại không.

## 3. Tải dataset

```bash
# Subset 100 giờ (vừa với A4000)
python scripts/01_download_dataset.py --target_hours 100 --output data/raw
```

Lưu ý: viVoice ~1TB total, streaming sẽ chỉ tải đủ 100h và dừng. Vẫn mất vài giờ tuỳ network.

## 4. Normalize text

```bash
python scripts/02_normalize_text.py \
    --input data/raw \
    --output data/normalized
```

## 5. Build phoneme tokenizer

```bash
python scripts/03_build_tokenizer.py \
    --metadata data/normalized/metadata_norm.csv \
    --output src/vi_phoneme_tokenizer.json \
    --dialect s --tone_format pham
```

**QUAN TRỌNG:** Output sẽ in ra `Vocab size: XXX`. Mở `src/config.py` và sửa:
```python
new_vocab_size: int = XXX  # giá trị từ output
```

## 6. Tạo default IPA→grapheme mapping cho warmup

```bash
python -c "from src.embedding_warmup import save_default_mapping; save_default_mapping('src/phoneme_to_grapheme_init.json')"
```

(Optional: mở file ra tinh chỉnh nếu muốn)

## 7. Resample + finalize dataset

```bash
python scripts/05_prepare_dataset.py \
    --input data/normalized \
    --input_wavs data/raw/wavs \
    --output data/ \
    --apply_loudness
```

Output: `data/wavs/*.wav` (24kHz) + `data/metadata.csv`.

## 8. Pre-compute speech tokens

```bash
python scripts/06_precompute_speech_tokens.py
```

Mỗi `xxx.wav` sẽ có thêm `xxx.speech_tokens.pt` + `xxx.cond.pt` cạnh nó.

## 9. Setup speaker reference

Đặt 1 file `.wav` (3-10 giây, giọng Nam, audio sạch) vào:
```
speaker_reference/reference.wav
```

## 10. Train

```bash
# Bắt đầu training
python train.py 2>&1 | tee train.log
```

Trong terminal khác để monitor:
```bash
# VRAM
watch -n 5 nvidia-smi

# Loss curve
tensorboard --logdir checkpoints/vi_phoneme/runs
```

**Expected loss progression (A4000, 100h dataset):**
| Step  | Loss    | Note                           |
|-------|---------|--------------------------------|
| 0     | ~7-9    | Embedding warmup giúp khởi điểm thấp |
| 1k    | ~5-6    | Bắt đầu học pattern            |
| 5k    | ~3.5-4  | Audio bắt đầu nhận diện được   |
| 10k   | ~2.5-3  | Khá tốt rồi                    |
| 20k+  | ~2-2.5  | Diminishing returns             |

Một epoch trên 100h ≈ 10000-15000 steps tuỳ batch effective size.

## 11. Inference

```bash
# Sửa TEXT_TO_SAY trong inference.py
python inference.py
```

Output: `output.wav`.

## Troubleshooting

### OOM trên A4000
1. Giảm `max_audio_length` trong config xuống 8s
2. Bật `use_lora=True`
3. Tăng `grad_accum` lên 32

### Loss không giảm
1. Verify embedding warmup chạy thực sự (đọc log)
2. Verify tokenizer encode đúng
3. Check learning rate (5e-5 cho full FT, 1e-4 cho LoRA)

### Audio output có nhiễu / không phải giọng Nam
1. Train chưa đủ - cần 5k+ steps
2. Reference audio không phải giọng Nam — đổi
3. Dataset chứa giọng Bắc lẫn vào — filter strict hơn

### vPhon ra kết quả lạ
- vPhon expect input lowercase, không có dấu câu
- Một số từ Hán-Việt cổ vPhon không cover
- Nếu nhiều từ fail, check pre-normalize có đủ aggressive
