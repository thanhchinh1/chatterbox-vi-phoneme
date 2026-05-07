# Caveats & Known Issues

Skeleton này là khung sườn. Có một số điểm cần bạn tự fill hoặc adapt theo phiên bản Chatterbox cụ thể.

## 🔴 Cần adapt theo API Chatterbox

### 1. T3 forward signature

File: `src/model.py` → `T3TrainerWrapper.forward()`

T3 từ `chatterbox-tts==0.1.2` có thể có signature khác với template. Cần inspect:
```python
from chatterbox.models.t3.t3 import T3
import inspect
print(inspect.signature(T3.forward))
```

Sửa wrapper cho khớp.

### 2. Text embedding key name

File: `src/model.py` → `resize_t3_text_embedding()`

Tên layer text embedding trong T3 có thể là một trong:
- `text_emb.weight`
- `text_token_emb.weight`  
- `tfmr.wte.weight` (cho Turbo)
- Hoặc khác

Inspect bằng:
```python
sd = tts_engine.t3.state_dict()
for k, v in sd.items():
    if "emb" in k.lower():
        print(k, v.shape)
```

Update list `text_emb_keys` cho khớp.

### 3. S3Gen tokenizer API

File: `scripts/06_precompute_speech_tokens.py` + `src/dataset.py`

API gọi để encode wav → speech tokens:
- Có thể là `engine.s3gen.tokenize(wav)`
- Hoặc `engine.s3gen.tokenizer(wav)`
- Hoặc `engine.s3gen.encode(wav)`

Test với:
```python
engine = ChatterboxTTS.from_local("pretrained_models", device="cuda")
print(dir(engine.s3gen))
```

### 4. Inference API

File: `inference.py` → `engine.generate()`

Hàm `generate()` có thể nhận `text=str` chứ không phải `text_token_ids=list`. Nếu vậy:
- Option A: Override để nhận token IDs trực tiếp
- Option B: Subclass tokenizer của Chatterbox để bypass tokenization mà nó tự làm

## 🟡 Logic chưa hoàn thiện

### 1. `normalize_numbers` đơn giản

File: `src/vi_text_processor.py`

Chỉ xử lý số nguyên cơ bản. Chưa xử lý:
- Số thập phân: "3.14" → "ba phẩy mười bốn"
- Tiền tệ: "1.000.000đ" → "một triệu đồng"
- Ngày tháng: "15/03/2024" → "ngày mười lăm tháng ba năm hai nghìn không trăm hai mươi tư"
- Số điện thoại
- Phần trăm: "50%" → "năm mươi phần trăm"
- Đơn vị: "5kg" → "năm ki lô gam"

Cần extend bằng regex + rules.

### 2. Abbreviation list ngắn

File: `src/vi_text_processor.py` → `_ABBREV`

Mới có ~20 viết tắt. Bạn có thể thêm:
- Tỉnh thành: HN, HCM, ĐN, HP, CT, ...
- Tổ chức: ĐH, CĐ, THPT, THCS, BGD, ...
- Common: ĐT (điện thoại), STK (số tài khoản), ...

### 3. IPA → grapheme map cho embedding warmup

File: `src/embedding_warmup.py` → `DEFAULT_IPA_TO_GRAPHEME`

Mapping mặc định khá thô. Nên tinh chỉnh theo:
- Tham khảo: https://en.wikipedia.org/wiki/Vietnamese_phonology
- Sau khi build vocab, xem token nào không có mapping → thêm vào

### 4. Speech token computation

File: `src/dataset.py`

Speech tokens cần được compute đúng theo S3Gen của Chatterbox. Code hiện tại là template — verify lại với `chatterbox-tts==0.1.2` thực tế.

## 🟢 Thông tin khác

### Storage requirement

- Pretrained models: ~1.5GB
- viVoice 100h: ~15-25GB tuỳ format
- Pre-computed speech tokens: ~5GB cho 100h
- Checkpoint per save: ~2GB (nếu save full T3 mỗi 1000 steps × 5 limit = ~10GB)
- **Tổng: ~50GB free disk space**

### Time budget

Trên A4000 với 100h dataset:
- Bước 1-7 (data prep): 4-8 giờ (chủ yếu là download)
- Bước 8 (precompute): 1-2 giờ
- Bước 10 (training, đến hội tụ): 4-7 ngày liên tục

Nếu dùng LoRA: bước 10 nhanh hơn ~2x.

### Watermark

Mọi audio output từ Chatterbox đều có Perth watermark (imperceptible neural watermark) của Resemble AI. Không thể disable.

### License của output model

Bạn fine-tune từ Chatterbox (MIT) trên dataset viVoice (CC-BY-NC-SA-4.0). Output model:
- KHÔNG được dùng cho mục đích thương mại (do viVoice license)
- Phải attribution viVoice + Chatterbox
- Phải share-alike

Nếu muốn commercial: phải dùng dataset khác (proprietary, hoặc public domain, hoặc license cho phép commercial).

## 📋 Checklist trước khi training

- [ ] `python setup.py` chạy thành công
- [ ] `tools/vPhon/` tồn tại + `python scripts/04_test_g2p.py` ra kết quả đúng
- [ ] Dataset đã download + normalize + resample về 24kHz
- [ ] Tokenizer đã build, vocab_size đã update vào `src/config.py`
- [ ] Speech tokens đã pre-compute (kiểm tra `data/wavs/*.speech_tokens.pt` có tồn tại)
- [ ] `speaker_reference/reference.wav` đã đặt đúng (giọng Nam, sạch, 3-10s)
- [ ] `nvidia-smi` thấy A4000 free RAM, không có process khác chiếm dụng
- [ ] T3 forward signature đã verify (xem section "Cần adapt" #1)
- [ ] Text embedding key đã verify (xem section "Cần adapt" #2)
