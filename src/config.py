"""
Train config cho Chatterbox Vietnamese phoneme fine-tuning.
Tinh chỉnh sẵn cho RTX A4000 (16GB VRAM).
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    # ============================================================
    # MODEL
    # ============================================================
    is_turbo: bool = False  # Repo này CHỈ support bản Chatterbox gốc
    model_dir: str = "pretrained_models"

    # CHỈNH GIÁ TRỊ NÀY sau khi chạy scripts/03_build_tokenizer.py
    # Lệnh đó sẽ in ra số token chính xác.
    new_vocab_size: int = 120  # placeholder, sẽ ~80-150 tuỳ dataset

    # ============================================================
    # TOKENIZER (PHONEME)
    # ============================================================
    use_phoneme: bool = True
    tokenizer_path: str = "src/vi_phoneme_tokenizer.json"
    dialect: str = "s"  # 's'=Southern, 'n'=Northern, 'c'=Central
    tone_format: str = "pham"  # 'pham', 'chao', 'cao', 'super'

    # ============================================================
    # DATASET
    # ============================================================
    data_dir: str = "data"
    metadata_csv: str = "data/metadata.csv"
    wavs_dir: str = "data/wavs"

    # Format: 'ljspeech' (filename|raw|normalized) hoặc 'file_based' hoặc 'json'
    dataset_format: str = "ljspeech"

    sample_rate: int = 24000  # Chatterbox chuẩn
    max_audio_length: float = 12.0  # giây - giảm xuống 8 nếu OOM
    min_audio_length: float = 1.0

    # Pre-process audio offline (loudness normalize, VAD trim, resample)
    preprocess: bool = True
    preprocessed_dir: str = "data/preprocessed"

    # ============================================================
    # TRAINING — TUNED FOR A4000 (16GB)
    # ============================================================
    batch_size: int = 1
    grad_accum: int = 16  # effective batch = 16
    learning_rate: float = 5e-5  # giảm vì pretrained là English
    num_epochs: int = 10
    warmup_steps: int = 500

    save_steps: int = 1000
    save_total_limit: int = 5
    logging_steps: int = 50

    output_dir: str = "checkpoints/vi_phoneme"
    dataloader_num_workers: int = 4

    # VRAM optimization (BẮT BUỘC cho A4000)
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    # ============================================================
    # FREEZE STRATEGY
    # ============================================================
    freeze_voice_encoder: bool = True
    freeze_s3gen: bool = True
    train_t3: bool = True  # cái duy nhất train

    # ============================================================
    # LORA (optional, bật nếu vẫn OOM ở batch=1)
    # ============================================================
    use_lora: bool = False
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # ============================================================
    # EMBEDDING WARMUP
    # ============================================================
    # Khởi tạo embedding mới (phoneme) bằng trung bình embedding grapheme tương tự
    # Bắt buộc để training hội tụ nhanh trên compute hạn chế
    use_embedding_warmup: bool = True
    embedding_warmup_map: str = "src/phoneme_to_grapheme_init.json"

    # ============================================================
    # INFERENCE CALLBACK (sinh audio mẫu mỗi N steps để theo dõi)
    # ============================================================
    is_inference: bool = True
    inference_every_steps: int = 1000
    inference_text: str = "Xin chào, đây là mô hình tiếng Việt giọng Nam."
    inference_audio_prompt: str = "speaker_reference/reference.wav"


if __name__ == "__main__":
    # Print config khi chạy trực tiếp để verify
    cfg = TrainConfig()
    print("=" * 60)
    print("Current TrainConfig:")
    print("=" * 60)
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
