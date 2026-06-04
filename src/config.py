"""Train config cho Chatterbox Vietnamese phoneme fine-tuning.
Tinh chỉnh sẵn cho RTX A4000 (16GB VRAM).
"""
import warnings
from dataclasses import dataclass, field
from typing import Optional

_PLACEHOLDER_VOCAB_SIZE = 58


@dataclass
class TrainConfig:
    # ============================================================
    # MODEL
    # ============================================================
    is_turbo: bool = False
    model_dir: str = "pretrained_models"

    new_vocab_size: int = _PLACEHOLDER_VOCAB_SIZE

    # ============================================================
    # TOKENIZER (PHONEME)
    # ============================================================
    use_phoneme: bool = True
    tokenizer_path: str = "src/vi_phoneme_tokenizer.json"
    dialect: str = "s"
    tone_format: str = "letter"

    # ============================================================
    # DATASET
    # ============================================================
    data_dir: str = "data"
    metadata_csv: str = "data/metadata.csv"
    wavs_dir: str = "data/wavs"
    dataset_format: str = "ljspeech"

    sample_rate: int = 24000
    max_audio_length: float = 12.0
    min_audio_length: float = 1.0

    preprocess: bool = True
    preprocessed_dir: str = "data/preprocessed"

    # ============================================================
    # TRAINING — TUNED FOR A4000 (16GB)
    # ============================================================
    batch_size: int = 1
    grad_accum: int = 16
    learning_rate: float = 1e-5
    num_epochs: int = 10
    warmup_steps: int = 500

    save_steps: int = 1000
    save_total_limit: int = 3
    logging_steps: int = 50

    output_dir: str = "checkpoints/vi_phoneme"
    dataloader_num_workers: int = 4

    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    # ============================================================
    # FREEZE STRATEGY
    # ============================================================
    freeze_voice_encoder: bool = True
    freeze_s3gen: bool = True
    train_t3: bool = True

    # ============================================================
    # LORA (optional)
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
    use_embedding_warmup: bool = True
    embedding_warmup_map: str = "src/phoneme_to_grapheme_init.json"

    # ============================================================
    # INFERENCE CALLBACK
    # ============================================================
    is_inference: bool = True
    inference_every_steps: int = 1000
    inference_text: str = "Xin chào, đây là mô hình tiếng Việt giọng Nam."
    inference_audio_prompt: str = "speaker_reference/reference.wav"

    # ============================================================
    # ENTROPY EARLY-STOP GUARD
    # ============================================================
    enable_entropy_guard: bool = True
    entropy_check_every_steps: int = 1000
    entropy_guard_samples: int = 32
    entropy_guard_batch_size: int = 4
    entropy_stop_threshold: float = 0.05
    top1_stop_threshold: float = 0.95
    entropy_guard_min_steps: int = 1000

    def __post_init__(self):
        if self.new_vocab_size == _PLACEHOLDER_VOCAB_SIZE:
            warnings.warn(
                f"[TrainConfig] new_vocab_size={_PLACEHOLDER_VOCAB_SIZE} is a placeholder. "
                "Run scripts/03_build_tokenizer.py then update this config "
                "to match the real vocab size (~80-150 tokens).",
                RuntimeWarning,
            )


if __name__ == "__main__":
    cfg = TrainConfig()
    print("=" * 60)
    print("Current TrainConfig:")
    print("=" * 60)
    for k, v in cfg.__dict__.items():
        print(f" {k}: {v}")
