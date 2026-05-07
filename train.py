"""
Main training script cho Chatterbox Vietnamese phoneme fine-tuning.

Usage:
    python train.py

Trước khi chạy, đảm bảo:
  1. python setup.py                              # tải pretrained
  2. git clone https://github.com/kirbyj/vPhon.git tools/vPhon
  3. python scripts/01_download_dataset.py        # tải viVoice
  4. python scripts/05_prepare_dataset.py         # convert sang LJSpeech
  5. python scripts/03_build_tokenizer.py         # build phoneme vocab
  6. Update src/config.py:new_vocab_size cho khớp tokenizer
"""
import os
import sys
import torch
from transformers import Trainer, TrainingArguments
from safetensors.torch import save_file

from src.config import TrainConfig
from src.dataset import ChatterboxViDataset, data_collator_phoneme
from src.model import (
    resize_t3_text_embedding,
    T3TrainerWrapper,
    freeze_non_t3_components,
    maybe_apply_lora,
)
from src.phoneme_tokenizer import PhonemeTokenizer
from src.utils import setup_logger, check_pretrained_models, count_parameters
from src.inference_callback import InferenceCallback

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = setup_logger("Train")


def main():
    cfg = TrainConfig()

    logger.info("=" * 70)
    logger.info("CHATTERBOX VIETNAMESE PHONEME FINE-TUNING")
    logger.info("=" * 70)
    logger.info(f"Dialect: {cfg.dialect} | Tone format: {cfg.tone_format}")
    logger.info(f"New vocab size: {cfg.new_vocab_size}")
    logger.info(f"LoRA: {cfg.use_lora} | bf16: {cfg.bf16} | grad_ckpt: {cfg.gradient_checkpointing}")

    # 0. CHECK FILES
    if not check_pretrained_models(cfg.model_dir):
        sys.exit(1)
    if not os.path.exists(cfg.tokenizer_path):
        logger.error(f"Tokenizer not found: {cfg.tokenizer_path}")
        logger.error("Run scripts/03_build_tokenizer.py first")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1. LOAD ORIGINAL CHATTERBOX (CPU để tiết kiệm VRAM)
    logger.info("Loading original Chatterbox...")
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.models.t3.t3 import T3

    tts_engine = ChatterboxTTS.from_local(cfg.model_dir, device="cpu")

    # Lưu state_dict + tokenizer cũ để dùng warmup
    pretrained_t3_sd = tts_engine.t3.state_dict()
    original_t3_hp = tts_engine.t3.hp
    old_tokenizer = tts_engine.tokenizer  # grapheme tokenizer

    # 2. CREATE NEW T3 VỚI VOCAB MỚI
    logger.info(f"Creating new T3 with vocab_size={cfg.new_vocab_size}")
    new_hp = original_t3_hp
    new_hp.text_tokens_dict_size = cfg.new_vocab_size
    if hasattr(new_hp, "use_cache"):
        new_hp.use_cache = False
    new_t3 = T3(hp=new_hp)

    # 3. LOAD WEIGHTS + RESIZE EMBEDDING + WARMUP
    logger.info("Loading pretrained weights with embedding resize + warmup...")
    new_phoneme_tokenizer = PhonemeTokenizer.load(cfg.tokenizer_path)

    # Load mapping IPA → grapheme nếu có
    ipa_to_grapheme = None
    if cfg.use_embedding_warmup and os.path.exists(cfg.embedding_warmup_map):
        from src.embedding_warmup import load_mapping
        ipa_to_grapheme = load_mapping(cfg.embedding_warmup_map)
        logger.info(f"Loaded warmup mapping: {len(ipa_to_grapheme)} entries")

    new_t3 = resize_t3_text_embedding(
        new_t3=new_t3,
        old_state_dict=pretrained_t3_sd,
        new_vocab_size=cfg.new_vocab_size,
        use_warmup=cfg.use_embedding_warmup,
        new_vocab=new_phoneme_tokenizer.vocab,
        old_tokenizer=old_tokenizer,
        ipa_to_grapheme=ipa_to_grapheme,
    )

    # Cleanup
    del pretrained_t3_sd

    # 4. INJECT NEW T3 vào engine + FREEZE
    tts_engine.t3 = new_t3
    freeze_non_t3_components(tts_engine)

    # 5. APPLY LORA (optional)
    if cfg.use_lora:
        tts_engine.t3 = maybe_apply_lora(tts_engine.t3, cfg)

    # Move T3 to device (VE và S3Gen vẫn ở CPU vì frozen)
    tts_engine.t3.to(device)

    # Print param count
    counts = count_parameters(tts_engine.t3)
    logger.info(f"T3 params: {counts['total']/1e6:.1f}M total, "
                f"{counts['trainable']/1e6:.1f}M trainable "
                f"({counts['trainable_pct']:.2f}%)")

    # 6. DATASET
    logger.info("Loading dataset...")
    train_ds = ChatterboxViDataset(cfg, tts_engine=None)  # speech tokens precomputed

    # 7. WRAPPER + TRAINER
    model_wrapper = T3TrainerWrapper(tts_engine.t3)

    # Callbacks
    callbacks = []
    if cfg.is_inference:
        callbacks.append(InferenceCallback(cfg, tts_engine_ref=tts_engine))

    # Training args
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        warmup_steps=cfg.warmup_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        remove_unused_columns=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to=["tensorboard"],
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator_phoneme,
        callbacks=callbacks,
    )

    # 8. TRAIN
    logger.info("Starting training...")
    trainer.train()

    # 9. SAVE FINAL MODEL
    logger.info("Training complete. Saving final model...")
    os.makedirs(cfg.output_dir, exist_ok=True)
    final_path = os.path.join(cfg.output_dir, "t3_vi_phoneme_final.safetensors")

    # Nếu dùng LoRA, save adapter; nếu không, save full state_dict
    if cfg.use_lora:
        tts_engine.t3.save_pretrained(os.path.join(cfg.output_dir, "lora_adapter"))
        logger.info(f"LoRA adapter saved to: {cfg.output_dir}/lora_adapter")
    else:
        save_file(tts_engine.t3.state_dict(), final_path)
        logger.info(f"Full T3 saved to: {final_path}")


if __name__ == "__main__":
    main()
