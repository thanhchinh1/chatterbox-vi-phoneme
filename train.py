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
import copy
import argparse
import random
import torch
from transformers import Trainer, TrainingArguments
from safetensors.torch import save_file
from torch.utils.data import Subset

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


def build_train_eval_split(dataset, eval_size: int = 500, seed: int = 42):
    if len(dataset) <= 1:
        return dataset, None

    eval_size = max(1, min(eval_size, len(dataset) // 10))
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    eval_indices = indices[:eval_size]
    train_indices = indices[eval_size:]
    return Subset(dataset, train_indices), Subset(dataset, eval_indices)


def find_latest_checkpoint(output_dir: str):
    """Tìm checkpoint mới nhất trong output_dir. Return None nếu không có."""
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    
    if not checkpoints:
        return None
    
    # Sort by checkpoint number (checkpoint-1000, checkpoint-2000, etc.)
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    latest = checkpoints[-1]
    latest_path = os.path.join(output_dir, latest)
    logger.info(f"Found latest checkpoint: {latest_path}")
    return latest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run short smoke training")
    parser.add_argument("--smoke_steps", type=int, default=100, help="Optimizer steps for smoke mode")
    parser.add_argument("--smoke_samples", type=int, default=50, help="Number of samples for smoke subset")
    parser.add_argument("--smoke_output_dir", type=str, default=None, help="Override output_dir for smoke mode")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.smoke:
        cfg.output_dir = args.smoke_output_dir or os.path.join(cfg.output_dir, "smoke")

    logger.info("=" * 70)
    logger.info("CHATTERBOX VIETNAMESE PHONEME FINE-TUNING")
    logger.info("=" * 70)
    logger.info(f"Dialect: {cfg.dialect} | Tone format: {cfg.tone_format}")
    logger.info(f"New vocab size: {cfg.new_vocab_size}")
    logger.info(f"LoRA: {cfg.use_lora} | bf16: {cfg.bf16} | grad_ckpt: {cfg.gradient_checkpointing}")
    if args.smoke:
        logger.info(f"SMOKE MODE enabled: steps={args.smoke_steps}, samples={args.smoke_samples}")

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
    import src.patch_perth  # noqa: F401
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.models.t3.t3 import T3

    tts_engine = ChatterboxTTS.from_local(cfg.model_dir, device="cpu")

    # Lưu state_dict + tokenizer cũ để dùng warmup
    pretrained_t3_sd = tts_engine.t3.state_dict()
    original_t3_hp = tts_engine.t3.hp
    old_tokenizer = tts_engine.tokenizer  # grapheme tokenizer

    # 2. CREATE NEW T3 VỚI VOCAB MỚI
    # 3. LOAD WEIGHTS + RESIZE EMBEDDING + WARMUP
    logger.info("Loading pretrained weights with embedding resize + warmup...")
    new_phoneme_tokenizer = PhonemeTokenizer.load(cfg.tokenizer_path)
    actual_vocab_size = len(new_phoneme_tokenizer.vocab)
    if actual_vocab_size != cfg.new_vocab_size:
        logger.warning(
            f"Config new_vocab_size={cfg.new_vocab_size} != tokenizer vocab={actual_vocab_size}; "
            f"using tokenizer vocab size."
        )

    logger.info(f"Creating new T3 with vocab_size={actual_vocab_size}")
    new_hp = copy.deepcopy(original_t3_hp)
    new_hp.text_tokens_dict_size = actual_vocab_size
    # Must match new tokenizer special token ids.
    new_hp.start_text_token = new_phoneme_tokenizer.bos_id
    new_hp.stop_text_token = new_phoneme_tokenizer.eos_id
    if hasattr(new_hp, "use_cache"):
        new_hp.use_cache = False
    new_t3 = T3(hp=new_hp)

    # Load mapping IPA → grapheme nếu có
    ipa_to_grapheme = None
    if cfg.use_embedding_warmup and os.path.exists(cfg.embedding_warmup_map):
        from src.embedding_warmup import load_mapping
        ipa_to_grapheme = load_mapping(cfg.embedding_warmup_map)
        logger.info(f"Loaded warmup mapping: {len(ipa_to_grapheme)} entries")

    new_t3 = resize_t3_text_embedding(
        new_t3=new_t3,
        old_state_dict=pretrained_t3_sd,
        new_vocab_size=actual_vocab_size,
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
    full_ds = ChatterboxViDataset(cfg, tts_engine=None)  # speech tokens precomputed
    train_ds, eval_ds = build_train_eval_split(full_ds, eval_size=500 if not args.smoke else 20)
    if args.smoke:
        n_subset = min(args.smoke_samples, len(train_ds))
        train_ds = Subset(train_ds, list(range(n_subset)))
        if eval_ds is not None:
            eval_subset = min(10, len(eval_ds))
            eval_ds = Subset(eval_ds, list(range(eval_subset)))
        logger.info(f"Using smoke subset: {n_subset} train samples")

    if eval_ds is not None:
        logger.info(f"Eval split size: {len(eval_ds)} samples")

    # 7. WRAPPER + TRAINER
    model_wrapper = T3TrainerWrapper(tts_engine.t3)

    # 7.5 TRY TO RESUME FROM LATEST CHECKPOINT
    latest_checkpoint = find_latest_checkpoint(cfg.output_dir)
    resume_from_checkpoint = None
    if latest_checkpoint and not args.smoke:
        logger.info(f"Attempting to resume from: {latest_checkpoint}")
        try:
            # Load model weights từ checkpoint
            checkpoint_model_path = os.path.join(latest_checkpoint, "pytorch_model.bin")
            if os.path.exists(checkpoint_model_path):
                logger.info(f"Loading model weights from {checkpoint_model_path}...")
                checkpoint_state = torch.load(checkpoint_model_path, map_location=device, weights_only=False)
                # Bỏ wrapper prefix nếu có
                if all(k.startswith("model.") for k in checkpoint_state.keys()):
                    checkpoint_state = {k[6:]: v for k, v in checkpoint_state.items()}
                model_wrapper.model.load_state_dict(checkpoint_state, strict=False)
                logger.info("✓ Loaded model weights from checkpoint")
                resume_from_checkpoint = latest_checkpoint
            else:
                logger.warning(f"No pytorch_model.bin found in {latest_checkpoint}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting from scratch instead")
    elif args.smoke:
        logger.info("Smoke mode: skipping checkpoint resume")

    # Callbacks
    callbacks = []
    if cfg.is_inference:
        callbacks.append(InferenceCallback(cfg, tts_engine_ref=tts_engine))

    # Training args
    train_batch_size = cfg.batch_size
    train_grad_accum = cfg.grad_accum
    train_num_epochs = cfg.num_epochs
    train_save_steps = cfg.save_steps
    train_logging_steps = cfg.logging_steps
    max_steps = -1
    if args.smoke:
        # Keep smoke fast and deterministic for API/training sanity checks.
        train_batch_size = 1
        train_grad_accum = 1
        train_num_epochs = 1
        train_save_steps = max(1, args.smoke_steps // 2)
        train_logging_steps = max(1, min(10, args.smoke_steps // 5))
        max_steps = args.smoke_steps

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=train_grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=train_num_epochs,
        max_steps=max_steps,
        warmup_steps=cfg.warmup_steps,
        save_strategy="steps",
        save_steps=train_save_steps,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=train_logging_steps,
        remove_unused_columns=False,
        dataloader_num_workers=(0 if args.smoke else cfg.dataloader_num_workers),
        report_to=["tensorboard"],
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_persistent_workers=(False if args.smoke else True),
        dataloader_pin_memory=True,
        evaluation_strategy=("steps" if eval_ds is not None else "no"),
        eval_steps=(train_save_steps if eval_ds is not None else None),
        load_best_model_at_end=False,  # Rely on save_total_limit=1 instead
    )

    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator_phoneme,
        callbacks=callbacks,
    )

    # 8. TRAIN
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if eval_ds is not None:
        metrics = trainer.evaluate()
        logger.info(f"Final eval metrics: {metrics}")

    if args.smoke:
        losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
        if losses:
            logger.info(f"Smoke loss trend: first={losses[0]:.4f}, last={losses[-1]:.4f}")
        ckpts = [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint-")]
        logger.info(f"Smoke checkpoints found: {sorted(ckpts)}")

    if args.smoke:
        logger.info("Smoke run complete. Skipping final full-model export to save disk.")
        return

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
