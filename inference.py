"""
Inference script: text tiếng Việt → audio dùng model đã fine-tune.

Pipeline:
  text → vi_punc_norm → expand_abbrev → normalize_numbers → vi_g2p → tokenize → T3 → S3Gen → wav
"""
import os
import re
import torch
import numpy as np
import soundfile as sf
import random
from safetensors.torch import load_file

from src.utils import setup_logger, trim_silence_with_vad
from src.config import TrainConfig
from src.phoneme_tokenizer import PhonemeTokenizer
from src.vi_text_processor import vi_text_to_phonemes

logger = setup_logger("Inference")
cfg = TrainConfig()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_DIR = cfg.model_dir
OUTPUT_DIR = cfg.output_dir


def find_latest_checkpoint_weights():
    """
    Tìm checkpoint mới nhất hoặc final weights.
    Priority: checkpoint-XXXX/pytorch_model.bin > t3_vi_phoneme_final.safetensors
    """
    if os.path.exists(OUTPUT_DIR):
        # Tìm tất cả checkpoint
        checkpoints = [
            d for d in os.listdir(OUTPUT_DIR)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(OUTPUT_DIR, d))
        ]
        
        if checkpoints:
            # Sort by checkpoint number
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            latest_ckpt = checkpoints[-1]
            ckpt_path = os.path.join(OUTPUT_DIR, latest_ckpt)
            model_bin = os.path.join(ckpt_path, "pytorch_model.bin")
            
            if os.path.exists(model_bin):
                logger.info(f"🎯 Found latest checkpoint: {latest_ckpt}")
                return model_bin, "checkpoint"
    
    # Fallback: final weights
    final_weights = os.path.join(OUTPUT_DIR, "t3_vi_phoneme_final.safetensors")
    if os.path.exists(final_weights):
        logger.info(f"🎯 Using final weights: {final_weights}")
        return final_weights, "safetensors"
    
    return None, None


FINETUNED_WEIGHTS, WEIGHTS_TYPE = find_latest_checkpoint_weights()

# === EDIT NHỮNG GIÁ TRỊ NÀY ===
TEXT_TO_SAY = "Xin chào, tôi là trợ lý giọng nói tiếng Việt. Hôm nay thời tiết rất đẹp."
AUDIO_PROMPT = "speaker_reference/vui-ve.wav"
OUTPUT_FILE = "output.wav"

PARAMS = {
    "temperature": 0.8,
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "repetition_penalty": 1.2,
}


class ChatterboxPhonemeTokenizerAdapter:
    """Adapter để `engine.generate()` gọi được tokenizer phoneme tùy chỉnh."""

    def __init__(self, phoneme_tokenizer: PhonemeTokenizer):
        self.phoneme_tokenizer = phoneme_tokenizer

    def text_to_tokens(self, text: str) -> torch.Tensor:
        # generate() sẽ tự thêm BOS/EOS, nên không thêm special token ở đây.
        ids = self.phoneme_tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], dtype=torch.long)


def load_finetuned_engine(device):
    """Load Chatterbox engine + thay T3 bằng version fine-tune."""
    # === Patch perth before importing chatterbox ===
    import sys, os
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    import src.patch_perth  # noqa: F401
    # ================================================
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.models.t3.t3 import T3

    logger.info(f"Loading base Chatterbox from: {BASE_MODEL_DIR}")
    tts_engine = ChatterboxTTS.from_local(BASE_MODEL_DIR, device="cpu")

    # Reconstruct T3 với vocab mới
    phoneme_tokenizer = PhonemeTokenizer.load(cfg.tokenizer_path)
    logger.info(f"Building new T3 with vocab_size={cfg.new_vocab_size}")
    hp = tts_engine.t3.hp
    hp.text_tokens_dict_size = cfg.new_vocab_size
    hp.start_text_token = phoneme_tokenizer.bos_id
    hp.stop_text_token = phoneme_tokenizer.eos_id
    new_t3 = T3(hp=hp)

    # Load fine-tuned weights
    if not FINETUNED_WEIGHTS:
        logger.error(f"No fine-tuned weights found in {OUTPUT_DIR}")
        raise FileNotFoundError(f"No checkpoint or final weights in {OUTPUT_DIR}")
    
    # Load dựa trên format
    if WEIGHTS_TYPE == "checkpoint":
        # pytorch_model.bin từ checkpoint
        logger.info(f"Loading checkpoint model: {FINETUNED_WEIGHTS}")
        try:
            sd = torch.load(FINETUNED_WEIGHTS, map_location="cpu", weights_only=False)
            # Bỏ wrapper prefix nếu có
            if all(k.startswith("model.") for k in sd.keys()):
                sd = {k[6:]: v for k, v in sd.items()}
            new_t3.load_state_dict(sd, strict=False)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    elif WEIGHTS_TYPE == "safetensors":
        # safetensors final weights
        logger.info(f"Loading safetensors model: {FINETUNED_WEIGHTS}")
        sd = load_file(FINETUNED_WEIGHTS, device="cpu")
        new_t3.load_state_dict(sd, strict=True)
    
    # LoRA adapter fallback
    else:
        lora_path = os.path.join(OUTPUT_DIR, "lora_adapter")
        if os.path.exists(lora_path):
            logger.info(f"Loading LoRA adapter from: {lora_path}")
            from peft import PeftModel
            new_t3 = PeftModel.from_pretrained(new_t3, lora_path)
        else:
            raise FileNotFoundError(f"No weights found in {OUTPUT_DIR}")

    tts_engine.t3 = new_t3
    tts_engine.t3.to(device).eval()
    tts_engine.s3gen.to(device).eval()
    tts_engine.ve.to(device).eval()
    tts_engine.device = device

    return tts_engine


def generate_sentence(engine, tokenizer, text, prompt_path, **kwargs):
    """Generate audio cho 1 câu, trả về (sample_rate, wav_np)."""
    try:
        phoneme_str = vi_text_to_phonemes(
            text, dialect=cfg.dialect, tone_format=cfg.tone_format
        )
        logger.info(f"  Phoneme: {phoneme_str[:80]}{'...' if len(phoneme_str) > 80 else ''}")

        wav_tensor = engine.generate(
            text=phoneme_str,
            audio_prompt_path=prompt_path,
            **kwargs,
        )

        wav_np = wav_tensor.squeeze().cpu().numpy()
        try:
            wav_trimmed = trim_silence_with_vad(wav_np, engine.sr)
        except Exception as vad_err:
            logger.warning(f"VAD trim skipped: {vad_err}")
            wav_trimmed = wav_np
        return engine.sr, wav_trimmed

    except Exception as e:
        logger.error(f"Error on '{text[:30]}...': {e}")
        return cfg.sample_rate, np.zeros(0, dtype=np.float32)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    logger.info(f"Device: {DEVICE}")

    if not os.path.exists(AUDIO_PROMPT):
        logger.error(f"Audio prompt not found: {AUDIO_PROMPT}")
        logger.error("Đặt 1 file .wav giọng tham chiếu (3-10s, giọng Nam) tại đường dẫn này.")
        return

    engine = load_finetuned_engine(DEVICE)
    tokenizer = PhonemeTokenizer.load(cfg.tokenizer_path)
    engine.tokenizer = ChatterboxPhonemeTokenizerAdapter(tokenizer)

    # Split câu
    sentences = re.split(r"(?<=[.?!])\s+", TEXT_TO_SAY.strip())
    sentences = [s for s in sentences if s.strip()]
    logger.info(f"Synthesizing {len(sentences)} sentences...")

    set_seed(42)

    chunks = []
    sr = cfg.sample_rate

    for i, sent in enumerate(sentences):
        logger.info(f"[{i+1}/{len(sentences)}] {sent}")
        sr, wav = generate_sentence(engine, tokenizer, sent, AUDIO_PROMPT, **PARAMS)
        if len(wav) > 0:
            chunks.append(wav)
            # Pause 200ms giữa các câu
            chunks.append(np.zeros(int(sr * 0.2), dtype=np.float32))

    if chunks:
        final = np.concatenate(chunks)
        sf.write(OUTPUT_FILE, final, sr)
        logger.info(f"Saved: {OUTPUT_FILE} ({len(final)/sr:.2f}s)")
    else:
        logger.error("No audio generated")


if __name__ == "__main__":
    main()
