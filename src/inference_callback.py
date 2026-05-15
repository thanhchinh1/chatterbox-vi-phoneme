"""
Callback sinh sample audio mỗi N steps để theo dõi tiến độ training.
"""
import os
import torch
import numpy as np
import soundfile as sf
from transformers import TrainerCallback


class _PhonemeTokenizerAdapter:
    """Adapter để engine.generate tokenize theo phoneme vocab mới."""

    def __init__(self, phoneme_tokenizer, device):
        self.phoneme_tokenizer = phoneme_tokenizer
        self.device = device

    def text_to_tokens(self, text: str) -> torch.Tensor:
        # generate() tự thêm BOS/EOS nên không add special tokens ở đây.
        ids = self.phoneme_tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], dtype=torch.long, device=self.device)


class InferenceCallback(TrainerCallback):
    """Sinh audio sample định kỳ trong quá trình training."""

    def __init__(self, cfg, tts_engine_ref=None):
        self.cfg = cfg
        self.tts_engine = tts_engine_ref  # reference, để gọi inference được
        self.sample_dir = os.path.join(cfg.output_dir, "samples")
        os.makedirs(self.sample_dir, exist_ok=True)
        self._adapter_ready = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self.cfg.is_inference:
            return

        if state.global_step == 0 or state.global_step % self.cfg.inference_every_steps != 0:
            return

        if self.tts_engine is None:
            return

        try:
            self._generate_sample(state.global_step)
        except Exception as e:
            print(f"[InferenceCallback] Error at step {state.global_step}: {e}")

    def _generate_sample(self, step: int):
        from src.vi_text_processor import vi_text_to_phonemes
        from src.phoneme_tokenizer import PhonemeTokenizer

        text = self.cfg.inference_text
        prompt = self.cfg.inference_audio_prompt

        if not os.path.exists(prompt):
            return

        # Set eval mode tạm thời.
        # During training, T3 is on GPU but VE/S3Gen may still be on CPU,
        # so generation needs all inference modules on the same device.
        device = next(self.tts_engine.t3.parameters()).device
        prev_s3gen_device = next(self.tts_engine.s3gen.parameters()).device
        prev_ve_device = next(self.tts_engine.ve.parameters()).device
        self.tts_engine.t3.eval()
        self.tts_engine.s3gen.to(device).eval()
        self.tts_engine.ve.to(device).eval()
        self.tts_engine.device = device

        # Ensure engine uses phoneme tokenizer for generation.
        if not self._adapter_ready:
            tokenizer = PhonemeTokenizer.load(self.cfg.tokenizer_path)
            self.tts_engine.tokenizer = _PhonemeTokenizerAdapter(tokenizer, device=device)
            self._adapter_ready = True

        try:
            with torch.no_grad():
                phoneme_str = vi_text_to_phonemes(
                    text, dialect=self.cfg.dialect,
                    tone_format=self.cfg.tone_format,
                )
                # Generate using current engine API (text, not text_token_ids).
                wav = self.tts_engine.generate(
                    text=phoneme_str,
                    audio_prompt_path=prompt,
                    temperature=0.8,
                    exaggeration=0.5,
                    cfg_weight=0.5,
                )

            out_path = os.path.join(self.sample_dir, f"step_{step:08d}.wav")
            wav_np = wav.squeeze().cpu().numpy()
            sf.write(out_path, wav_np, self.tts_engine.sr)
            print(f"[InferenceCallback] Saved sample: {out_path}")
        finally:
            self.tts_engine.s3gen.to(prev_s3gen_device).eval()
            self.tts_engine.ve.to(prev_ve_device).eval()
            self.tts_engine.t3.train()
