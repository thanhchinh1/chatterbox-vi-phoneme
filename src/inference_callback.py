"""
Callback sinh sample audio mỗi N steps để theo dõi tiến độ training.
"""
import os
import torch
import numpy as np
import soundfile as sf
from transformers import TrainerCallback


class InferenceCallback(TrainerCallback):
    """Sinh audio sample định kỳ trong quá trình training."""

    def __init__(self, cfg, tts_engine_ref=None):
        self.cfg = cfg
        self.tts_engine = tts_engine_ref  # reference, để gọi inference được
        self.sample_dir = os.path.join(cfg.output_dir, "samples")
        os.makedirs(self.sample_dir, exist_ok=True)

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

        # Set eval mode tạm thời
        self.tts_engine.t3.eval()

        with torch.no_grad():
            phoneme_str = vi_text_to_phonemes(
                text, dialect=self.cfg.dialect,
                tone_format=self.cfg.tone_format,
            )
            tokenizer = PhonemeTokenizer.load(self.cfg.tokenizer_path)
            token_ids = tokenizer.encode(phoneme_str)

            # Generate (API có thể khác tuỳ Chatterbox version)
            wav = self.tts_engine.generate(
                text_token_ids=token_ids,
                audio_prompt_path=prompt,
                temperature=0.8,
                exaggeration=0.5,
                cfg_weight=0.5,
            )

        out_path = os.path.join(self.sample_dir, f"step_{step:08d}.wav")
        wav_np = wav.squeeze().cpu().numpy()
        sf.write(out_path, wav_np, self.tts_engine.sr)
        print(f"[InferenceCallback] Saved sample: {out_path}")

        self.tts_engine.t3.train()
