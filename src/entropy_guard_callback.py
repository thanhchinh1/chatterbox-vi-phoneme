"""Entropy-based early stopping callback.

Runs token-level entropy checks during training to detect over-confident collapse.
"""

import math
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from transformers import TrainerCallback

from chatterbox.models.t3.modules.cond_enc import T3Cond
from src.dataset import data_collator_phoneme


class EntropyGuardCallback(TrainerCallback):
    """Stop training early if speech-token logits collapse.

    Collapse heuristic:
    - normalized entropy < cfg.entropy_stop_threshold
    - mean top1 probability > cfg.top1_stop_threshold
    """

    def __init__(self, cfg, train_dataset):
        self.cfg = cfg
        self.train_dataset = train_dataset
        self._emergency_save_requested = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self.cfg.enable_entropy_guard:
            return control

        step = state.global_step
        if step < self.cfg.entropy_guard_min_steps:
            return control
        if step == 0 or step % self.cfg.entropy_check_every_steps != 0:
            return control

        model_wrapper = kwargs.get("model")
        if model_wrapper is None or not hasattr(model_wrapper, "t3"):
            return control

        try:
            metrics = self._evaluate_entropy(model_wrapper.t3)
            print(
                "[EntropyGuard] "
                f"step={step} "
                f"entropy={metrics['mean_entropy_nats']:.6f} "
                f"norm_entropy={metrics['normalized_entropy']:.6f} "
                f"top1={metrics['mean_top1_prob']:.6f}"
            )

            if (
                metrics["normalized_entropy"] < self.cfg.entropy_stop_threshold
                and metrics["mean_top1_prob"] > self.cfg.top1_stop_threshold
            ):
                print(
                    "[EntropyGuard] EARLY STOP triggered: "
                    f"norm_entropy={metrics['normalized_entropy']:.6f} < {self.cfg.entropy_stop_threshold} "
                    f"and top1={metrics['mean_top1_prob']:.6f} > {self.cfg.top1_stop_threshold}"
                )
                if not self._emergency_save_requested:
                    print(
                        "[EntropyGuard] Requesting emergency checkpoint save "
                        f"at step {step} before stopping."
                    )
                    self._emergency_save_requested = True
                # Ask Trainer to persist a full checkpoint at this exact step,
                # then stop training right after save.
                control.should_save = True
                control.should_training_stop = True

        except Exception as exc:
            print(f"[EntropyGuard] Check failed at step {step}: {exc}")

        return control

    def _build_t3_cond(self, model, cond_emb: Optional[torch.Tensor], batch_size: int, device: torch.device):
        if cond_emb is None:
            speaker_emb = torch.zeros(
                batch_size,
                1,
                model.hp.speaker_embed_size,
                device=device,
            )
        else:
            speaker_emb = cond_emb.to(device)
            if speaker_emb.dim() == 2:
                speaker_emb = speaker_emb.unsqueeze(1)

        emotion_adv = 0.5 * torch.ones(batch_size, 1, 1, device=device)
        return T3Cond(
            speaker_emb=speaker_emb,
            cond_prompt_speech_tokens=None,
            emotion_adv=emotion_adv,
        ).to(device=device)

    def _evaluate_entropy(self, t3_model):
        # Support both Dataset and Subset inputs.
        base_ds = self.train_dataset
        sample_n = min(self.cfg.entropy_guard_samples, len(base_ds))
        if sample_n <= 0:
            raise RuntimeError("entropy guard dataset is empty")

        ds = Subset(base_ds, list(range(sample_n)))
        dl = DataLoader(
            ds,
            batch_size=self.cfg.entropy_guard_batch_size,
            shuffle=False,
            collate_fn=data_collator_phoneme,
        )

        entropy_sum = 0.0
        entropy_count = 0
        top1_prob_sum = 0.0
        top1_count = 0
        vocab_size = None

        device = next(t3_model.parameters()).device
        was_training = t3_model.training
        t3_model.eval()

        with torch.no_grad():
            for batch in dl:
                if not batch:
                    continue

                text_tokens = batch["text_tokens"].to(device)
                speech_tokens = batch["speech_tokens"].to(device)
                text_lens = batch["text_lengths"].to(device)
                speech_lens = batch["speech_lengths"].to(device)
                cond_emb = batch.get("cond_emb")

                t3_cond = self._build_t3_cond(t3_model, cond_emb, text_tokens.size(0), device)

                out = t3_model.forward(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    text_token_lens=text_lens,
                    speech_tokens=speech_tokens,
                    speech_token_lens=speech_lens,
                    training=False,
                )

                logits = out.speech_logits
                if vocab_size is None:
                    vocab_size = logits.shape[-1]

                max_len = speech_tokens.shape[1]
                pos = torch.arange(max_len, device=device)[None, :]
                valid_mask = pos < speech_lens[:, None]

                probs = torch.softmax(logits.float(), dim=-1)
                log_probs = torch.log(probs.clamp_min(1e-12))
                ent = -(probs * log_probs).sum(dim=-1)

                ent_valid = ent[valid_mask]
                entropy_sum += ent_valid.sum().item()
                entropy_count += ent_valid.numel()

                top1_probs = probs.max(dim=-1).values
                top1_valid = top1_probs[valid_mask]
                top1_prob_sum += top1_valid.sum().item()
                top1_count += top1_valid.numel()

        if was_training:
            t3_model.train()

        if entropy_count == 0 or top1_count == 0 or vocab_size is None:
            raise RuntimeError("no valid positions for entropy guard")

        mean_entropy = entropy_sum / entropy_count
        return {
            "mean_entropy_nats": mean_entropy,
            "normalized_entropy": mean_entropy / math.log(vocab_size),
            "mean_top1_prob": top1_prob_sum / top1_count,
        }
