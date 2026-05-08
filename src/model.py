"""
Wrapper class cho T3 fine-tuning:
  1. Resize text embedding từ vocab cũ (~2454) → vocab mới (~120 phoneme)
  2. Embedding warmup từ grapheme tương tự
  3. Freeze VE, S3Gen — chỉ train T3
  4. Wrapper compatible với HuggingFace Trainer

T3 trong Chatterbox là LM (Llama-based) sinh speech tokens từ:
  - text tokens (cái ta thay)
  - speaker conditioning embedding
  - prompt speech tokens (CFG, voice clone)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from chatterbox.models.t3.modules.cond_enc import T3Cond


def resize_t3_text_embedding(
    new_t3,
    old_state_dict: Dict[str, torch.Tensor],
    new_vocab_size: int,
    use_warmup: bool = True,
    new_vocab: Optional[Dict[str, int]] = None,
    old_tokenizer=None,
    ipa_to_grapheme: Optional[Dict] = None,
):
    """Load weights từ pretrained T3 vào new_t3, resize text embedding cho khớp vocab mới.

    Args:
        new_t3: T3 instance đã được tạo với hp.text_tokens_dict_size = new_vocab_size.
        old_state_dict: state_dict của T3 pretrained (lấy từ ChatterboxTTS).
        new_vocab_size: kích thước vocab mới (= len(phoneme_tokenizer)).
        use_warmup: nếu True, init embedding mới bằng warmup từ grapheme.
        new_vocab: dict {phoneme: id} (cần khi use_warmup=True).
        old_tokenizer: tokenizer cũ (cần khi use_warmup=True).
        ipa_to_grapheme: mapping cho warmup (optional, dùng default nếu None).

    Returns:
        new_t3 với weights đã load.
    """
    new_state_dict = new_t3.state_dict()

    # Chatterbox 0.1.2: text branch cần resize đồng thời embedding + head.
    text_resize_keys = [k for k in old_state_dict if k in {"text_emb.weight", "text_head.weight"}]
    if not text_resize_keys:
        # Fallback cho các version khác.
        text_resize_keys = [
            k for k in old_state_dict
            if "text_emb" in k or "text_token_emb" in k or k.endswith("text_head.weight")
        ]

    print(f"[resize_t3] Found text resize keys: {text_resize_keys}")

    for key in list(old_state_dict.keys()):
        if key not in new_state_dict:
            # Key có trong old nhưng không có trong new → skip (e.g. Turbo wte)
            print(f"[resize_t3] Skip key not in new model: {key}")
            continue

        old_w = old_state_dict[key]
        new_w_shape = new_state_dict[key].shape

        if old_w.shape == new_w_shape:
            new_state_dict[key] = old_w
        elif key in text_resize_keys:
            # Đây là text embedding — cần resize
            print(f"[resize_t3] Resizing {key}: {old_w.shape} → {new_w_shape}")
            new_emb = torch.zeros_like(new_state_dict[key])

            if use_warmup and new_vocab is not None and old_tokenizer is not None:
                from src.embedding_warmup import warmup_embedding
                new_emb = warmup_embedding(
                    new_emb=new_emb,
                    old_emb=old_w,
                    new_vocab=new_vocab,
                    old_tokenizer=old_tokenizer,
                    ipa_to_grapheme=ipa_to_grapheme,
                )
            else:
                # Fallback: random init theo distribution của old
                new_emb.normal_(mean=old_w.mean().item(), std=old_w.std().item())

            new_state_dict[key] = new_emb
        else:
            # Layer khác bị mismatch → cảnh báo
            print(f"[resize_t3] WARNING: shape mismatch on {key}: {old_w.shape} vs {new_w_shape}")
            # Có thể là output projection layer cũng cần resize song song với embedding
            # Tuỳ kiến trúc T3 cụ thể

    new_t3.load_state_dict(new_state_dict, strict=False)
    return new_t3


class T3TrainerWrapper(nn.Module):
    """Wrapper cho HuggingFace Trainer.

    Trainer expect model có forward(**inputs) → output có .loss attribute.
    Adapter này nhận batch từ data_collator_phoneme và gọi T3 đúng cách.
    """

    def __init__(self, t3_model):
        super().__init__()
        self.t3 = t3_model

    def _build_t3_cond(self, cond_emb: Optional[torch.Tensor], batch_size: int, device: torch.device):
        if cond_emb is None:
            speaker_emb = torch.zeros(
                batch_size,
                1,
                self.t3.hp.speaker_embed_size,
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

    def forward(self, text_tokens=None, speech_tokens=None,
                text_mask=None, speech_mask=None,
                text_lengths=None, speech_lengths=None,
                cond_emb=None, **kwargs):
        """
        T3 forward signature có thể khác tuỳ version Chatterbox.
        Đây là một template — bạn cần adapt theo API thực tế.

        Logic chuẩn:
          1. Concat: [BOS, text_tokens, EOS_text, speech_tokens, EOS_speech]
          2. Forward qua T3 (LM)
          3. Compute cross-entropy loss CHỈ trên speech_tokens portion
             (để model học predict speech từ text + prompt)
        """
        if text_tokens is None or speech_tokens is None:
            raise ValueError("text_tokens and speech_tokens are required")

        device = text_tokens.device
        text_tokens = text_tokens.to(device=device, dtype=torch.long)
        speech_tokens = speech_tokens.to(device=device, dtype=torch.long)

        if text_lengths is None:
            if text_mask is not None:
                text_lengths = text_mask.long().sum(dim=1)
            else:
                text_lengths = torch.full(
                    (text_tokens.size(0),),
                    text_tokens.size(1),
                    dtype=torch.long,
                    device=device,
                )
        else:
            text_lengths = text_lengths.to(device=device, dtype=torch.long)

        if speech_lengths is None:
            if speech_mask is not None:
                speech_lengths = speech_mask.long().sum(dim=1)
            else:
                speech_lengths = torch.full(
                    (speech_tokens.size(0),),
                    speech_tokens.size(1),
                    dtype=torch.long,
                    device=device,
                )
        else:
            speech_lengths = speech_lengths.to(device=device, dtype=torch.long)

        t3_cond = self._build_t3_cond(cond_emb, text_tokens.size(0), device)
        out = self.t3.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_lengths,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_lengths,
            training=True,
        )

        ignore_id = -100
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        mask_text = torch.arange(len_text, device=device)[None] >= text_lengths[:, None]
        mask_speech = torch.arange(len_speech, device=device)[None] >= speech_lengths[:, None]
        masked_text = text_tokens.masked_fill(mask_text, ignore_id)
        masked_speech = speech_tokens.masked_fill(mask_speech, ignore_id)
        loss_text = F.cross_entropy(
            out.text_logits.permute(0, 2, 1),
            masked_text,
            ignore_index=ignore_id,
        )
        loss_speech = F.cross_entropy(
            out.speech_logits.permute(0, 2, 1),
            masked_speech,
            ignore_index=ignore_id,
        )
        # Ưu tiên speech objective, vẫn log text loss để debug convergence.
        total_loss = loss_speech

        return {
            "loss": total_loss,
            "loss_speech": loss_speech.detach(),
            "loss_text": loss_text.detach(),
        }

    def gradient_checkpointing_enable(self, **kwargs):
        """Forward to T3 if it supports it."""
        if hasattr(self.t3, "gradient_checkpointing_enable"):
            self.t3.gradient_checkpointing_enable(**kwargs)
        elif hasattr(self.t3, "tfmr") and hasattr(self.t3.tfmr, "gradient_checkpointing_enable"):
            self.t3.tfmr.gradient_checkpointing_enable(**kwargs)


def freeze_non_t3_components(tts_engine):
    """Freeze VoiceEncoder và S3Gen, chỉ T3 có gradient."""
    print("[freeze] Freezing VoiceEncoder...")
    for p in tts_engine.ve.parameters():
        p.requires_grad = False

    print("[freeze] Freezing S3Gen...")
    for p in tts_engine.s3gen.parameters():
        p.requires_grad = False

    print("[freeze] Enabling T3 training...")
    tts_engine.t3.train()
    for p in tts_engine.t3.parameters():
        p.requires_grad = True


def maybe_apply_lora(t3_model, cfg):
    """Wrap T3 với LoRA nếu cfg.use_lora=True. Trả về model (đã wrap hoặc không)."""
    if not cfg.use_lora:
        return t3_model

    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError("Cần install peft: pip install peft==0.17.1")

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,  # T3 là LM
    )

    print(f"[LoRA] Applying LoRA: rank={cfg.lora_rank}, alpha={cfg.lora_alpha}")
    print(f"[LoRA] Target modules: {cfg.lora_target_modules}")

    t3_model = get_peft_model(t3_model, lora_config)
    t3_model.print_trainable_parameters()

    return t3_model
