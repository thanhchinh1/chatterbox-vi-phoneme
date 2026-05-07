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
from typing import Dict, Optional


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

    # Tên các key text embedding có thể là:
    #   "text_emb.weight" hoặc "text_token_emb.weight" hoặc "tfmr.wte.weight"
    # Tuỳ vào kiến trúc T3 cụ thể của Chatterbox version. Cần inspect và adapt.
    text_emb_keys = [
        k for k in old_state_dict
        if "text_emb" in k or "text_token_emb" in k or k.endswith(".wte.weight")
    ]

    print(f"[resize_t3] Found text embedding keys: {text_emb_keys}")

    for key in list(old_state_dict.keys()):
        if key not in new_state_dict:
            # Key có trong old nhưng không có trong new → skip (e.g. Turbo wte)
            print(f"[resize_t3] Skip key not in new model: {key}")
            continue

        old_w = old_state_dict[key]
        new_w_shape = new_state_dict[key].shape

        if old_w.shape == new_w_shape:
            new_state_dict[key] = old_w
        elif key in text_emb_keys:
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
        # === IMPLEMENT THEO API T3 CỦA BẠN ===
        # Một template tham khảo:

        outputs = self.t3(
            text_tokens=text_tokens,
            text_mask=text_mask,
            speech_tokens=speech_tokens,
            speech_mask=speech_mask,
            cond_emb=cond_emb,
            return_loss=True,  # T3 nội bộ tính loss
        )

        # T3 trả về (logits, loss) hoặc dict — adapt accordingly
        if isinstance(outputs, dict):
            return outputs
        elif isinstance(outputs, tuple) and len(outputs) >= 2:
            return {"loss": outputs[1], "logits": outputs[0]}
        else:
            raise ValueError(f"Unexpected T3 output: {type(outputs)}")

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
