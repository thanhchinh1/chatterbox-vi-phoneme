import inspect
import os
import sys
import copy

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.patch_perth  # noqa: F401
from src.config import TrainConfig
from src.dataset import ChatterboxViDataset, data_collator_phoneme
from src.model import T3TrainerWrapper, resize_t3_text_embedding
from src.phoneme_tokenizer import PhonemeTokenizer
from chatterbox.models.t3.t3 import T3
from chatterbox.tts import ChatterboxTTS


def main():
    cfg = TrainConfig()
    engine = ChatterboxTTS.from_local(cfg.model_dir, device="cpu")

    print("SIGNATURE", inspect.signature(T3.forward))

    sd = engine.t3.state_dict()
    emb_keys = [
        (k, tuple(v.shape))
        for k, v in sd.items()
        if "emb" in k.lower() or k.endswith("wte.weight")
    ]
    print("EMB_KEYS_COUNT", len(emb_keys))
    for k, shape in emb_keys:
        print("EMB_KEY", k, shape)

    tok = PhonemeTokenizer.load(cfg.tokenizer_path)
    new_hp = copy.deepcopy(engine.t3.hp)
    new_hp.text_tokens_dict_size = len(tok.vocab)
    new_hp.start_text_token = tok.bos_id
    new_hp.stop_text_token = tok.eos_id
    if hasattr(new_hp, "use_cache"):
        new_hp.use_cache = False

    new_t3 = T3(hp=new_hp)
    new_t3 = resize_t3_text_embedding(
        new_t3=new_t3,
        old_state_dict=sd,
        new_vocab_size=len(tok.vocab),
        use_warmup=False,
        new_vocab=tok.vocab,
        old_tokenizer=engine.tokenizer,
    )

    resized_emb_keys = [
        (k, tuple(v.shape))
        for k, v in new_t3.state_dict().items()
        if "emb" in k.lower() or k.endswith("wte.weight")
    ]
    print("RESIZED_EMB_KEYS_COUNT", len(resized_emb_keys))
    for k, shape in resized_emb_keys:
        print("RESIZED_EMB_KEY", k, shape)

    train_ds = ChatterboxViDataset(cfg, tts_engine=None)
    samples = [train_ds[i] for i in range(min(2, len(train_ds)))]
    batch = data_collator_phoneme(samples)

    print("BATCH_KEYS", sorted(batch.keys()))
    print("TEXT_SHAPE", tuple(batch["text_tokens"].shape))
    print("SPEECH_SHAPE", tuple(batch["speech_tokens"].shape))
    if "cond_emb" in batch:
        print("COND_SHAPE", tuple(batch["cond_emb"].shape))

    wrapper = T3TrainerWrapper(new_t3)
    wrapper.train()

    out = wrapper(**batch)
    print("OUT_TYPE", type(out).__name__)
    if isinstance(out, dict):
        print("OUT_KEYS", sorted(out.keys()))

    loss = out.get("loss") if isinstance(out, dict) else None
    print("LOSS_IS_TENSOR", torch.is_tensor(loss))
    if not torch.is_tensor(loss):
        print("OPT_STEP", "skipped_no_loss_tensor")
        return

    print("LOSS_VAL", float(loss.detach().cpu()))
    opt = torch.optim.AdamW(wrapper.parameters(), lr=1e-5)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    print("OPT_STEP", "ok")


if __name__ == "__main__":
    main()
