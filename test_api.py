"""Test Chatterbox API. Run: python test_api.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import src.patch_perth  # MUST be first

import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS

print("Loading...")
engine = ChatterboxTTS.from_local("pretrained_models", device="cuda")
print("Loaded.\n")

wav, sr = torchaudio.load("data/wavs/00000000.wav")
wav_gpu = wav.to("cuda")
print(f"Wav: shape={wav.shape}, sr={sr}\n")

print("=== S3Gen tokenizer output ===")
with torch.no_grad():
    out = engine.s3gen.tokenizer(wav_gpu)
print(f"Type: {type(out).__name__}")
if isinstance(out, tuple):
    print(f"Tuple of {len(out)} items:")
    for i, x in enumerate(out):
        if hasattr(x, "shape"):
            print(f"  [{i}] shape={x.shape}, dtype={x.dtype}, "
                  f"min={x.min().item()}, max={x.max().item()}")
        else:
            print(f"  [{i}] {type(x).__name__}: {x}")
elif hasattr(out, "shape"):
    print(f"Tensor: shape={out.shape}, dtype={out.dtype}")
elif isinstance(out, dict):
    print(f"Dict keys: {list(out.keys())}")
    for k, v in out.items():
        if hasattr(v, "shape"):
            print(f"  '{k}': shape={v.shape}, dtype={v.dtype}")
print()

print("=== VoiceEncoder output ===")
ve_methods = [
    ("engine.ve(wav_gpu)", lambda: engine.ve(wav_gpu)),
    ("engine.ve(wav_gpu.squeeze())", lambda: engine.ve(wav_gpu.squeeze())),
    ("engine.ve.embeds_from_wavs([wav_np])",
     lambda: engine.ve.embeds_from_wavs(
         [wav.squeeze().numpy()], sample_rate=24000)),
]
for name, fn in ve_methods:
    try:
        with torch.no_grad():
            ve_out = fn()
        if hasattr(ve_out, "shape"):
            print(f"✓ {name}: shape={ve_out.shape}, dtype={ve_out.dtype}")
        elif isinstance(ve_out, tuple):
            print(f"✓ {name}: tuple of {len(ve_out)}")
            for i, x in enumerate(ve_out):
                if hasattr(x, "shape"):
                    print(f"    [{i}] shape={x.shape}")
        else:
            print(f"✓ {name}: type={type(ve_out).__name__}")
        break  # Stop ở method đầu tiên work
    except Exception as e:
        print(f"✗ {name}: {type(e).__name__}: {str(e)[:120]}")