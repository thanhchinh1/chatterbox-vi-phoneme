import argparse
import os
import csv
import sys
import torch
from pathlib import Path
from tqdm import tqdm
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import TrainConfig
from src.utils import setup_logger, check_pretrained_models

logger = setup_logger("PrecomputeTokens")


def _first_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--wavs_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    args = parser.parse_args()

    cfg = TrainConfig()
    metadata_path = args.metadata or cfg.metadata_csv
    wavs_dir = Path(args.wavs_dir or cfg.wavs_dir)

    if not check_pretrained_models(cfg.model_dir):
        sys.exit(1)

    # Load engine để dùng s3gen + ve
    logger.info("Loading Chatterbox engine...")
    # === Patch perth before importing chatterbox ===
    import src.patch_perth  # noqa: F401
    # ================================================
    from chatterbox.tts import ChatterboxTTS, S3_SR
    engine = ChatterboxTTS.from_local(cfg.model_dir, device=args.device)
    engine.s3gen.eval()
    engine.ve.eval()

    # Iterate
    with open(metadata_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="|"))

    n_done = 0
    n_skip = 0
    n_err = 0

    with torch.no_grad():
        for row in tqdm(rows, desc="Precomputing"):
            if len(row) < 1:
                continue
            fname = row[0]
            wav_path = wavs_dir / fname
            if not wav_path.suffix:
                wav_path = wav_path.with_suffix(".wav")

            if not wav_path.exists():
                continue

            tokens_path = wav_path.with_suffix(".speech_tokens.pt")
            cond_path = wav_path.with_suffix(".cond.pt")

            if args.skip_existing and tokens_path.exists() and cond_path.exists():
                n_skip += 1
                continue

            try:
                wav, sr = torchaudio.load(str(wav_path))
                if sr != cfg.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, cfg.sample_rate)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                wav = wav.to(args.device)
                wav_16k = wav
                if cfg.sample_rate != S3_SR:
                    wav_16k = torchaudio.functional.resample(wav, cfg.sample_rate, S3_SR)

                # S3Gen tokenizer
                # tokenizer() returns a tuple: (tokens, lengths)
                speech_tokens = _first_tensor(engine.s3gen.tokenizer(wav_16k)).cpu().squeeze(0).long()
                torch.save(speech_tokens, str(tokens_path))

                # Voice encoder expects mel features; use the raw-audio helper.
                cond = torch.from_numpy(
                    engine.ve.embeds_from_wavs([wav_16k.squeeze(0).cpu().numpy()], sample_rate=S3_SR)
                ).float()
                torch.save(cond, str(cond_path))

                n_done += 1

            except Exception as e:
                logger.warning(f"Error on {fname}: {e}")
                n_err += 1
                continue

    logger.info(f"Done! Computed: {n_done}, Skipped: {n_skip}, Errors: {n_err}")


if __name__ == "__main__":
    main()
