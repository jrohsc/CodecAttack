"""
Validate EnCodec proxy as approximation of real Opus/AAC codec.

Takes existing adversarial audio files, runs them through:
  1. EnCodec@{1.5, 3.0, 6.0} kbps (our proxy)
  2. Opus@{64, 128} kbps (real codec)

Compares:
  - Spectral similarity (mel distance) between proxy and real codec outputs
  - Model output agreement (does the model produce the same text?)
  - Success agreement (do both preserve the adversarial target?)

Pass criterion: >70% success agreement between EnCodec@1.5kbps and Opus@128kbps.

Usage:
    python validate_codec_proxy.py --audio-dir results_qa_robust/.../audio/ \\
        --target-model qwen2_audio --target-text "Turn left"

    python validate_codec_proxy.py --audio-files adv1.wav adv2.wav \\
        --target-model qwen2_audio --target-text "Turn left"
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile

import torch
import torchaudio
import numpy as np
import soundfile as sf
from encodec import EncodecModel

from config import (
    WHISPER_INJECT_ROOT, TARGET_MODEL, TARGET_SAMPLE_RATE,
    ENCODEC_SAMPLE_RATE, PROMPT_MODES, DEFAULT_PROMPT_MODE,
)
from music_carrier import mel_distance

sys.path.insert(0, WHISPER_INJECT_ROOT)


def encode_decode_encodec(audio_24k: torch.Tensor, model: EncodecModel,
                          bandwidth: float) -> torch.Tensor:
    """Run audio through EnCodec encode→decode at given bandwidth."""
    model.set_target_bandwidth(bandwidth)
    with torch.no_grad():
        frames = model.encode(audio_24k)
        decoded = model.decode(frames)
    return decoded[..., :audio_24k.shape[-1]]


def encode_decode_opus(audio_np: np.ndarray, sr: int, bitrate: int) -> np.ndarray:
    """Run audio through real Opus codec via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
        sf.write(f_in.name, audio_np, sr)
        in_path = f_in.name

    with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as f_opus:
        opus_path = f_opus.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        out_path = f_out.name

    try:
        # Encode to Opus
        subprocess.run([
            "ffmpeg", "-y", "-i", in_path,
            "-c:a", "libopus", "-b:a", f"{bitrate}k",
            "-ar", str(sr), opus_path
        ], capture_output=True, check=True)

        # Decode back to WAV
        subprocess.run([
            "ffmpeg", "-y", "-i", opus_path,
            "-ar", str(sr), out_path
        ], capture_output=True, check=True)

        audio_out, _ = sf.read(out_path)
        return audio_out.astype(np.float32)
    finally:
        for p in [in_path, opus_path, out_path]:
            if os.path.exists(p):
                os.unlink(p)


def load_audio_file(path: str, target_sr: int = ENCODEC_SAMPLE_RATE) -> torch.Tensor:
    """Load audio file and resample to target_sr. Returns [1, 1, T]."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.unsqueeze(0)  # [1, 1, T]


def main():
    parser = argparse.ArgumentParser(description="Validate EnCodec proxy vs real Opus")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio-dir", type=str,
                       help="Directory containing adversarial WAV files")
    group.add_argument("--audio-files", nargs="+",
                       help="Explicit list of adversarial WAV files")

    parser.add_argument("--target-model", type=str, default=TARGET_MODEL)
    parser.add_argument("--target-text", type=str, default=None,
                        help="Target text to check for (if None, just compare outputs)")
    parser.add_argument("--prompt-mode", type=str, default=DEFAULT_PROMPT_MODE)
    parser.add_argument("--encodec-bandwidths", type=str, default="1.5,3.0,6.0",
                        help="Comma-separated EnCodec bandwidths to test")
    parser.add_argument("--opus-bitrates", type=str, default="64,128",
                        help="Comma-separated Opus bitrates to test")
    parser.add_argument("--max-files", type=int, default=10,
                        help="Max number of files to process")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for results")

    args = parser.parse_args()

    # Gather audio files
    if args.audio_dir:
        wav_files = sorted([
            os.path.join(args.audio_dir, f)
            for f in os.listdir(args.audio_dir)
            if f.endswith(".wav") and "adv" in f.lower()
        ])
    else:
        wav_files = args.audio_files

    if not wav_files:
        print("ERROR: No adversarial WAV files found")
        return

    wav_files = wav_files[:args.max_files]
    print(f"Processing {len(wav_files)} files")

    encodec_bws = [float(b) for b in args.encodec_bandwidths.split(",")]
    opus_brs = [int(b) for b in args.opus_bitrates.split(",")]
    prompt_text = PROMPT_MODES[args.prompt_mode]["prompt"]

    # Load EnCodec
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encodec_model = EncodecModel.encodec_model_24khz().to(device)
    encodec_model.eval()

    # Load target model
    print(f"Loading target model: {args.target_model}")
    if args.target_model == "qwen2_audio":
        from models.qwen2_audio import Qwen2AudioModel
        target_model = Qwen2AudioModel(device=device)
    elif args.target_model == "audio_flamingo":
        from models.audio_flamingo import AudioFlamingoModel
        target_model = AudioFlamingoModel(device=device)
    else:
        print(f"Unsupported model: {args.target_model}")
        return

    results = []

    for fi, wav_path in enumerate(wav_files):
        print(f"\n{'='*60}")
        print(f"[{fi+1}/{len(wav_files)}] {os.path.basename(wav_path)}")
        print(f"{'='*60}")

        audio_24k = load_audio_file(wav_path).to(device)
        audio_np = audio_24k.squeeze().cpu().numpy()

        file_result = {"file": os.path.basename(wav_path), "encodec": {}, "opus": {}}

        # Direct eval (no codec)
        audio_16k = torchaudio.functional.resample(
            audio_24k.squeeze(0).to(device), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
        )
        with torch.no_grad():
            direct_output = target_model.generate(audio_16k, prompt=prompt_text)
        file_result["direct_output"] = direct_output
        direct_match = (args.target_text and
                        args.target_text.lower() in direct_output.lower())
        file_result["direct_match"] = direct_match
        print(f"  Direct: {'OK' if direct_match else 'FAIL'} | {direct_output[:80]}")

        # EnCodec proxy at each bandwidth
        for bw in encodec_bws:
            decoded = encode_decode_encodec(audio_24k, encodec_model, bw)
            decoded_np = decoded.squeeze().cpu().numpy()

            # Mel distance from original
            mel_dist = mel_distance(
                decoded.squeeze(0).cpu(),
                audio_24k.squeeze(0).cpu(),
                sample_rate=ENCODEC_SAMPLE_RATE,
            ).item()

            # Model eval
            dec_16k = torchaudio.functional.resample(
                decoded.squeeze(0).to(device), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            with torch.no_grad():
                enc_output = target_model.generate(dec_16k, prompt=prompt_text)

            enc_match = (args.target_text and
                         args.target_text.lower() in enc_output.lower())

            file_result["encodec"][str(bw)] = {
                "output": enc_output,
                "match": enc_match,
                "mel_distance": mel_dist,
            }
            print(f"  EnCodec@{bw}kbps: {'OK' if enc_match else 'FAIL'} | "
                  f"mel_dist={mel_dist:.4f} | {enc_output[:60]}")

        # Real Opus at each bitrate
        for br in opus_brs:
            opus_np = encode_decode_opus(audio_np, ENCODEC_SAMPLE_RATE, br)
            opus_tensor = torch.FloatTensor(opus_np).unsqueeze(0)  # [1, T]

            # Mel distance from original
            min_len = min(opus_tensor.shape[-1], audio_24k.squeeze(0).shape[-1])
            mel_dist = mel_distance(
                opus_tensor[..., :min_len],
                audio_24k.squeeze(0)[..., :min_len].cpu(),
                sample_rate=ENCODEC_SAMPLE_RATE,
            ).item()

            # Model eval
            opus_16k = torchaudio.functional.resample(
                opus_tensor.to(device), ENCODEC_SAMPLE_RATE, TARGET_SAMPLE_RATE
            )
            with torch.no_grad():
                opus_output = target_model.generate(opus_16k, prompt=prompt_text)

            opus_match = (args.target_text and
                          args.target_text.lower() in opus_output.lower())

            file_result["opus"][str(br)] = {
                "output": opus_output,
                "match": opus_match,
                "mel_distance": mel_dist,
            }
            print(f"  Opus@{br}kbps:    {'OK' if opus_match else 'FAIL'} | "
                  f"mel_dist={mel_dist:.4f} | {opus_output[:60]}")

        results.append(file_result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    n = len(results)

    # Success rates
    direct_ok = sum(1 for r in results if r.get("direct_match"))
    print(f"Direct:          {direct_ok}/{n} ({100*direct_ok/n:.0f}%)")

    for bw in encodec_bws:
        ok = sum(1 for r in results if r["encodec"][str(bw)]["match"])
        print(f"EnCodec@{bw}kbps: {ok}/{n} ({100*ok/n:.0f}%)")

    for br in opus_brs:
        ok = sum(1 for r in results if r["opus"][str(br)]["match"])
        print(f"Opus@{br}kbps:    {ok}/{n} ({100*ok/n:.0f}%)")

    # Agreement: EnCodec vs Opus
    print(f"\nProxy-vs-Opus Agreement:")
    for bw in encodec_bws:
        for br in opus_brs:
            agree = sum(
                1 for r in results
                if r["encodec"][str(bw)]["match"] == r["opus"][str(br)]["match"]
            )
            pct = 100 * agree / n
            label = "PASS" if pct >= 70 else "FAIL"
            print(f"  EnCodec@{bw} vs Opus@{br}: {agree}/{n} ({pct:.0f}%) [{label}]")

    # Average mel distances
    print(f"\nAverage Mel Distances:")
    for bw in encodec_bws:
        avg = np.mean([r["encodec"][str(bw)]["mel_distance"] for r in results])
        print(f"  EnCodec@{bw}kbps: {avg:.4f}")
    for br in opus_brs:
        avg = np.mean([r["opus"][str(br)]["mel_distance"] for r in results])
        print(f"  Opus@{br}kbps:    {avg:.4f}")

    # Save results
    output_path = args.output or "codec_proxy_validation.json"
    with open(output_path, "w") as f:
        json.dump({"files": results, "config": vars(args)}, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
