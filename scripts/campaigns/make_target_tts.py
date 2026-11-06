"""
Synthesize spoken renderings of the S1 finance attack targets.

These are the "target command audio" that ZQ-Attack (arXiv:2406.19311) uses for
perturbation initialization, and the source of the target encoder embeddings for
the U-TLSA-style (arXiv:2512.23881) per-frame cosine objective in
encoder_transfer_attack.py.

Voice is deliberately disjoint from the 10 S1 carrier voices so the target audio
cannot be mistaken for a carrier-voice match.

Must run somewhere with network access (edge-tts hits Microsoft's endpoint).
Output: 16 kHz mono PCM WAVs under data/speech/01_finance/target_audio/.
"""

import asyncio
import hashlib
import json
import os
import tempfile

import edge_tts
import librosa
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.join(HERE, "0_all_combined/data/speech/01_finance/carrier_to_target.json")
OUT_DIR = os.path.join(HERE, "0_all_combined/data/speech/01_finance/target_audio")
TARGET_SR = 16000
VOICE = "en-US-JennyNeural"


def target_id(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


async def synth(text: str, out_wav: str) -> float:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    try:
        await edge_tts.Communicate(text, VOICE).save(tmp.name)
        audio, _ = librosa.load(tmp.name, sr=TARGET_SR, mono=True)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    sf.write(out_wav, audio, TARGET_SR, subtype="PCM_16")
    return len(audio) / TARGET_SR


async def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = json.load(open(PAIRS))["pairs"]
    targets = sorted({p["target"] for p in pairs})
    print(f"{len(pairs)} pairs -> {len(targets)} unique targets, voice={VOICE}")

    index = {}
    for i, text in enumerate(targets, 1):
        tid = target_id(text)
        out_wav = os.path.join(OUT_DIR, f"{tid}.wav")
        if os.path.isfile(out_wav):
            dur = sf.info(out_wav).duration
            print(f"[{i}/{len(targets)}] cached  {tid}  {dur:.2f}s  {text[:60]}")
        else:
            dur = await synth(text, out_wav)
            print(f"[{i}/{len(targets)}] synth   {tid}  {dur:.2f}s  {text[:60]}")
        index[text] = {"id": tid, "wav": out_wav, "duration_s": dur}

    with open(os.path.join(OUT_DIR, "index.json"), "w") as f:
        json.dump({"voice": VOICE, "sample_rate": TARGET_SR, "targets": index}, f, indent=1)
    print(f"\nwrote {OUT_DIR}/index.json")


if __name__ == "__main__":
    asyncio.run(main())
