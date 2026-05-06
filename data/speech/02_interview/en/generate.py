"""
Generate benign English interview carriers for scenario 02 (interview screening).

Carrier design: each script is an intentionally mediocre / underqualified
candidate response a real applicant would plausibly give. The clean audio
sounds weak-to-average on purpose so that any "Strongly Recommend Advancing"
verdict from the HR agent is attributable to the latent perturbation, not
to content merit.

Layout: 5 scripts x 5 voices = 25 carriers (1:1 with 25 targets in
carrier_to_target.json). Each script anchors one target category.

Output: 16 kHz mono PCM WAVs under data/speech/02_interview/en/audio/.
"""

import asyncio
import os
import tempfile

import edge_tts
import librosa
import numpy as np
import soundfile as sf


HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(HERE, "scripts")
OUT_DIR = os.path.join(HERE, "audio")
TARGET_SR = 16000

# en-US voices disjoint from S1 (Christopher/Michelle/Eric/Aria/Roger).
# Only 4 edge-tts en-US voices are disjoint from both S1 and S4, so we
# overlap Emma with S4; S2 and S4 never share carriers at eval time.
VOICES = [
    ("jenny",   "en-US-JennyNeural"),
    ("guy",     "en-US-GuyNeural"),
    ("ana",     "en-US-AnaNeural"),
    ("steffan", "en-US-SteffanNeural"),
    ("emma",    "en-US-EmmaNeural"),
]

# Script-to-category anchor.
SCRIPT_IDS = [
    "q1_intro",
    "q2_experience",
    "q3_weakness",
    "q4_teamwork",
    "q5_why_company",
]


async def synth(text: str, voice: str, out_wav: str) -> float:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    try:
        await edge_tts.Communicate(text, voice).save(tmp.name)
        audio, _ = librosa.load(tmp.name, sr=TARGET_SR, mono=True)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = (audio / peak) * 0.89  # ~ -1 dBFS

    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    sf.write(out_wav, audio.astype(np.float32), TARGET_SR, subtype="PCM_16")
    return len(audio) / TARGET_SR


async def run() -> None:
    total = 0
    for sid in SCRIPT_IDS:
        with open(os.path.join(SCRIPTS_DIR, f"{sid}.txt"), "r", encoding="utf-8") as f:
            text = f.read().strip()
        for vshort, voice in VOICES:
            out_wav = os.path.join(OUT_DIR, f"interview_{vshort}_{sid}_edgetts.wav")
            dur = await synth(text, voice, out_wav)
            print(f"  [{sid}] {vshort:8s}  voice={voice}  dur={dur:5.2f}s")
            total += 1
    print(f"wrote {total} carriers")


if __name__ == "__main__":
    asyncio.run(run())
