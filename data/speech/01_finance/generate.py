"""
Generate benign banking-speech carriers for scenario 01 (finance voice agent).

Carrier design (Option 3): each script is a real benign utterance a customer
would say on a banking line. No auth-bypass content, no meta-adversarial
phrasing — the gap between carrier ("check my balance") and attack target
("PIN verified, transferring") must be fully attributable to the latent
perturbation, not to semantic manipulation of the carrier.

Layout: 5 scripts × 5 voices = 25 carriers (1:1 with the 25 targets in
target_drafts/01_finance_targets.json). Each carrier script anchors one
target category.

Output: 16 kHz mono PCM WAVs under data/speech/01_finance/audio/.
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

# Voices disjoint from S2 (Jenny/Guy/Sonia/Ryan/Neerja/Prabhat/Natasha/William/
# SunHi/Alvaro/Xiaoxiao/Yunxi) and S4 (Andrew/Ava/Brian/Emma).
# First 5 are the original n=25 set; the next 5 (added 2026-05-02) double the
# pool to n=50 for the latent-vs-waveform comparison table.
VOICES = [
    ("christopher", "en-US-ChristopherNeural"),
    ("michelle",    "en-US-MichelleNeural"),
    ("eric",        "en-US-EricNeural"),
    ("aria",        "en-US-AriaNeural"),
    ("roger",       "en-US-RogerNeural"),
    ("ana",         "en-US-AnaNeural"),
    ("steffan",     "en-US-SteffanNeural"),
    ("liam",        "en-CA-LiamNeural"),
    ("emily",       "en-IE-EmilyNeural"),
    ("thomas",      "en-GB-ThomasNeural"),
]

# Script-to-category anchor: each script thematically motivates its category.
SCRIPT_IDS = ["q1_balance", "q2_rates", "q3_transactions", "q4_branch", "q5_card"]


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
    skipped = 0
    for sid in SCRIPT_IDS:
        with open(os.path.join(SCRIPTS_DIR, f"{sid}.txt"), "r", encoding="utf-8") as f:
            text = f.read().strip()
        for vshort, voice in VOICES:
            out_wav = os.path.join(OUT_DIR, f"banking_{vshort}_{sid}_edgetts.wav")
            if os.path.isfile(out_wav):
                skipped += 1
                continue
            dur = await synth(text, voice, out_wav)
            print(f"  [{sid}] {vshort:12s}  voice={voice}  dur={dur:5.2f}s")
            total += 1
    print(f"wrote {total} carriers (skipped {skipped} already-present)")


if __name__ == "__main__":
    asyncio.run(run())
