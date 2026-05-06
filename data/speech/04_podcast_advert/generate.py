"""
Generate podcast-monologue and ad-spot speech carriers for scenario 04
(simulated OTA / physical agent). Modeled on dataset/interview/generate_tts.py.

Output: 16 kHz mono PCM WAVs under data/speech/04_podcast_advert/audio/.
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

# (script_stem, carrier_style, voice)
# Voices chosen to be distinct from scenarios 01/02 (Jenny, Guy, Sonia, Neerja,
# Prabhat, Xiaoxiao, Yunxi) so carrier type is acoustically separable.
PLAN = [
    ("podcast_1", "podcast", "en-US-AndrewNeural"),      # casual male podcast host
    ("podcast_2", "podcast", "en-US-AvaNeural"),          # conversational female narrator
    ("advert_1",  "advert",  "en-US-BrianNeural"),        # warm commercial male
    ("advert_2",  "advert",  "en-US-EmmaNeural"),         # upbeat commercial female
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
    for stem, style, voice in PLAN:
        script_path = os.path.join(SCRIPTS_DIR, f"{stem}.txt")
        with open(script_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        out_wav = os.path.join(OUT_DIR, f"{style}_{stem}_edgetts.wav")
        print(f"[{stem}] voice={voice}  style={style}")
        dur = await synth(text, voice, out_wav)
        print(f"  wrote {os.path.relpath(out_wav, HERE)}  dur={dur:5.2f}s")


if __name__ == "__main__":
    asyncio.run(run())
