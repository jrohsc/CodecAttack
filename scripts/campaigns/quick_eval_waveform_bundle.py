"""Quick parallel eval of a waveform-comparison bundle across 12 channels.

Avoids audio_agent_bench overhead; loads Qwen2-Audio once and runs each adv
wav through clean + Opus {192,128,64,32,24,16} + AAC {192,256,128} + MP3
{320,192}. Reports per-channel target_substring_match % + per-pair SNR(dB)
of adv vs clean carrier.

Usage:
    python scripts/quick_eval_waveform_bundle.py --eps 0.05
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODEC = ROOT.parent

sys.path.insert(0, '${PROJECT_ROOT}/external/whisper-inject-v2-fork')
sys.path.append(str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("config", str(CODEC / "config.py"))
_cfg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cfg)

from audio_agent_bench.channels.opus import OpusChannel  # noqa: E402
from audio_agent_bench.channels.aac import AACChannel    # noqa: E402
from audio_agent_bench.channels.mp3 import MP3Channel    # noqa: E402

# Opus channel prepends codec_attack/ to sys.path, which shadows the fork's
# `models` package with codec_attack/models/ — re-insert fork to win lookup.
_FORK = '${PROJECT_ROOT}/external/whisper-inject-v2-fork'
if _FORK in sys.path:
    sys.path.remove(_FORK)
sys.path.insert(0, _FORK)


def _load_model(model_name: str):
    if model_name == "qwen2_audio":
        from models.qwen2_audio import Qwen2AudioModel  # noqa: E402
        return Qwen2AudioModel(
            model_path=_cfg.MODEL_PATHS["qwen2_audio"],
            device="cuda", dtype=torch.bfloat16,
        )
    if model_name == "qwen25_omni":
        from models.qwen25_omni import Qwen25OmniModel  # noqa: E402
        return Qwen25OmniModel(
            model_path=_cfg.MODEL_PATHS["qwen25_omni"],
            device="cuda", dtype=torch.bfloat16,
        )
    if model_name == "audio_flamingo":
        from models.audio_flamingo import AudioFlamingoModel  # noqa: E402
        return AudioFlamingoModel(
            model_path=_cfg.MODEL_PATHS["audio_flamingo"],
            device="cuda", dtype=torch.bfloat16,
        )
    raise ValueError(f"unknown model: {model_name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eps", type=float, default=None,
                   help="S3a default-mode only: locates bundle at results/03a/waveform_comparison/<model>/eps_<eps>/")
    p.add_argument("--model", default="qwen2_audio",
                   choices=["qwen2_audio", "qwen25_omni", "audio_flamingo"])
    p.add_argument("--bundle", type=Path, default=None,
                   help="Full path to bundle dir (overrides --eps / S3a default).")
    p.add_argument("--carrier-map", type=Path, default=None,
                   help="Path to carrier_to_target.json (overrides S3a default).")
    p.add_argument("--prompt-file", type=Path, default=None,
                   help="Path to system_prompt.txt (overrides S3a default).")
    p.add_argument("--scenario", type=str, default=None,
                   help="Scenario tag for output JSON (overrides S3a default).")
    args = p.parse_args()

    if args.bundle is not None:
        BUNDLE = args.bundle
        scenario_tag = args.scenario or BUNDLE.parts[-3] if len(BUNDLE.parts) >= 3 else "unknown"
    else:
        if args.eps is None:
            p.error("--eps required when --bundle not given")
        BUNDLE = ROOT / f"results/03a_ai_detection_bypass/waveform_comparison/{args.model}/eps_{args.eps}"
        scenario_tag = args.scenario or "03a_ai_detection_bypass"
    AUDIO_DIR = BUNDLE / "audio"
    cmap_path = args.carrier_map or (ROOT / "data/music/03a_ai_detection_bypass/carrier_to_target.json")
    CMAP = json.load(open(cmap_path))
    CARRIER_ROOT = ROOT / CMAP["carrier_root"]
    prompt_path = args.prompt_file or (ROOT / "scenarios/03a_ai_detection_bypass/system_prompt.txt")
    PROMPT = prompt_path.read_text().strip()

    channels = [
        ("clean",   None),
        ("opus192", OpusChannel(192)), ("opus128", OpusChannel(128)), ("opus64", OpusChannel(64)),
        ("opus32",  OpusChannel(32)),  ("opus24",  OpusChannel(24)),  ("opus16",  OpusChannel(16)),
        ("aac256",  AACChannel(256)),  ("aac192",  AACChannel(192)),  ("aac128", AACChannel(128)),
        ("mp3_320", MP3Channel(320)),  ("mp3_192", MP3Channel(192)),
    ]

    m = _load_model(args.model)
    print(f"loaded {args.model}; free GPU mem after load: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    def _snr_db(ref: np.ndarray, test: np.ndarray) -> float:
        n = min(len(ref), len(test))
        ref = ref[:n].astype(np.float64); test = test[:n].astype(np.float64)
        noise = test - ref
        rp = float(np.mean(ref ** 2)); np_ = float(np.mean(noise ** 2))
        if np_ <= 1e-20: return 120.0
        if rp <= 1e-20: return -120.0
        return 10.0 * np.log10(rp / np_)

    results = []
    for i, pair in enumerate(CMAP["pairs"]):
        # First try S3a-style pair{NNN} suffix; fall back to carrier-stem lookup
        # (used by S1/S2 bundles where adv wavs are named *<carrier_stem>__target_*_attacked.wav).
        wavs = list(AUDIO_DIR.glob(f"*pair{i:03d}_attacked.wav"))
        if not wavs:
            stem = pair["carrier"].rsplit(".", 1)[0]
            wavs = list(AUDIO_DIR.glob(f"*{stem}__target_*_attacked.wav"))
        if not wavs:
            print(f"pair{i:03d} ({pair['carrier']}): MISSING wav"); continue
        w = wavs[0]
        target = pair["target"]

        # Load adv (clean) and carrier at adv's sample rate for SNR
        adv_np, adv_sr = sf.read(str(w))
        if adv_np.ndim > 1: adv_np = adv_np.mean(axis=1)
        car_np, car_sr = sf.read(str(CARRIER_ROOT / pair["carrier"]))
        if car_np.ndim > 1: car_np = car_np.mean(axis=1)
        if car_sr != adv_sr:
            car_np = torchaudio.functional.resample(
                torch.FloatTensor(car_np).unsqueeze(0), car_sr, adv_sr
            ).squeeze(0).numpy()
        snr_clean_db = _snr_db(car_np, adv_np)

        row = {"pair": i, "carrier": pair["carrier"], "target": target,
               "snr_clean_db": snr_clean_db, "per_channel": {}}
        for name, ch in channels:
            with tempfile.TemporaryDirectory() as td:
                inp = str(w)
                if ch is not None:
                    op = Path(td) / "out.wav"
                    ch.apply(str(w), str(op))
                    inp = str(op)
                audio, sr = sf.read(inp)
                if audio.ndim > 1: audio = audio.mean(axis=1)
                a16 = torchaudio.functional.resample(
                    torch.FloatTensor(audio).unsqueeze(0), sr, _cfg.TARGET_SAMPLE_RATE
                ).to("cuda")
                with torch.no_grad():
                    out = m.generate(a16, prompt=PROMPT)
            match = target.lower() in out.lower()
            row["per_channel"][name] = {"match": match, "out": out}
        results.append(row)
        ch_str = " ".join(f"{n}:{'Y' if row['per_channel'][n]['match'] else 'n'}" for n, _ in channels)
        print(f"pair{i:03d}  {row['carrier'][:30]:30s}  snr={snr_clean_db:5.1f}dB  {ch_str}")

    # Aggregate
    print("\n=== Per-channel success rate (n={} pairs) ===".format(len(results)))
    for name, _ in channels:
        k = sum(r["per_channel"][name]["match"] for r in results if name in r["per_channel"])
        print(f"  {name:8s}  {k}/{len(results)}  ({100*k/len(results):.1f}%)")

    snrs = [r["snr_clean_db"] for r in results]
    if snrs:
        print(f"\n=== SNR(adv vs carrier, clean) — n={len(snrs)} ===")
        print(f"  mean={np.mean(snrs):.2f} dB  median={np.median(snrs):.2f} dB  "
              f"min={np.min(snrs):.2f}  max={np.max(snrs):.2f}")

    out_path = BUNDLE / "quick_eval_per_channel.json"
    out_path.write_text(json.dumps({
        "scenario": scenario_tag,
        "target_model": args.model,
        "attack_space": "waveform",
        "eps": args.eps,
        "n_pairs": len(results),
        "channels": [n for n, _ in channels],
        "rows": results,
    }, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
