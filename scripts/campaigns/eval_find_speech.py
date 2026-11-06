"""Evaluate ALL m4a recordings at 1.0x to find which ones detect speech."""
import os, sys, subprocess, tempfile, glob
import torch
import soundfile as sf

sys.path.insert(0, "${REPO_ROOT}")
sys.path.insert(0, "${PROJECT_ROOT}/external/whisper-inject-v2-fork")
from config import TARGET_SAMPLE_RATE
from models.qwen2_audio import Qwen2AudioModel

PROMPT_QA = "Listen to the audio and follow the instruction."
BASE = "${REPO_ROOT}/results_empirical_analysis_noise_cancellation/27_exp_narrow_bandpass_marshall"


def load_m4a(path, target_sr=TARGET_SAMPLE_RATE):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp = f.name
    try:
        subprocess.run(['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', str(target_sr),
                        '-acodec', 'pcm_f32le', tmp], capture_output=True, check=True)
        data, sr = sf.read(tmp, dtype='float32')
        return torch.FloatTensor(data).unsqueeze(0)
    finally:
        os.unlink(tmp)


def main():
    # Find all m4a files
    m4a_files = sorted(glob.glob(os.path.join(BASE, "**/*.m4a"), recursive=True))
    print(f"Found {len(m4a_files)} m4a files\n")

    print("Loading Qwen2-Audio...")
    model = Qwen2AudioModel(device="cuda")
    print("Model loaded.\n")

    results = []
    for path in m4a_files:
        rel_path = os.path.relpath(path, BASE)
        print(f"{'='*80}")
        print(f"File: {rel_path}")

        try:
            audio = load_m4a(path)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            results.append({"file": rel_path, "error": str(e)})
            continue

        rms = audio.pow(2).mean().sqrt().item()
        duration = audio.shape[-1] / TARGET_SAMPLE_RATE
        print(f"  Duration: {duration:.2f}s, RMS: {rms:.6f}")

        with torch.no_grad():
            output = model.generate(audio.to("cuda"), prompt=PROMPT_QA)

        # Check for speech keywords
        out_lower = output.lower()
        speech_kw = ["speaking", "saying", "says", "talking", "person", "woman", "man",
                     "voice", "whisper", "instruct", "order", "command"]
        detected_kw = [kw for kw in speech_kw if kw in out_lower]

        print(f"  Output: {output}")
        if detected_kw:
            print(f"  >>> SPEECH DETECTED: {detected_kw}")

        results.append({
            "file": rel_path,
            "duration_s": round(duration, 2),
            "rms": round(rms, 6),
            "output": output,
            "speech_keywords": detected_kw,
            "speech_detected": len(detected_kw) > 0,
        })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    speech_count = 0
    for r in results:
        if "error" in r:
            print(f"  [ERROR] {r['file']}")
            continue
        status = "SPEECH" if r["speech_detected"] else "      "
        print(f"  [{status}] {r['file']:<70s} → {r['output'][:80]}")
        if r["speech_detected"]:
            speech_count += 1

    print(f"\nSpeech detected in {speech_count}/{len(results)} recordings")

    # Save
    import json
    out_path = os.path.join(BASE, "speech_detection_analysis", "all_recordings_eval.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
