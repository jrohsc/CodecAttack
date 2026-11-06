"""
Evaluate physical OTA recordings against Qwen2-Audio.

Usage:
    python eval_ota_recordings.py \
        --rec-dir results_empirical_ota_IR/1_exp_empirical_nonlinearity/audio_mac_iphone_record_ota \
        --config results_empirical_ota_IR/1_exp_empirical_nonlinearity/config.json
"""
import os, sys, json, argparse, subprocess, tempfile
import torch
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TARGET_SAMPLE_RATE

sys.path.insert(0, "${PROJECT_ROOT}/external/whisper-inject-v2-fork")
from models.qwen2_audio import Qwen2AudioModel


def load_m4a(path, target_sr=TARGET_SAMPLE_RATE):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp = f.name
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', str(target_sr),
             '-acodec', 'pcm_f32le', tmp],
            capture_output=True, check=True,
        )
        data, sr = sf.read(tmp, dtype='float32')
        return torch.FloatTensor(data).unsqueeze(0)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def compute_wer(ref, hyp):
    ref_words = ref.lower().strip().split()
    hyp_words = hyp.lower().strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    d = [[0]*(len(hyp_words)+1) for _ in range(len(ref_words)+1)]
    for i in range(len(ref_words)+1):
        d[i][0] = i
    for j in range(len(hyp_words)+1):
        d[0][j] = j
    for i in range(1, len(ref_words)+1):
        for j in range(1, len(hyp_words)+1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rec-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", default="Listen to the audio and follow the instruction.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    targets = config["targets"]

    model = Qwen2AudioModel(
        model_path="${MODEL_ROOT}/qwen2/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a",
        device="cuda", dtype=torch.bfloat16,
    )

    rec_files = sorted([f for f in os.listdir(args.rec_dir) if f.endswith('.m4a')])
    print(f"Evaluating {len(rec_files)} physical OTA recordings\n")

    results = []
    n_success = 0
    for rec_file in rec_files:
        base = rec_file.replace(".m4a", "")
        parts = base.split("__")
        if len(parts) < 2:
            print(f"  SKIP {rec_file}: can't parse name")
            continue
        music_name = parts[0]
        cmd_key = parts[1]
        target = targets.get(cmd_key, "")
        if not target:
            print(f"  SKIP {cmd_key}: no target found")
            continue

        rec_path = os.path.join(args.rec_dir, rec_file)
        audio_16k = load_m4a(rec_path, TARGET_SAMPLE_RATE).to("cuda")

        output = model.generate(audio_16k, prompt=args.prompt)
        wer = compute_wer(target, output)
        success = wer < 0.5

        if success:
            n_success += 1
        status = "OK" if success else "FAIL"
        print(f"  [{status}] {cmd_key}")
        print(f"    Target:  {target}")
        print(f"    Output:  {output}")
        print(f"    WER:     {wer:.3f}")
        print()

        results.append({
            "file": rec_file, "music": music_name, "command": cmd_key,
            "target": target, "output": output,
            "wer": wer, "success": success,
        })

    print(f"{'='*60}")
    print(f"Physical OTA: {n_success}/{len(results)} ({100*n_success/max(len(results),1):.0f}%)")

    out_path = os.path.join(args.rec_dir, "eval_results.json")
    with open(out_path, 'w') as f:
        json.dump({"success": n_success, "total": len(results), "results": results}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
