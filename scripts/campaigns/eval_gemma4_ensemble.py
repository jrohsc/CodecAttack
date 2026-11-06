"""Held-out Gemma 4 E4B-it clean-delivery evaluation on ensemble WAVs.

Run in the isolated Gemma 4 virtual environment. Uses the locally cached
instruction-tuned checkpoint; writes one record after each evaluated clip.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Gemma4ForConditionalGeneration


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MODEL_CACHE = Path("${MODEL_ROOT}/gemma/hub/models--google--gemma-4-E4B-it/snapshots")
PROMPT = "What does the person say in this audio?"


def bundle(seed, shard):
    suffix = f"_zeroeot_seed{seed}" if seed != 1 else ""
    name = ("benchmark_ensemble3_opuseot_transcribe_eps10.0_jazz_1"
            f"_bestpostcodec_steps600{suffix}_shard{shard}of4")
    return RESULTS / name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-targets", type=int, default=12)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--resume-from", type=Path)
    ap.add_argument("--delivery-root", type=Path)
    ap.add_argument("--channels", default="clean")
    ap.add_argument("--seeds", default="1,7,42")
    args = ap.parse_args()
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    files = {}
    for seed in (1, 7, 42):
        files[seed] = {}
        for shard in (0, 1):
            b = bundle(seed, shard)
            files[seed].update({p.name: (p, b) for p in (b / "audio").glob("*.wav")})
    common = sorted(set.intersection(*(set(d) for d in files.values())))
    if len(common) != 112:
        raise RuntimeError(f"expected 112 common targets, found {len(common)}")
    if not (1 <= args.n_targets <= len(common)):
        raise ValueError("n-targets must be between 1 and 112")
    indices = [(i * len(common)) // args.n_targets for i in range(args.n_targets)]
    selected = [common[i] for i in indices]

    checkpoint = next(MODEL_CACHE.iterdir())
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map="cuda",
        local_files_only=True, attn_implementation="sdpa").eval()

    rows = json.loads(args.resume_from.read_text()) if args.resume_from else []
    if any(r.get("prompt_text") != PROMPT for r in rows):
        raise ValueError("resume records use a different prompt")
    completed = {(r["seed"], r["clip"], r.get("channel", "clean")) for r in rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for filename in selected:
            source, b = files[seed][filename]
            record = b / "qwen2_audio" / source.stem / "record.json"
            target = json.loads(record.read_text())["target_text"]
            for channel in channels:
                if (seed, source.stem, channel) in completed:
                    continue
                wav = (args.delivery_root / str(seed) / source.stem / f"{channel}.wav"
                       if args.delivery_root else source)
                if not wav.is_file():
                    raise FileNotFoundError(wav)
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "audio", "audio": str(wav)},
                ]}]
                start = time.monotonic()
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, return_dict=True, return_tensors="pt",
                    add_generation_prompt=True, enable_thinking=False).to("cuda")
                with torch.inference_mode():
                    output_ids = model.generate(**inputs, max_new_tokens=100,
                                                do_sample=False)
                output = processor.decode(
                    output_ids[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True).strip()
                row = {"seed": seed, "clip": source.stem, "channel": channel,
                       "target": target, "prompt_text": PROMPT,
                       "output": output, "exact": target.lower() in output.lower(),
                       "seconds": round(time.monotonic() - start, 2)}
                rows.append(row)
                args.output.write_text(json.dumps(rows, indent=2) + "\n")
                print(f"seed={seed} {source.stem} {channel} exact={row['exact']} "
                      f"time={row['seconds']}s output={output[:90]!r}", flush=True)
    for seed in seeds:
        subset = [r for r in rows if r["seed"] == seed]
        hits = sum(r["exact"] for r in subset)
        print(f"seed={seed}: {hits}/{len(subset)} ({100*hits/len(subset):.1f}%)",
              flush=True)


if __name__ == "__main__":
    main()
