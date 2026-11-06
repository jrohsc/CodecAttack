"""Matched held-out evaluation for Audio Flamingo Next Instruct."""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MODEL = ROOT.parents[0] / "model_overlays" / "afnext_hf"
PROMPT = "What does the person say in this audio?"


def bundle(seed, shard):
    suffix = f"_zeroeot_seed{seed}" if seed != 1 else ""
    name = ("benchmark_ensemble3_opuseot_transcribe_eps10.0_jazz_1"
            f"_bestpostcodec_steps600{suffix}_shard{shard}of4")
    return RESULTS / name


def generate(processor, model, wav):
    conversation = [[{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "audio", "path": str(wav)},
    ]}]]
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True,
        return_dict=True).to(model.device)
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(model.dtype)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=100,
                                    do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    return processor.batch_decode(
        output_ids[:, prompt_len:], skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-targets", type=int, default=12)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--resume-from", type=Path)
    ap.add_argument("--delivery-root", type=Path)
    ap.add_argument("--channels", default="clean")
    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--control-wav", type=Path)
    args = ap.parse_args()
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = MusicFlamingoForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        local_files_only=True, attn_implementation="sdpa").eval()
    if args.control_wav:
        control = generate(processor, model, args.control_wav)
        print("control:", control, flush=True)
        if "alarm clock" not in control.lower():
            raise RuntimeError("speech control did not mention the alarm clock")
        return
    if not args.output:
        ap.error("--output is required unless --control-wav is used")

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
                start = time.monotonic()
                output = generate(processor, model, wav)
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
