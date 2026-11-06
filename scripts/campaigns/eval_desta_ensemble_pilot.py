"""Small, reproducible DeSTA held-out probe on common ensemble targets.

Run with the audio-llm environment and PYTHONPATH pointing at the cached
DeSTA2.5-Audio repository. The base model, Llama, and Whisper weights are
read from existing local caches; only tokenizer metadata is in the overlay.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from desta.models.modeling_desta25 import DeSTA25AudioModel, DeSTA25Config


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OVERLAY = ROOT.parents[0] / "model_overlays"
BASE = Path("${MODEL_ROOT}/desta/hub/models--DeSTA-ntu--DeSTA2.5-Audio-Llama-3.1-8B/snapshots")
LLAMA = OVERLAY / "desta_llama8b"
TEMPLATE = OVERLAY / "desta_hf_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots"
PROMPT = "What does the person say in this audio?"


def bundle(seed, shard):
    suffix = (f"_zeroeot_seed{seed}" if seed != 1 else "")
    name = ("benchmark_ensemble3_opuseot_transcribe_eps10.0_jazz_1"
            f"_bestpostcodec_steps600{suffix}_shard{shard}of4")
    return RESULTS / name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-targets", type=int, default=12)
    ap.add_argument("--full224", action="store_true",
                    help="use all four shards instead of the original shared 112")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--resume-from", type=Path,
                    help="Reuse completed (seed, clip) records from a previous probe")
    ap.add_argument("--delivery-root", type=Path)
    ap.add_argument("--channels", default="clean")
    ap.add_argument("--seeds", default="1,7,42")
    args = ap.parse_args()
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    files = {}
    for seed in (1, 7, 42):
        files[seed] = {}
        for shard in range(4 if args.full224 else 2):
            b = bundle(seed, shard)
            files[seed].update({p.name: (p, b) for p in (b / "audio").glob("*.wav")})
    common = sorted(set.intersection(*(set(d) for d in files.values())))
    expected = 224 if args.full224 else 112
    if len(common) != expected:
        raise RuntimeError(f"expected {expected} common targets, found {len(common)}")
    if not (1 <= args.n_targets <= len(common)):
        raise ValueError(f"n-targets must be between 1 and {expected}")
    # Spread the pilot across the common target set, not just its first category.
    indices = [(i * len(common)) // args.n_targets for i in range(args.n_targets)]
    selected = [common[i] for i in indices]

    base = next(BASE.iterdir())
    # DeSTA25Config loads the Llama config during construction. Point it at the
    # local overlay before that happens. Keep the Whisper model ID because the
    # connector selects its layers by that exact ID; its weights are cached.
    config_data = json.loads((base / "config.json").read_text())
    config_data["llm_model_id"] = str(LLAMA)
    cfg = DeSTA25Config(**config_data)
    model = DeSTA25AudioModel(cfg)
    model.load_state_dict(load_file(str(base / "model.safetensors")), strict=False)
    model.to("cuda").eval()
    model._setup_generation()
    chat_config = next(TEMPLATE.glob("*/tokenizer_config.json"))
    model.tokenizer.chat_template = json.loads(chat_config.read_text())["chat_template"]

    rows = json.loads(args.resume_from.read_text()) if args.resume_from else []
    if any(r.get("prompt_text") != PROMPT or r.get("system_prompt") != "" for r in rows):
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
                messages = [{
                    "role": "user", "content": f"<|AUDIO|>\n{PROMPT}",
                    "audios": [{"audio": str(wav), "text": None}],
                }]
                start = time.monotonic()
                with torch.no_grad():
                    output = model.generate(messages=messages, do_sample=False,
                                            max_new_tokens=100).text[0]
                row = {"seed": seed, "clip": source.stem, "channel": channel,
                       "target": target, "prompt_text": PROMPT,
                       "system_prompt": "", "output": output,
                       "exact": target.lower() in output.lower(),
                       "seconds": round(time.monotonic() - start, 2)}
                rows.append(row)
                args.output.write_text(json.dumps(rows, indent=2) + "\n")
                print(f"seed={seed} {source.stem} {channel} exact={row['exact']} "
                      f"time={row['seconds']}s output={output[:90]!r}", flush=True)
    for seed in seeds:
        subset = [r for r in rows if r["seed"] == seed]
        hits = sum(r["exact"] for r in subset)
        print(f"seed={seed}: {hits}/{len(subset)} ({100*hits/len(subset):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
