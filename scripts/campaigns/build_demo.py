#!/usr/bin/env python3
"""Build script to aggregate attack results into a static demo site.

Reads all results_transcribe/benchmark_*/ directories, extracts config + record
data, writes demo/demo_data.json, and converts audio to OGG for GitHub Pages.

Usage:
    python build_demo.py                    # JSON + symlinks (local dev)
    python build_demo.py --deploy           # JSON + convert WAV->OGG (for Pages)
    python -m http.server 8080 --directory demo/
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results_transcribe"
DEMO_DIR = Path(__file__).parent / "demo"
MUSIC_DIR = Path(__file__).parent / "data" / "music"

MODEL_DISPLAY_NAMES = {
    "qwen2_audio": "Qwen2-Audio",
    "audio_flamingo": "Audio Flamingo 3",
    "kimi_audio": "Kimi Audio 7B",
}

DIR_PATTERN = re.compile(
    r"benchmark_(qwen2_audio|audio_flamingo|kimi_audio)_eps([\d.]+)_(.*)"
)

CATEGORY_ORDER = [
    "navigation", "manipulation", "locomotion", "sensing",
    "communication", "safety", "delivery", "inspection",
    "maintenance", "interaction",
]

CATEGORY_FROM_PREFIX = {
    "nav": "navigation",
    "manip": "manipulation",
    "loco": "locomotion",
    "sense": "sensing",
    "comm": "communication",
    "safe": "safety",
    "deliv": "delivery",
    "insp": "inspection",
    "maint": "maintenance",
    "interact": "interaction",
}


def get_category(command_name: str) -> str:
    prefix = command_name.split("_")[0]
    return CATEGORY_FROM_PREFIX.get(prefix, "other")


def make_experiment_id(model: str, eps: float, music: str) -> str:
    return f"{model}_eps{eps}_{music}"


def convert_to_mp3(src: Path, dst: Path, bitrate: str = "64k"):
    """Convert an audio file to MP3 using ffmpeg + libmp3lame."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-c:a", "libmp3lame", "-b:a", bitrate,
         "-vn", str(dst)],
        capture_output=True, check=True,
    )


def process_benchmark_dir(bench_dir: Path, audio_ext: str) -> dict | None:
    match = DIR_PATTERN.match(bench_dir.name)
    if not match:
        print(f"  Skipping {bench_dir.name} (no pattern match)")
        return None

    model_key = match.group(1)
    eps = float(match.group(2))
    music = match.group(3)

    config_path = bench_dir / "config.json"
    if not config_path.exists():
        print(f"  Skipping {bench_dir.name} (no config.json)")
        return None

    with open(config_path) as f:
        config = json.load(f)

    exp_id = make_experiment_id(model_key, eps, music)
    model_subdir = bench_dir / model_key

    if not model_subdir.exists():
        print(f"  Skipping {bench_dir.name} (no model subdir {model_key})")
        return None

    samples = []
    total = 0
    direct_success = 0
    opus_success = {}

    for sample_dir in sorted(model_subdir.iterdir()):
        record_path = sample_dir / "record.json"
        if not record_path.exists():
            continue

        with open(record_path) as f:
            record = json.load(f)

        total += 1
        command_name = record["command_name"]
        category = get_category(command_name)

        if record["direct"]["success"]:
            direct_success += 1

        audio_filename = f"{record['music']}__{command_name}.{audio_ext}"
        sample = {
            "name": command_name,
            "category": category,
            "command_text": record["command_text"],
            "audio_path": f"audio/{exp_id}/{audio_filename}",
            "source_wav": str(bench_dir / "audio" / f"{record['music']}__{command_name}.wav"),
            "direct": {
                "success": record["direct"]["success"],
                "output": record["direct"]["output"],
                "snr_db": round(record["direct"].get("snr_db", 0), 1),
            },
            "original_output": record.get("original_output", ""),
        }

        opus_data = record.get("opus", {})
        for bitrate_str, data in opus_data.items():
            bitrate = int(bitrate_str)
            key = f"opus{bitrate}"
            sample[key] = {
                "success": data["success"],
                "output": data["output"],
                "snr_db": round(data.get("snr_db", 0), 1),
            }
            opus_success.setdefault(key, 0)
            if data["success"]:
                opus_success[key] += 1

        samples.append(sample)

    if total == 0:
        print(f"  Skipping {bench_dir.name} (no samples)")
        return None

    stats = {
        "total": total,
        "direct": round(100 * direct_success / total, 1),
    }
    for key in sorted(opus_success.keys()):
        stats[key] = round(100 * opus_success[key] / total, 1)

    return {
        "id": exp_id,
        "model": MODEL_DISPLAY_NAMES.get(model_key, model_key),
        "model_key": model_key,
        "eps": eps,
        "music": music,
        "stats": stats,
        "prompt_mode": config.get("prompt_mode", ""),
        "prompt_text": config.get("prompt_text", ""),
        "attack_steps": config.get("attack_steps", 0),
        "samples": samples,
    }


def create_symlinks(experiments: list[dict]):
    """Create symlinks for local development."""
    audio_dir = DEMO_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for exp in experiments:
        link_name = audio_dir / exp["id"]
        bench_name = f"benchmark_{exp['model_key']}_eps{exp['eps']}_{exp['music']}"
        target = RESULTS_DIR / bench_name / "audio"

        if link_name.is_symlink():
            link_name.unlink()
        elif link_name.is_dir():
            shutil.rmtree(link_name)

        if target.exists():
            rel_target = os.path.relpath(target, link_name.parent)
            link_name.symlink_to(rel_target)
            print(f"  Symlink: {link_name.name} -> {rel_target}")

    music_link = audio_dir / "music"
    if music_link.is_symlink():
        music_link.unlink()
    elif music_link.is_dir():
        shutil.rmtree(music_link)
    rel_music = os.path.relpath(MUSIC_DIR, music_link.parent)
    music_link.symlink_to(rel_music)
    print(f"  Symlink: music -> {rel_music}")


def convert_all_audio(experiments: list[dict]):
    """Convert all WAV files to OGG for deployment."""
    audio_dir = DEMO_DIR / "audio"

    # Remove old symlinks in audio dir
    if audio_dir.exists():
        for item in audio_dir.iterdir():
            if item.is_symlink():
                item.unlink()

    # Collect all conversion jobs
    jobs = []
    for exp in experiments:
        for sample in exp["samples"]:
            src = Path(sample["source_wav"])
            dst = audio_dir / exp["id"] / Path(sample["audio_path"]).name
            if src.exists():
                jobs.append((src, dst))

    # Convert music files
    music_out = audio_dir / "music"
    music_out.mkdir(parents=True, exist_ok=True)
    if MUSIC_DIR.exists():
        for f in MUSIC_DIR.iterdir():
            if f.suffix in (".mp3", ".wav"):
                dst = music_out / (f.stem + ".mp3")
                jobs.append((f, dst))

    print(f"  Converting {len(jobs)} audio files to MP3 (parallel)...")

    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for src, dst in jobs:
            if dst.exists():
                done += 1
                continue
            bitrate = "96k" if "music" in str(dst.parent) else "64k"
            futures[pool.submit(convert_to_mp3, src, dst, bitrate)] = (src, dst)

        total_to_convert = len(futures)
        converted = 0
        for future in as_completed(futures):
            converted += 1
            src, dst = futures[future]
            try:
                future.result()
                done += 1
            except Exception as e:
                failed += 1
                print(f"    FAILED: {src.name}: {e}")
            if converted % 200 == 0 or converted == total_to_convert:
                print(f"    Progress: {converted}/{total_to_convert}")

    print(f"  Done: {done} converted, {failed} failed")


def build_category_list(experiments: list[dict]) -> list[str]:
    cats = set()
    for exp in experiments:
        for s in exp["samples"]:
            cats.add(s["category"])
    return [c for c in CATEGORY_ORDER if c in cats]


def build_command_list(experiments: list[dict]) -> list[dict]:
    seen = {}
    for exp in experiments:
        for s in exp["samples"]:
            if s["name"] not in seen:
                seen[s["name"]] = {
                    "name": s["name"],
                    "category": s["category"],
                    "command_text": s["command_text"],
                }
    return sorted(seen.values(), key=lambda x: (x["category"], x["name"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true",
                        help="Convert WAV->OGG for GitHub Pages deployment")
    args = parser.parse_args()

    audio_ext = "mp3" if args.deploy else "wav"

    print(f"Building demo data (mode={'deploy' if args.deploy else 'local'})...")
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    experiments = []
    for bench_dir in sorted(RESULTS_DIR.iterdir()):
        if not bench_dir.is_dir():
            continue
        print(f"Processing {bench_dir.name}...")
        exp = process_benchmark_dir(bench_dir, audio_ext)
        if exp:
            experiments.append(exp)

    print(f"\nFound {len(experiments)} experiments")
    experiments.sort(key=lambda e: (e["model"], e["eps"], e["music"]))

    # Build music list
    music_ext = "mp3" if args.deploy else None
    music_files = []
    if MUSIC_DIR.exists():
        for f in sorted(MUSIC_DIR.iterdir()):
            if f.suffix in (".mp3", ".wav"):
                out_name = f"{f.stem}.{music_ext}" if music_ext else f.name
                music_files.append({
                    "name": f.stem,
                    "path": f"audio/music/{out_name}",
                })

    # Strip source_wav from samples before writing JSON
    clean_experiments = []
    for exp in experiments:
        clean_exp = {k: v for k, v in exp.items() if k != "samples"}
        clean_exp["samples"] = [
            {k: v for k, v in s.items() if k != "source_wav"}
            for s in exp["samples"]
        ]
        clean_experiments.append(clean_exp)

    demo_data = {
        "experiments": clean_experiments,
        "categories": build_category_list(experiments),
        "commands": build_command_list(experiments),
        "music_files": music_files,
    }

    out_path = DEMO_DIR / "demo_data.json"
    with open(out_path, "w") as f:
        json.dump(demo_data, f, indent=2)
    print(f"Wrote {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")

    if args.deploy:
        print("\nConverting audio to OGG...")
        convert_all_audio(experiments)
        print("\nDeploy-ready! Contents of demo/ can be pushed to GitHub Pages.")
    else:
        print("\nCreating audio symlinks...")
        create_symlinks(experiments)
        print("\nDone! To preview:")
        print("  python -m http.server 8080 --directory demo/")

    total_size = sum(
        f.stat().st_size for f in DEMO_DIR.rglob("*") if f.is_file()
    ) / (1024 * 1024)
    print(f"\nTotal demo/ size: {total_size:.0f} MB")


if __name__ == "__main__":
    main()
