#!/usr/bin/env python
"""Local GPU helper for the e3phi_rep seed-7/42 replicate shards.

Crafts clips from the TAIL of each shard's 56-command list into the same bundle
the SLURM task is filling from the head. --seed-codec-per-clip makes each clip's
trajectory independent of order, so a locally crafted clip is the same clip.

The SLURM task loaded its `completed` set at startup and would redo the tail, so
a watcher requeues it once all 56 records exist: the requeued run reloads
`completed`, skips everything, passes the 56-wav check and exits 0, which keeps
the aftercorr dependency of e3phi1eval intact (scancel would break it).

Run detached:  setsid nohup python local_tail_helper.py > logs/e3phi_local.out 2>&1 &
"""
import json, os, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path("${REPO_ROOT}")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from run_benchmark import filter_commands  # noqa: E402

ARRAYS = {"7": "64881331", "42": "64881334"}
BLOCK = 4          # clips per local invocation (amortizes the 3-model load)
MARGIN = 3         # never claim within this many clips of the SLURM front
CLAIMS = ROOT / "logs" / "e3phi_local_claims.json"

ALL = list(filter_commands(None, mode="transcribe", dataset="ours"))
assert len(ALL) == 224


def bundle(seed, sh):
    return ROOT / ("results/benchmark_ensemble_omni7_af3_phi4_opuseot_transcribe_eps10.0"
                   f"_jazz_1_steps600_seed{seed}_shard{sh}of4")


def shard_cmds(sh):
    return ALL[sh::4]


def done(seed, sh):
    d = bundle(seed, sh) / "qwen25_omni"
    return {c for c in shard_cmds(sh) if (d / f"jazz_1__{c}" / "record.json").is_file()}


def slurm_state(seed, sh):
    r = subprocess.run(["squeue", "-h", "-j", f"{ARRAYS[seed]}_{sh}", "-o", "%T"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


claims = json.loads(CLAIMS.read_text()) if CLAIMS.exists() else {}


def watcher():
    pending = {}
    while True:
        for key in [k for k, v in claims.items() if v]:
            seed, sh = key.split(":"); sh = int(sh)
            if len(done(seed, sh)) == 56 and slurm_state(seed, sh) == "RUNNING":
                pending[key] = pending.get(key, 0) + 1
                if pending[key] >= 2:  # two polls apart: not SLURM's own clean exit
                    log(f"requeue {ARRAYS[seed]}_{sh} (all 56 present, SLURM redoing tail)")
                    subprocess.run(["scontrol", "requeue", f"{ARRAYS[seed]}_{sh}"])
                    pending[key] = -10  # back off while it cycles
            else:
                pending[key] = 0
        time.sleep(60)


threading.Thread(target=watcher, daemon=True).start()

while True:
    # Pick the shard furthest behind (records + outstanding local claims).
    best = None
    for seed in ARRAYS:
        for sh in range(4):
            cmds, d = shard_cmds(sh), done(seed, sh)
            front = max((i for i, c in enumerate(cmds) if c in d
                         and c not in claims.get(f"{seed}:{sh}", [])), default=-1)
            free = [c for i, c in enumerate(cmds)
                    if i > front + MARGIN and c not in d]
            if len(free) < BLOCK:
                continue
            if best is None or len(d) < best[0]:
                best = (len(d), seed, sh, free[-BLOCK:])
    if best is None:
        log("no shard has a safe tail block left; waiting for watcher")
        while any(len(done(k.split(":")[0], int(k.split(":")[1]))) < 56 for k in claims):
            time.sleep(300)
        log("all claimed shards complete; exiting")
        break
    _, seed, sh, block = best
    key = f"{seed}:{sh}"
    claims.setdefault(key, []).extend(block)
    CLAIMS.write_text(json.dumps(claims, indent=1))
    log(f"seed{seed} shard{sh}: crafting {block}")
    env = dict(os.environ, ENSEMBLE_PERSISTENT_WORKER="1", ENSEMBLE_WORKER_GPU="0",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    rc = subprocess.run(
        [sys.executable, "-u", "run_benchmark.py", "--ensemble",
         "--ensemble-models", "qwen25_omni,audio_flamingo,phi4",
         "--target-model", "qwen25_omni",
         "--ensemble-channel-mode", "multi_bitrate", "--ensemble-best-after-warmup",
         "--ensemble-strict-workers",
         "--steps", "600", "--eps", "10.0", "--alpha", "0.2", "--perceptual-weight", "0.1",
         "--prompt-mode", "transcribe", "--seed", seed, "--seed-codec-per-clip",
         "--music", "jazz_1", "--output-dir", str(bundle(seed, sh)),
         "--commands", *block], env=env).returncode
    missing = [c for c in block if c not in done(seed, sh)]
    log(f"seed{seed} shard{sh}: rc={rc}, missing after block: {missing}")
    if missing:  # release unfinished claims so the SLURM task still owns them
        claims[key] = [c for c in claims[key] if c not in missing]
        CLAIMS.write_text(json.dumps(claims, indent=1))
        if rc != 0:
            log("local run failed; stopping helper")
            break
