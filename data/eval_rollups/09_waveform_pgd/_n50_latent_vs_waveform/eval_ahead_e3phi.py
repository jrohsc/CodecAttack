#!/usr/bin/env python
"""Evaluate e3phi seed 7/42 clips WHILE they are still being crafted.

The SLURM eval chain (sbatch_ensemble_omni_af3_phi_eval.sh) runs 6 victims x 56
clips per shard (~37 min) only after a shard's crafting ends. ensemble_heldout_eval.py
caches per clip (<bundle>/<OUT>/<stem>/record.json, reused iff it has clean + every
requested channel), so records written here in advance turn that chain into a
few-minute top-up.

Per round, per seed, per victim: clips that are READY (wav + primary record present,
in the main bundle or the _STEAL bundle) and lack a cached record for that victim
are linked into a staging bundle and evaluated there with the exact SLURM
arguments. New records are copied into <main shard bundle>/<OUT>/<stem>/record.json
with "wav" rewritten to the main-bundle path. summary.json is never written into a
main bundle; SLURM's eval writes it from the cache. Shards whose SLURM eval task has
started are left alone.
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path("${REPO_ROOT}")
ENVROOT = "${WORK_ROOT}/miniconda3/envs"
STATE = REPO / ".e3phi_steal"
STAGE = STATE / "eval_stage"
LOGDIR = REPO / "logs" / "e3phi_steal"
MUSIC = "jazz_1"
CH = "opus64,opus128,mp3_64,mp3_128,aac_64,aac_128"
REQ = ["clean"] + CH.split(",")
TRACKS = {7: (64881331, 64881332), 42: (64881334, 64881335)}
VICTIMS = [("qwen25_omni", "omni7b_MEMBER"), ("audio_flamingo", "af3_MEMBER"),
           ("phi4", "phi4_MEMBER"), ("qwen2_audio", "qwen2_HELDOUT"),
           ("kimi_audio", "kimi_HELDOUT_100tok"), ("qwen25_omni_3b", "omni3b_HELDOUT")]
PYFOR = {"qwen2_audio": "codec-attack", "qwen25_omni": "qwen-omni", "qwen25_omni_3b": "qwen-omni",
         "audio_flamingo": "flamingo3", "kimi_audio": "kimi-audio", "phi4": "audio-llm"}
MIN_BATCH = int(os.environ.get("MIN_BATCH", "8"))
GPU = os.environ.get("GPU", "0")
MIN_FREE_MIB = 26000

os.chdir(REPO)
sys.path.insert(0, str(REPO))
from run_benchmark import filter_commands  # noqa: E402
ALL = list(filter_commands(None, mode="transcribe", dataset="ours"))
assert len(ALL) == 224
SHARD_OF = {c: i % 4 for i, c in enumerate(ALL)}


def log(msg):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(LOGDIR / "eval_ahead.log", "a") as f:
        f.write(line + "\n")


def bundle(seed, sh):
    return REPO / "results" / (
        "benchmark_ensemble_omni7_af3_phi4_opuseot_transcribe_eps10.0_jazz_1_steps600"
        f"_seed{seed}_shard{sh}of4")


def sq_state(j):
    r = subprocess.run(["squeue", "-j", j, "-h", "-o", "%T"], capture_output=True, text=True)
    return r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""


def slurm_eval_started(seed, sh):
    st = sq_state(f"{TRACKS[seed][1]}_{sh}")
    if st in ("RUNNING", "COMPLETING"):
        return True
    # gone from the queue after its shard was merged -> it ran (or died); hands off
    return st == "" and (STATE / f"merged_{seed}_{sh}").exists()


def cached_ok(p):
    try:
        r = json.loads(p.read_text())
        return all(c in r for c in REQ)
    except Exception:
        return False


def ready_clips(seed):
    """cmd -> (wav path, primary record dir) for clips whose craft finished."""
    out = {}
    for sh in range(4):
        if slurm_eval_started(seed, sh):
            continue
        for b in (bundle(seed, sh), Path(str(bundle(seed, sh)) + "_STEAL")):
            for rec in (b / "qwen25_omni").glob(f"{MUSIC}__*/record.json"):
                stem = rec.parent.name
                wav = b / "audio" / f"{stem}.wav"
                cmd = stem[len(MUSIC) + 2:]
                if wav.is_file() and cmd in SHARD_OF and cmd not in out:
                    out[cmd] = (wav, rec.parent)
    return out


def gpu_free_mib():
    r = subprocess.run(["nvidia-smi", "-i", GPU, "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def run_batch(seed, model, out, need, ready, final):
    st = STAGE / f"seed{seed}_{out}"
    shutil.rmtree(st, ignore_errors=True)
    (st / "audio").mkdir(parents=True)
    (st / "qwen25_omni").mkdir()
    os.symlink(bundle(seed, 0) / "config.json", st / "config.json")
    for cmd in need:
        wav, recdir = ready[cmd]
        os.symlink(wav, st / "audio" / wav.name)
        os.symlink(recdir, st / "qwen25_omni" / recdir.name)
    while gpu_free_mib() < MIN_FREE_MIB:
        log(f"  waiting for GPU memory ({gpu_free_mib()} MiB free)"); time.sleep(60)
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUTF8="1", PYTHONNOUSERSITE="1",
               CUDA_VISIBLE_DEVICES=GPU,
               PATH=f"{ENVROOT}/codec-attack/bin:" + os.environ.get("PATH", ""),
               KIMI_INFER_ROOT="${WORK_ROOT}/audio_benign_finetuning/Kimi-Audio")
    env["PYTHONPATH"] = env["KIMI_INFER_ROOT"] + ":" + os.environ.get("PYTHONPATH", "")
    if model in ("qwen25_omni", "qwen25_omni_3b"):
        env.pop("PYTHONNOUSERSITE")
    cmdline = [f"{ENVROOT}/{PYFOR[model]}/bin/python", "-u",
               "0_all_combined/scripts/ensemble_heldout_eval.py",
               "--bundle-dir", str(st), "--target-model", model, "--out-name", out,
               "--channels", CH] + (["--max-new-tokens", "100"] if model == "kimi_audio" else [])
    t0 = time.time()
    with open(LOGDIR / f"evalahead_seed{seed}_{out}.log", "a") as f:
        f.write(f"\n=== {time.ctime()} {len(need)} clips ===\n"); f.flush()
        rc = subprocess.run(cmdline, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    copied = 0
    for cmd in need:
        stem = f"{MUSIC}__{cmd}"
        src = st / out / stem / "record.json"
        if not cached_ok(src):
            continue
        dst = bundle(seed, SHARD_OF[cmd]) / out / stem / "record.json"
        if cached_ok(dst):
            continue
        r = json.loads(src.read_text())
        r["wav"] = str(bundle(seed, SHARD_OF[cmd]) / "audio" / f"{stem}.wav")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(r, indent=2)); os.replace(tmp, dst)
        copied += 1
    log(f"  seed{seed} {out}: rc={rc} {copied}/{len(need)} records cached in "
        f"{(time.time()-t0)/60:.1f} min")
    return rc


def main():
    STATE.mkdir(exist_ok=True)
    lock = STATE / "eval_ahead.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        sys.exit(f"{lock} exists: another eval-ahead is running")
    log(f"=== eval-ahead start (gpu {GPU}, MIN_BATCH {MIN_BATCH}) ===")
    fails = 0
    try:
        while not (STATE / "STOP_EVAL").exists():
            open_shards = [(s, h) for s in TRACKS for h in range(4) if not slurm_eval_started(s, h)]
            if not open_shards:
                log("=== every SLURM eval task has started; eval-ahead done ==="); return
            crafting_over = all((STATE / f"merged_{s}_{h}").exists() for s, h in open_shards)
            did = 0
            for seed in TRACKS:
                ready = ready_clips(seed)
                for model, out in VICTIMS:
                    need = sorted(c for c in ready if not cached_ok(
                        bundle(seed, SHARD_OF[c]) / out / f"{MUSIC}__{c}" / "record.json"))
                    if not need or (len(need) < MIN_BATCH and not crafting_over):
                        continue
                    log(f"seed{seed} {out}: {len(need)} clips ({len(ready)} ready)")
                    rc = run_batch(seed, model, out, need, ready, crafting_over)
                    did += 1
                    fails = fails + 1 if rc else 0
                    if fails >= 3:
                        log("!! 3 consecutive eval failures; exiting"); return
            if not did:
                time.sleep(300)
    finally:
        shutil.rmtree(lock, ignore_errors=True)


if __name__ == "__main__":
    main()
