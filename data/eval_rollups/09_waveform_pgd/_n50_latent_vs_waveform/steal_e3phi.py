#!/usr/bin/env python
"""Work-stealing for the e3phi seed 7/42 crafting shards (Omni7B+AF3+Phi4, eps10).

SLURM shards craft their 56 clips front-to-back. Extra GPUs (outside the 8-GPU
account cap) claim clips from the BACK of the slowest shard, one clip per
run_benchmark call (run_benchmark ignores --commands order, so a multi-clip call
cannot run backward). Claimed clips always form a suffix of the shard.
Per-clip Opus seeding (--seed-codec-per-clip hashes seed:music:cmd) makes a clip's
trajectory independent of which process crafts it.

  worker     craft stolen clips on one GPU (run as many as you have GPUs; any
             host that mounts /work)
  supervisor (exactly one, no GPU) when a SLURM craft task starts a clip that a
             helper already published, `scontrol requeue` it: the rerun reloads
             `completed`, skips every published clip, and (once all 56 exist) passes
             the 56-wav check and exits 0, so the eval's aftercorr dependency holds.

v2 (coexists with local_tail_helper.py, which crafts 4-clip tail blocks straight
into the main bundle and tracks claims in logs/e3phi_local_claims.json): each
finished clip is published into the MAIN bundle at once (wav, then record.json).
A published tail clip moves that helper's inferred SLURM front to the shard end,
so it stops stealing there; this worker skips every shard in its claims file.
  status     one-line-per-shard table
"""
import argparse, os, re, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path("${REPO_ROOT}")
PY = "${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python"
STATE = REPO / ".e3phi_steal"
LOGDIR = REPO / "logs" / "e3phi_steal"
MUSIC = "jazz_1"
# seed -> (craft array, eval array)
TRACKS = {7: (64881331, 64881332), 42: (64881334, 64881335)}
STALE_S = 25 * 60   # an active claim older than this is abandoned (a clip takes ~9 min)
MIN_AHEAD = 3   # only steal clip i if i >= slurm_current + MIN_AHEAD

os.chdir(REPO)
sys.path.insert(0, str(REPO))


def all_commands():
    from run_benchmark import filter_commands
    cmds = list(filter_commands(None, mode="transcribe", dataset="ours"))
    assert len(cmds) == 224, len(cmds)
    return cmds


ALL = all_commands()


def bundle(seed, sh):
    return REPO / "results" / (
        "benchmark_ensemble_omni7_af3_phi4_opuseot_transcribe_eps10.0_jazz_1_steps600"
        f"_seed{seed}_shard{sh}of4")


def steal_dir(seed, sh):
    return Path(str(bundle(seed, sh)) + "_STEAL")


def shard_cmds(sh):
    return ALL[sh::4]


def wav_cmds(d):
    a = d / "audio"
    if not a.is_dir():
        return set()
    return {p.stem[len(MUSIC) + 2:] for p in a.glob("*.wav")}


def log(msg, name):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(LOGDIR / f"{name}.log", "a") as f:
        f.write(line + "\n")


def sq_state(jobid):
    r = subprocess.run(["squeue", "-j", jobid, "-h", "-o", "%T"],
                       capture_output=True, text=True)
    return r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""


def slurm_current(seed, sh):
    craft, _ = TRACKS[seed]
    f = REPO / "logs" / f"e3phi_rep_{craft}_{sh}.out"
    cur = None
    try:
        for m in re.finditer(r"^\[\d+/\d+\] music=\S+, cmd=(\S+)", f.read_text(), re.M):
            cur = m.group(1)
    except FileNotFoundError:
        pass
    return cur


HELPER_CLAIMS = REPO / "logs" / "e3phi_local_claims.json"


def helper_claims():
    """{(seed, sh): set(cmds)} claimed by local_tail_helper.py"""
    import json
    try:
        raw = json.loads(HELPER_CLAIMS.read_text())
    except Exception:
        return {}
    return {(int(k.split(":")[0]), int(k.split(":")[1])): set(v) for k, v in raw.items()}


def publish(seed, sh, cmd, name):
    """copy a finished _STEAL clip into the main bundle: wav first, record last."""
    stem = f"{MUSIC}__{cmd}"
    B, T = bundle(seed, sh), steal_dir(seed, sh)
    rec_dst = B / "qwen25_omni" / stem / "record.json"
    if rec_dst.exists() or (B / "audio" / f"{stem}.wav").exists():
        log(f"  publish skip seed{seed}/sh{sh} {cmd}: main already has it", name); return
    tmp = B / "audio" / f".{stem}.wav.tmp"
    shutil.copy2(T / "audio" / f"{stem}.wav", tmp); os.replace(tmp, B / "audio" / f"{stem}.wav")
    rec_dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = rec_dst.parent / ".record.json.tmp"
    shutil.copy2(T / "qwen25_omni" / stem / "record.json", tmp); os.replace(tmp, rec_dst)
    log(f"  published seed{seed}/sh{sh} {cmd} -> main bundle", name)


def claims(seed, sh):
    """cmd -> 'done' | 'active'"""
    out = {}
    for d in STATE.glob(f"claim_{seed}_{sh}__*"):
        cmd = d.name.split("__", 1)[1]
        if (d / "done").exists():
            out[cmd] = "done"
        elif time.time() - d.stat().st_mtime > STALE_S:
            # owner died mid-clip (e.g. interactive allocation expired): free the clip
            shutil.rmtree(d, ignore_errors=True)
        else:
            out[cmd] = "active"
    return out


def shard_view(seed, sh):
    craft, ev = TRACKS[seed]
    cmds = shard_cmds(sh)
    M, T, C = wav_cmds(bundle(seed, sh)), wav_cmds(steal_dir(seed, sh)), claims(seed, sh)
    H = helper_claims().get((seed, sh), set())
    st = sq_state(f"{craft}_{sh}")
    cur = slurm_current(seed, sh) if st == "RUNNING" else None
    uncovered = [i for i, c in enumerate(cmds) if c not in M and c not in T and c not in C and c not in H]
    if st == "RUNNING" and cur in cmds:
        s_idx = cmds.index(cur)
    else:     # pending (requeued, may wait for a node) or gone: everything is stealable
        s_idx = -10**6
    stealable = [i for i in uncovered if i >= s_idx + MIN_AHEAD]
    return dict(seed=seed, sh=sh, cmds=cmds, M=M, T=T, C=C, H=H, st=st, cur=cur,
                s_idx=s_idx, uncovered=uncovered, stealable=stealable,
                done=(STATE / f"merged_{seed}_{sh}").exists())


# ---------------------------------------------------------------- worker
def worker(a):
    name = f"worker_{os.uname().nodename}_gpu{a.gpu}_{os.environ.get('SLURM_JOB_ID', 'local')}"
    STATE.mkdir(exist_ok=True)
    fails = 0
    while True:
        if (STATE / "STOP").exists():
            log("STOP file present; exiting", name); return
        views = [shard_view(s, h) for s in TRACKS for h in range(4)]
        cand = [v for v in views if v["stealable"] and not v["H"]]
        mine = [v for v in cand if v["C"]]
        fresh = [v for v in cand if not v["C"]]
        if len(fresh) > 1:   # helper picks the min-published shard next; leave it that one
            fresh.remove(min(fresh, key=lambda v: len(v["M"])))
        cand = mine + fresh
        if not cand:
            if all(not v["uncovered"] for v in views):
                log("nothing left to steal; exiting", name); return
            time.sleep(60); continue
        v = max(cand, key=lambda v: (len(v["uncovered"]), len(v["stealable"])))
        i = max(v["stealable"]); cmd = v["cmds"][i]
        cl = STATE / f"claim_{v['seed']}_{v['sh']}__{cmd}"
        try:
            cl.mkdir()
        except FileExistsError:
            continue
        (cl / "owner").write_text(f"{name} pid={os.getpid()}\n")
        seed, sh = v["seed"], v["sh"]
        log(f"claim seed{seed}/sh{sh} idx={i} cmd={cmd} (slurm at idx {v['s_idx']}, "
            f"{len(v['uncovered'])} uncovered)", name)
        env = dict(os.environ, ENSEMBLE_PERSISTENT_WORKER="1",
                   ENSEMBLE_WORKER_GPU="0", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        if a.gpu != "inherit":   # under SLURM keep the allocation's CUDA_VISIBLE_DEVICES
            env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
        clog = LOGDIR / f"craft_seed{seed}_sh{sh}_{cmd}.log"
        t0 = time.time()
        with open(clog, "a") as f:
            rc = subprocess.run(
                [PY, "-u", "run_benchmark.py", "--ensemble",
                 "--ensemble-models", "qwen25_omni,audio_flamingo,phi4",
                 "--target-model", "qwen25_omni",
                 "--ensemble-channel-mode", "multi_bitrate", "--ensemble-best-after-warmup",
                 "--ensemble-strict-workers",
                 "--steps", "600", "--eps", "10.0", "--alpha", "0.2", "--perceptual-weight", "0.1",
                 "--prompt-mode", "transcribe", "--seed", str(seed), "--seed-codec-per-clip",
                 "--music", MUSIC, "--output-dir", str(steal_dir(seed, sh)), "--commands", cmd],
                env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        wav = steal_dir(seed, sh) / "audio" / f"{MUSIC}__{cmd}.wav"
        rec = steal_dir(seed, sh) / "qwen25_omni" / f"{MUSIC}__{cmd}" / "record.json"
        if rc == 0 and wav.exists() and rec.exists():
            cl.mkdir(exist_ok=True); (cl / "done").touch(); fails = 0
            publish(seed, sh, cmd, name)
            log(f"done  seed{seed}/sh{sh} cmd={cmd} in {(time.time()-t0)/60:.1f} min", name)
        else:
            shutil.rmtree(cl, ignore_errors=True); fails += 1
            log(f"FAIL  seed{seed}/sh{sh} cmd={cmd} rc={rc} (claim released; see {clog})", name)
            if fails >= 2:
                log("two consecutive failures; exiting", name); return


# ------------------------------------------------------------ supervisor
def sctl(*args):
    r = subprocess.run(["scontrol", *args], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def supervisor(a):
    name = "supervisor"
    STATE.mkdir(exist_ok=True)
    lock = STATE / "supervisor.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        sys.exit(f"{lock} exists: another supervisor is running (remove it if not)")
    (lock / "pid").write_text(f"{os.uname().nodename} {os.getpid()}\n")
    log("=== supervisor v2 (requeue) start ===", name)
    last_rq = {}
    try:
        while True:
            left = 0
            for s in TRACKS:
                craft, _ = TRACKS[s]
                for h in range(4):
                    v = shard_view(s, h)
                    task = f"{craft}_{h}"
                    if not v["st"]:
                        if len(v["M"]) != 56 and not (STATE / f"warn_{s}_{h}").exists():
                            (STATE / f"warn_{s}_{h}").touch()
                            log(f"!! seed{s}/sh{h}: craft task left queue with {len(v['M'])}/56 "
                                f"published wavs; eval dependency may be dead -- check", name)
                        continue
                    left += 1
                    if v["st"] != "RUNNING" or not v["cur"]:
                        continue
                    stolen = {c for c, st in v["C"].items() if st == "done"} | v["H"]
                    if v["cur"] in stolen and v["cur"] in v["M"]:
                        if time.time() - last_rq.get(task, 0) < 600:
                            continue
                        r = subprocess.run(["scontrol", "requeue", task], capture_output=True, text=True)
                        last_rq[task] = time.time()
                        log(f"seed{s}/sh{h}: SLURM started already-published {v['cur']} "
                            f"({len(v['M'])}/56 published) -> requeue {task} rc={r.returncode} "
                            f"{(r.stdout + r.stderr).strip()}", name)
            if left == 0:
                log("=== every craft task has left the queue; supervisor exits ===", name)
                return
            time.sleep(a.poll)
    finally:
        shutil.rmtree(lock, ignore_errors=True)


def status(a):
    print(f"{'shard':<11}{'slurm':<9}{'at':>4}{'main':>6}{'stolen':>8}{'active':>8}{'uncov':>7}{'steal':>7}  merged")
    for s in TRACKS:
        for h in range(4):
            v = shard_view(s, h)
            act = sum(st == "active" for st in v["C"].values())
            print(f"s{s}/sh{h:<6}{v['st'] or '-':<9}{v['s_idx'] if v['s_idx'] > -1000 else '-':>4}"
                  f"{len(v['M']):>6}{len(v['T']):>8}{act:>8}{len(v['uncovered']):>7}"
                  f"{len(v['stealable']):>7}  {v['done']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    w = sub.add_parser("worker"); w.add_argument("--gpu", default="0")
    s = sub.add_parser("supervisor"); s.add_argument("--poll", type=int, default=30)
    sub.add_parser("status")
    a = p.parse_args()
    {"worker": worker, "supervisor": supervisor, "status": status}[a.mode](a)
