#!/bin/bash
# One-pass eval-ahead on SLURM: one (seed, victim) per array task, 0-11.
# Reuses eval_ahead_e3phi.run_batch (stages only fully crafted clips, copies
# records into <main shard bundle>/<OUT>/), with a private staging dir so it
# cannot collide with the local eval_ahead_e3phi.py loop.
#SBATCH --job-name=e3phi_evahead
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|h100|a40
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd ${REPO_ROOT}
IDX="${SLURM_ARRAY_TASK_ID:?submit array 0-11}"
exec ${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python -u - "$IDX" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "ea", "0_all_combined/n50_latent_vs_waveform/eval_ahead_e3phi.py")
ea = importlib.util.module_from_spec(spec); spec.loader.exec_module(ea)
ea.STAGE = ea.STATE / "eval_stage_slurm"
ea.GPU = __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "0")
ea.MIN_FREE_MIB = 0
idx = int(sys.argv[1])
seed = [7, 42][idx // 6]
model, out = ea.VICTIMS[idx % 6]
ready = ea.ready_clips(seed)
need = sorted(c for c in ready if not ea.cached_ok(
    ea.bundle(seed, ea.SHARD_OF[c]) / out / f"{ea.MUSIC}__{c}" / "record.json"))
ea.log(f"[slurm {idx}] seed{seed} {out}: {len(need)} to eval ({len(ready)} ready)")
if need:
    rc = ea.run_batch(seed, model, out, need, ready, False)
    sys.exit(1 if rc else 0)
PY
