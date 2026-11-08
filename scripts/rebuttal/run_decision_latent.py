"""S1 decision-flip pilot: explicit optimization seed, fixed development cohort."""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import config
import torch
from robust_latent_attack import RobustLatentCodecAttacker
from music_carrier import load_music
from reviewer_controls.run_attack import load_config, partition_pairs, duration_s, stable_id, seed_all
from reviewer_controls.common import write_pcm16_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--split', choices=['development', 'evaluation'], default='development')
    args = ap.parse_args()
    cfg = load_config(ROOT / 'scripts/reviewer_controls/config.json')
    pairs = partition_pairs(cfg, 's1', args.split, False)
    assert len(pairs) == (10 if args.split == 'development' else 40)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # A local warm-start and the queued batch may reach this cell together.
    # Serialize them before inspecting or writing its completion manifest.
    cell_lock = (args.output / 'cell.lock').open('w')
    fcntl.flock(cell_lock, fcntl.LOCK_EX)
    manifest = args.output / 'manifest.jsonl'
    signature = dict(alpha=args.alpha, epsilon=1.0, steps=1000, seed=args.seed,
                     checkpoint='last iterate', warmup_ratio=0.3, perceptual_weight=0,
                     config_sha256=hashlib.sha256((ROOT/'scripts/reviewer_controls/config.json').read_bytes()).hexdigest())
    if args.split != 'development':
        signature['split'] = args.split
    run_config = args.output/'run_config.json'
    if run_config.exists() and json.loads(run_config.read_text()) != signature:
        raise RuntimeError('Configuration differs from previous cell')
    run_config.write_text(json.dumps(signature, indent=2)+'\n')
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if not Path(row['adv_path']).is_file():raise RuntimeError('Missing completed audio')
            if row['example_id'] in done:raise RuntimeError('Duplicate completed example')
            done.add(row['example_id'])
    if len(done) == len(pairs):return
    seed_all(args.seed)
    attacker = RobustLatentCodecAttacker(target_model='qwen2_audio', eps=1, alpha=args.alpha,
        perceptual_weight=0, channel_mode='multi_bitrate', warmup_ratio=0.3, n_eot_samples=1,
        multi_bitrate_kbps=[16,24,32,64,128], verbose=False)
    attacker.target_model.model.requires_grad_(False)
    attacker.codec.model.requires_grad_(False)
    for index, pair in pairs:
        eid = stable_id('s1', index, pair['carrier'], pair['target'])
        if eid in done:continue
        seed_all(args.seed)
        carrier = ROOT/cfg['scenarios']['s1']['carrier_root']/pair['carrier']
        audio = load_music(str(carrier), duration=duration_s(carrier))
        start = time.perf_counter()
        result = attacker.attack(audio, pair['target'], steps=1000, check_every=0,
                                prompt=cfg['scenarios']['s1']['prompt'], music_name=carrier.stem)
        adv = result.adversarial_wav.detach().squeeze().float().cpu().numpy()
        ref = audio.squeeze().numpy()
        if len(ref) != len(adv):raise RuntimeError('Audio length mismatch')
        if not torch.isfinite(torch.as_tensor(adv)).all():raise RuntimeError('Nonfinite output')
        metrics = write_pcm16_pair(args.output/'audio', eid, ref, adv, 24000)
        row = dict(example_id=eid, scenario='s1', split=args.split, pair_index=index,
            method='latent_final_iterate', carrier=pair['carrier'], target=pair['target'],
            prompt=cfg['scenarios']['s1']['prompt'], epsilon=1.0, alpha=args.alpha,
            alpha_over_epsilon=args.alpha, seed=args.seed, steps=1000,
            checkpoint_rule='last iterate; check_every=0', elapsed_s=time.perf_counter()-start,
            carrier_sha256=hashlib.sha256(carrier.read_bytes()).hexdigest(),
            in_memory_attack_output=result.adversarial_output, **metrics)
        with manifest.open('a') as f:
            f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        print(f'{eid}: {row["elapsed_s"]:.1f}s, SNR {row["snr_db"]:.2f}', flush=True)


if __name__ == '__main__':main()
